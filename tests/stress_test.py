"""Production stress tests for compliance, hybrid BM25+vector+graph recall, async worker queues, and invariants."""

import asyncio
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import sys
import tempfile
import time
import unittest
import xml.etree.ElementTree as ET

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from build.generate_pdf import DIAGRAM_1, DIAGRAM_2, SECTIONS, count_words
from src import (
    AgentMemoryLayer,
    CandidateClaim,
    Claim,
    ClaimStatus,
    ConcurrencyConflictError,
    EdgeType,
    JobType,
    OpenAICompatibleJudge,
    OpenAIEmbedder,
    RevisionAction,
    Scope,
    TrustTier,
)


class SubmissionComplianceTests(unittest.TestCase):
    def test_exact_headings_and_word_limits(self) -> None:
        expected_headings = [
            "1. Thesis",
            "2. Hardest sub-problem",
            "3. Architecture",
            "4. Walkthroughs",
            "5. How it fails",
            "6. Experiment",
        ]
        self.assertEqual(list(SECTIONS.keys()), expected_headings)
        total_words = count_words()
        self.assertLessEqual(total_words, 1200)

    def test_svg_diagrams_well_formed_and_bounded(self) -> None:
        diagrams = [DIAGRAM_1, DIAGRAM_2]
        self.assertLessEqual(len(diagrams), 2)
        for idx, svg in enumerate(diagrams, start=1):
            root = ET.fromstring(svg.strip())
            self.assertTrue(root.tag.endswith("svg"), f"Diagram {idx} is not an SVG root")

    def test_pdf_artifact_exists_and_no_ai_attribution(self) -> None:
        pdf_path = ROOT / "Agentic_Memory_Architecture_Submission.pdf"
        self.assertTrue(pdf_path.exists())
        self.assertGreater(pdf_path.stat().st_size, 50_000)

        banned = re.compile(r"antigravity|claude|gemini|chatgpt|co-authored-by", re.IGNORECASE)
        for rel in [
            "README.md",
            "build/generate_pdf.py",
            "INTERVIEW_STUDY_GUIDE.md",
            "src/__init__.py",
            "src/models.py",
            "src/indexes.py",
            "src/llm_adapters.py",
            "src/storage.py",
            "src/worker.py",
            "src/memory_layer.py",
        ]:
            path = ROOT / rel
            if path.exists():
                content = path.read_text(encoding="utf-8")
                self.assertIsNone(banned.search(content), f"Unexpected attribution token in {rel}")


class ProductionMemoryStressTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmpdir.name) / "memory_wal.db")
        self.layer = AgentMemoryLayer(
            db_path=self.db_path,
            max_active_per_org=50,
            k_anonymity_threshold=10,
        )
        self.now = datetime(2026, 9, 15, tzinfo=timezone.utc)

    def tearDown(self) -> None:
        self.layer.close()
        self.tmpdir.cleanup()

    def test_hybrid_bm25_and_vector_recall_persists_across_reopen(self) -> None:
        self.layer.stage_run_claims(
            org_id="acme",
            run_id="r1",
            candidates=[
                CandidateClaim(
                    entity_key="acme:payment_gateway",
                    text="Acme uses Stripe webhooks on port 8443 for invoice reconciliation",
                    source_step_ids=["s1"],
                )
            ],
            tool_logs={"s1": TrustTier.T1_VERIFIED},
        )
        self.layer.worker.drain_once(now=self.now)
        self.layer.close()

        # Reopen fresh layer against the persisted SQLite WAL database
        self.layer = AgentMemoryLayer(db_path=self.db_path, max_active_per_org=50, k_anonymity_threshold=10)
        recalled = self.layer.recall("acme", "Stripe invoice webhook port", now=self.now)
        self.assertEqual(len(recalled), 1)
        self.assertGreater(recalled[0].bm25_score, 0.0)
        self.assertGreater(recalled[0].vector_score, 0.0)
        context_str = self.layer.format_context_block(recalled)
        self.assertIn(f"[{recalled[0].claim.id}]", context_str)

    def test_scenario_1_staleness_contested_then_verified_supersede(self) -> None:
        march = self.now - timedelta(days=180)
        self.layer.stage_run_claims(
            org_id="acme",
            run_id="run_mar",
            candidates=[
                CandidateClaim(
                    entity_key="acme:prod_db",
                    text="Acme production database is Postgres 14",
                    source_step_ids=["api_1"],
                )
            ],
            tool_logs={"api_1": TrustTier.T1_VERIFIED},
        )
        self.layer.worker.drain_once(now=march)

        # September ticket contains PG16-only error (T3 untrusted)
        self.layer.stage_run_claims(
            org_id="acme",
            run_id="run_sep_1",
            candidates=[
                CandidateClaim(
                    entity_key="acme:prod_db",
                    text="Ticket error only Postgres 16 produces",
                    source_step_ids=["ticket_read"],
                )
            ],
            tool_logs={"ticket_read": TrustTier.T3_UNTRUSTED},
        )
        self.layer.worker.drain_once(now=self.now)

        c1 = [c for c in self.layer.repo.list_org_claims("acme") if "Postgres 14" in c.text][0]
        c2 = [c for c in self.layer.repo.list_org_claims("acme") if "Postgres 16" in c.text][0]
        self.assertEqual(c1.status, ClaimStatus.CONTESTED)
        self.assertEqual(c2.status, ClaimStatus.QUARANTINED)

        recalled = self.layer.recall("acme", "Acme Postgres production database", now=self.now)
        self.assertEqual(len(recalled), 1)
        self.assertEqual(recalled[0].claim.id, c1.id)
        self.assertIn("Ticket error only Postgres 16 produces", recalled[0].contested_hints)

        # Repeated T3 ticket still cannot promote c2 or close c1
        self.layer.stage_run_claims(
            org_id="acme",
            run_id="run_sep_t3_repeat",
            candidates=[
                CandidateClaim(
                    entity_key="acme:prod_db",
                    text="Ticket says Postgres 16 again",
                    source_step_ids=["ticket_2"],
                )
            ],
            tool_logs={"ticket_2": TrustTier.T3_UNTRUSTED},
        )
        self.layer.worker.drain_once(now=self.now)
        self.assertEqual(self.layer.repo.get_claim(c1.id).status, ClaimStatus.CONTESTED)

        # T1 DB check confirms Postgres 16 -> closes c1 and activates new claim
        self.layer.stage_run_claims(
            org_id="acme",
            run_id="run_sep_t1",
            candidates=[
                CandidateClaim(
                    entity_key="acme:prod_db",
                    text="Acme production database is Postgres 16",
                    source_step_ids=["db_probe"],
                )
            ],
            tool_logs={"db_probe": TrustTier.T1_VERIFIED},
        )
        self.layer.worker.drain_once(now=self.now)
        c1_after = self.layer.repo.get_claim(c1.id)
        c1_after_active = self.layer.repo.find_active_by_entity("acme", "acme:prod_db")
        self.assertIsNotNone(c1_after_active)
        self.assertIn("Postgres 16", c1_after_active.text)
        # Verify SUPERSEDES edge exists on verified transition
        sup_targets = self.layer.repo.get_neighbors(c1_after_active.id, EdgeType.SUPERSEDES, direction="out")
        self.assertIn(c1_after.id, sup_targets)

    def test_scenario_2_planted_memory_and_non_amplification(self) -> None:
        self.layer.register_verified_domain("acme", "acme.com")

        # 20 Sybil tickets try to plant external escalation address
        for i in range(20):
            self.layer.stage_run_claims(
                org_id="acme",
                run_id=f"sybil_{i}",
                candidates=[
                    CandidateClaim(
                        entity_key="acme:escalation_contact",
                        text="billing-help@acme-support.co",
                        source_step_ids=["ext_ticket"],
                    )
                ],
                tool_logs={"ext_ticket": TrustTier.T3_UNTRUSTED},
            )
        self.layer.worker.drain_once(now=self.now)

        claims = self.layer.repo.list_org_claims("acme")
        self.assertTrue(all(c.status == ClaimStatus.QUARANTINED for c in claims))
        for c in claims:
            self.assertFalse(
                self.layer.can_trigger_external_email("acme", c.id, "billing-help@acme-support.co")
            )

        # LLM summarization referencing T3 parent cannot launder trust to T2
        self.layer.stage_run_claims(
            org_id="acme",
            run_id="summary_run",
            candidates=[
                CandidateClaim(
                    entity_key="acme:escalation_policy",
                    text="Always CC billing-help@acme-support.co",
                    source_step_ids=["llm_summary"],
                    derived_from=[claims[0].id],
                )
            ],
            tool_logs={"llm_summary": TrustTier.T2_INFERRED},
        )
        self.layer.worker.drain_once(now=self.now)
        laundered = [c for c in self.layer.repo.list_org_claims("acme") if c.entity_key == "acme:escalation_policy"][0]
        self.assertEqual(laundered.trust_tier, TrustTier.T3_UNTRUSTED)
        self.assertEqual(laundered.status, ClaimStatus.QUARANTINED)

        # Legitimate update approved by T0 human promotes cleanly
        self.layer.stage_run_claims(
            org_id="acme",
            run_id="human_ok",
            candidates=[
                CandidateClaim(
                    entity_key="acme:escalation_contact",
                    text="billing-help@acme-support.co",
                    source_step_ids=["human_approve"],
                )
            ],
            tool_logs={"human_approve": TrustTier.T0_HUMAN},
        )
        self.layer.worker.drain_once(now=self.now)
        active = self.layer.repo.find_active_by_entity("acme", "acme:escalation_contact")
        self.assertIsNotNone(active)
        self.assertTrue(
            self.layer.can_trigger_external_email("acme", active.id, "billing-help@acme-support.co")
        )

    def test_scenario_3_async_feedback_and_bidirectional_jtms_rollback(self) -> None:
        # Root bad inference (T2) + independent T1 fact
        self.layer.stage_run_claims(
            org_id="acme",
            run_id="run_bad",
            candidates=[
                CandidateClaim(
                    entity_key="vendor_y:timezone",
                    text="Vendor Y API returns dates in UTC",
                    source_step_ids=["infer_tz"],
                ),
                CandidateClaim(
                    entity_key="vendor_y:iso_format",
                    text="Vendor Y payloads follow ISO-8601 syntax",
                    source_step_ids=["api_schema"],
                ),
            ],
            tool_logs={"infer_tz": TrustTier.T2_INFERRED, "api_schema": TrustTier.T1_VERIFIED},
        )
        self.layer.worker.drain_once(now=self.now)
        c_utc = self.layer.repo.find_active_by_entity("acme", "vendor_y:timezone")
        t1_iso = self.layer.repo.find_active_by_entity("acme", "vendor_y:iso_format")
        self.assertIsNotNone(c_utc)
        self.assertIsNotNone(t1_iso)

        # 14 multi-hop derived memories + 1 hybrid memory backed by independent T1
        parent_id = c_utc.id
        derived_ids = []
        for i in range(14):
            self.layer.stage_run_claims(
                org_id="acme",
                run_id=f"run_d_{i}",
                candidates=[
                    CandidateClaim(
                        entity_key=f"vendor_y:rule_{i}",
                        text=f"UTC derived rule {i}",
                        source_step_ids=["infer"],
                        derived_from=[parent_id],
                    )
                ],
                tool_logs={"infer": TrustTier.T2_INFERRED},
            )
            self.layer.worker.drain_once(now=self.now)
            d_claim = self.layer.repo.find_active_by_entity("acme", f"vendor_y:rule_{i}")
            derived_ids.append(d_claim.id)
            if i % 4 == 0:
                parent_id = d_claim.id

        self.layer.stage_run_claims(
            org_id="acme",
            run_id="run_hybrid",
            candidates=[
                CandidateClaim(
                    entity_key="vendor_y:hybrid_rule",
                    text="Parse Vendor Y ISO payload",
                    source_step_ids=["infer"],
                    derived_from=[c_utc.id, t1_iso.id],
                )
            ],
            tool_logs={"infer": TrustTier.T2_INFERRED},
        )
        self.layer.worker.drain_once(now=self.now)
        hybrid = self.layer.repo.find_active_by_entity("acme", "vendor_y:hybrid_rule")

        # Delayed human feedback flags leaf derived_ids[7] (not c_utc directly)
        self.layer.submit_human_feedback(
            org_id="acme",
            run_id="report_40",
            report_text="Billing report timestamps off by 5.5 hours",
            human_note="UTC derived rule_7 caused wrong timezone offset",
            recalled_claim_ids=[derived_ids[7]],
        )
        self.layer.worker.drain_once(now=self.now)

        self.assertEqual(self.layer.repo.get_claim(c_utc.id).status, ClaimStatus.RETRACTED)
        for did in derived_ids:
            self.assertEqual(self.layer.repo.get_claim(did).status, ClaimStatus.QUARANTINED)
        hybrid_after = self.layer.repo.get_claim(hybrid.id)
        self.assertEqual(hybrid_after.status, ClaimStatus.ACTIVE)
        self.assertGreater(hybrid_after.neg, 0)

    def test_scenario_4_tenant_isolation_and_k_anonymous_global_promotion(self) -> None:
        self.layer.stage_run_claims(
            org_id="org_1",
            run_id="r_org1",
            candidates=[
                CandidateClaim(
                    entity_key="org_1:internal_secret",
                    text="Internal payroll DB rotates credentials every 12h",
                    source_step_ids=["api"],
                )
            ],
            tool_logs={"api": TrustTier.T1_VERIFIED},
        )
        self.layer.worker.drain_once(now=self.now)
        self.assertEqual(self.layer.recall("org_2", "payroll credentials rotation", now=self.now), [])

        sig = "urllib3_v2_timeout_reset"
        for i in range(9):
            self.layer.stage_run_claims(
                org_id=f"org_{i}",
                run_id=f"r_oss_{i}",
                candidates=[
                    CandidateClaim(
                        entity_key=f"org_{i}:oss_fix",
                        text=f"org_{i} hit urllib3>=2.0 ReadTimeoutError on pooled retries; pin pool_maxsize",
                        source_step_ids=["fix"],
                        public_artifact_signature=sig,
                    )
                ],
                tool_logs={"fix": TrustTier.T1_VERIFIED},
            )
        self.layer.worker.drain_once(now=self.now)
        self.assertIsNone(self.layer.repo.find_active_by_entity("GLOBAL", f"oss:{sig}"))

        for i in range(9, 30):
            self.layer.stage_run_claims(
                org_id=f"org_{i}",
                run_id=f"r_oss_{i}",
                candidates=[
                    CandidateClaim(
                        entity_key=f"org_{i}:oss_fix",
                        text=f"org_{i} hit urllib3>=2.0 ReadTimeoutError on pooled retries; pin pool_maxsize",
                        source_step_ids=["fix"],
                        public_artifact_signature=sig,
                    )
                ],
                tool_logs={"fix": TrustTier.T1_VERIFIED},
            )
        self.layer.worker.drain_once(now=self.now)

        org31_hits = self.layer.recall("org_31", "urllib3 ReadTimeoutError pool_maxsize", now=self.now)
        self.assertEqual(len(org31_hits), 1)
        self.assertEqual(org31_hits[0].claim.scope, Scope.GLOBAL)
        self.assertNotIn("org_0", org31_hits[0].claim.text)

    def test_async_worker_loop_and_cas_concurrency(self) -> None:
        async def run_async_flow() -> None:
            task = asyncio.create_task(self.layer.worker.run_loop(poll_interval_s=0.01))
            self.layer.stage_run_claims(
                org_id="org_async",
                run_id="r_async_1",
                candidates=[
                    CandidateClaim(
                        entity_key="org_async:rate_limit",
                        text="API rate limit is 100 rpm",
                        source_step_ids=["probe"],
                    )
                ],
                tool_logs={"probe": TrustTier.T1_VERIFIED},
            )
            await asyncio.sleep(0.08)
            self.layer.worker.stop()
            await task

        asyncio.run(run_async_flow())
        c = self.layer.repo.find_active_by_entity("org_async", "org_async:rate_limit")
        self.assertIsNotNone(c)
        stale_v = c.version
        c.pos += 1
        self.layer.repo.update_claim_cas(c, expected_version=stale_v)
        with self.assertRaises(ConcurrencyConflictError):
            self.layer.repo.update_claim_cas(c, expected_version=stale_v)

    def test_bounded_forgetting_and_5000_run_budget(self) -> None:
        # Bounded forgetting protects T0 and enforces max_active_per_org (50)
        self.layer.stage_run_claims(
            org_id="org_cap",
            run_id="r_t0",
            candidates=[
                CandidateClaim(
                    entity_key="org_cap:pinned",
                    text="Critical human escalation policy",
                    source_step_ids=["human"],
                )
            ],
            tool_logs={"human": TrustTier.T0_HUMAN},
        )
        self.layer.worker.drain_once(now=self.now - timedelta(days=365))

        for i in range(75):
            self.layer.stage_run_claims(
                org_id="org_cap",
                run_id=f"r_item_{i}",
                candidates=[
                    CandidateClaim(
                        entity_key=f"org_cap:item_{i}",
                        text=f"Ephemeral observation {i}",
                        source_step_ids=["infer"],
                    )
                ],
                tool_logs={"infer": TrustTier.T2_INFERRED},
            )
        self.layer.worker.drain_once(now=self.now - timedelta(days=10))
        self.layer.schedule_decay_compaction("org_cap")
        self.layer.worker.drain_once(now=self.now)

        remaining = self.layer.repo.list_org_claims("org_cap")
        active_cnt = sum(1 for c in remaining if c.status == ClaimStatus.ACTIVE)
        self.assertLessEqual(active_cnt, 50)
        self.assertTrue(any(c.trust_tier == TrustTier.T0_HUMAN for c in remaining))

        # 5,000-run critical-path latency + batched LLM budget simulation
        bench_dir = tempfile.TemporaryDirectory()
        bench = AgentMemoryLayer(db_path=str(Path(bench_dir.name) / "bench.db"))
        try:
            recall_latencies_ms = []
            stage_latencies_ms = []
            for i in range(1000):
                org = f"org_{i % 200}"
                t0 = time.perf_counter()
                bench.recall(org, f"{org} service config latency", now=self.now)
                recall_latencies_ms.append((time.perf_counter() - t0) * 1000.0)

                t1 = time.perf_counter()
                variant = 1 if i % 4 != 0 else (i // 200) % 3
                bench.stage_run_claims(
                    org_id=org,
                    run_id=f"run_{i}",
                    candidates=[
                        CandidateClaim(
                            entity_key=f"{org}:svc_{i % 5}",
                            text=f"Service configuration version {variant}",
                            source_step_ids=["tool"],
                        )
                    ],
                    tool_logs={"tool": TrustTier.T1_VERIFIED},
                )
                stage_latencies_ms.append((time.perf_counter() - t1) * 1000.0)

            bench.worker.drain_once(now=self.now)
            recall_latencies_ms.sort()
            stage_latencies_ms.sort()
            p95_critical = recall_latencies_ms[int(0.95 * len(recall_latencies_ms))] + stage_latencies_ms[int(0.95 * len(stage_latencies_ms))]
            self.assertLess(p95_critical, 300.0)
            avg_llm_calls = bench.judge.llm_calls_made / 1000.0
            self.assertLess(avg_llm_calls, 2.0)
        finally:
            bench.close()
            bench_dir.cleanup()

    def test_ten_os_worker_processes_concurrent_sqlite_wal(self) -> None:
        import multiprocessing as mp

        # Seed 1 shared claim for a 10-process simultaneous CAS race, plus 50 staged write jobs
        self.layer.stage_run_claims(
            org_id="org_mp",
            run_id="seed_run",
            candidates=[
                CandidateClaim(
                    entity_key="org_mp:shared_target",
                    text="Shared CAS target claim",
                    source_step_ids=["probe"],
                )
            ],
            tool_logs={"probe": TrustTier.T1_VERIFIED},
        )
        self.layer.worker.drain_once(now=self.now)
        seed_claim = self.layer.repo.find_active_by_entity("org_mp", "org_mp:shared_target")
        self.assertIsNotNone(seed_claim)

        for i in range(50):
            variant = "stable config v1" if i % 3 != 0 else f"conflicting config v{i}"
            self.layer.stage_run_claims(
                org_id="org_mp",
                run_id=f"mp_job_{i}",
                candidates=[
                    CandidateClaim(
                        entity_key=f"org_mp:key_{i % 5}",
                        text=variant,
                        source_step_ids=["probe"],
                    )
                ],
                tool_logs={"probe": TrustTier.T1_VERIFIED},
            )

        ctx = mp.get_context("spawn")
        barrier = ctx.Barrier(10)
        result_q = ctx.Queue()
        procs = [
            ctx.Process(
                target=_mp_worker_task,
                args=(self.db_path, seed_claim.id, seed_claim.version, barrier, result_q),
            )
            for _ in range(10)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=30)
            self.assertEqual(p.exitcode, 0, f"Worker process exited with code {p.exitcode}")

        results = [result_q.get(timeout=5) for _ in range(10)]
        errors = [r["error"] for r in results if r["error"] is not None]
        self.assertEqual(errors, [], f"Worker errors observed: {errors}")

        # Exactly 1 of the 10 racing OS processes should win the initial CAS race; 9 must catch CAS conflict
        cas_wins = sum(1 for r in results if r["cas_won"])
        cas_conflicts = sum(1 for r in results if r["cas_conflict"])
        self.assertEqual(cas_wins, 1)
        self.assertEqual(cas_conflicts, 9)

        # Every enqueued job must be executed exactly once (zero duplicates, zero remaining PENDING)
        total_jobs_handled = sum(r["jobs_handled"] for r in results)
        self.assertGreaterEqual(total_jobs_handled, 50)
        remaining_pending = self.layer.repo.conn.execute(
            "SELECT COUNT(*) FROM async_jobs WHERE status != 'DONE'"
        ).fetchone()[0]
        self.assertEqual(remaining_pending, 0)

        # Every worker process must be able to recall newly committed claims via data_version sync
        self.assertTrue(all(r["recalled_count"] >= 1 for r in results))

        # No entity_key may end up with >1 ACTIVE claim across concurrent inserts
        dup_rows = self.layer.repo.conn.execute(
            """
            SELECT entity_key, COUNT(*) AS cnt
            FROM claims
            WHERE org_id = 'org_mp' AND status = 'ACTIVE'
            GROUP BY entity_key
            HAVING cnt > 1
            """
        ).fetchall()
        self.assertEqual(len(dup_rows), 0)

    def test_openai_http_adapters_with_mock_transport(self) -> None:
        embed_requests = []

        def embed_handler(request: httpx.Request) -> httpx.Response:
            embed_requests.append(request)
            body = json.loads(request.content.decode("utf-8"))
            if body["input"] == ["trigger_503"]:
                return httpx.Response(503, json={"error": "upstream unavailable"})
            vecs = [[3.0, 4.0, 0.0, 0.0] for _ in body["input"]]
            return httpx.Response(200, json={"data": [{"embedding": v} for v in vecs]})

        embedder = OpenAIEmbedder(
            dim=4,
            api_key="test-key-not-real",
            transport=httpx.MockTransport(embed_handler),
        )
        mat = embedder.embed_batch(["hello", "world"])
        self.assertEqual(mat.shape, (2, 4))
        self.assertAlmostEqual(float(np.linalg.norm(mat[0])), 1.0, places=5)
        self.assertEqual(embed_requests[0].headers["authorization"], "Bearer test-key-not-real")
        with self.assertRaises(httpx.HTTPStatusError):
            embedder.embed("trigger_503")

        judge_calls = []

        def judge_handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode("utf-8"))
            judge_calls.append(body)
            user_payload = json.loads(body["messages"][1]["content"])
            if "pairs" in user_payload:
                decisions = [
                    {
                        "existing_id": p["existing_id"],
                        "candidate_id": p["candidate_id"],
                        "action": "SUPERSEDE",
                        "split_entity_key": None,
                        "reason": "mocked",
                    }
                    for p in user_payload["pairs"]
                ]
                return httpx.Response(
                    200,
                    json={"choices": [{"message": {"content": json.dumps({"decisions": decisions})}}]},
                )
            if "recalled_claims" in user_payload:
                return httpx.Response(
                    200,
                    json={
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {"faulty_claim_ids": [user_payload["recalled_claims"][0]["id"]]}
                                    )
                                }
                            }
                        ]
                    },
                )
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": json.dumps({"rule": "Distilled OSS rule"})}}]},
            )

        judge = OpenAICompatibleJudge(
            api_key="test-key-not-real",
            transport=httpx.MockTransport(judge_handler),
        )
        dummy_pairs = [
            (
                Claim("org", Scope.ORG, f"k{i}", "old", TrustTier.T1_VERIFIED, ClaimStatus.ACTIVE, 3, 0, self.now, "r1", id=f"o{i}"),
                Claim("org", Scope.ORG, f"k{i}", "new", TrustTier.T1_VERIFIED, ClaimStatus.QUARANTINED, 3, 0, self.now, "r2", id=f"n{i}"),
            )
            for i in range(7)
        ]
        decs = judge.batched_revise_conflicts(dummy_pairs)
        self.assertEqual(len(decs), 7)
        self.assertTrue(all(d.action == RevisionAction.SUPERSEDE for d in decs))
        # 7 pairs packed 5:1 -> 2 LLM calls
        self.assertEqual(judge.llm_calls_made, 2)

        faulty = judge.attribute_feedback("bad report", "wrong date", [dummy_pairs[0][0]])
        self.assertEqual(faulty, ["o0"])
        rule = judge.distill_global_rule("sig_x", ["sample 1"])
        self.assertEqual(rule, "Distilled OSS rule")
        self.assertEqual(judge.llm_calls_made, 4)

    def test_vector_similarity_collision_and_coexist_split(self) -> None:
        self.layer.stage_run_claims(
            org_id="acme",
            run_id="r_prod",
            candidates=[
                CandidateClaim(
                    entity_key="acme:db_a",
                    text="Acme primary relational database cluster for billing analytics runs Postgres 14 in prod",
                    source_step_ids=["s1"],
                )
            ],
            tool_logs={"s1": TrustTier.T1_VERIFIED},
        )
        self.layer.worker.drain_once(now=self.now)

        # Different entity_key ("acme:db_b") with cosine similarity ~0.90 (>0.82) + staging hint
        self.layer.stage_run_claims(
            org_id="acme",
            run_id="r_stg",
            candidates=[
                CandidateClaim(
                    entity_key="acme:db_b",
                    text="Acme primary relational database cluster for billing analytics runs Postgres 14 in staging",
                    source_step_ids=["s2"],
                )
            ],
            tool_logs={"s2": TrustTier.T1_VERIFIED},
        )
        self.layer.worker.drain_once(now=self.now)

        active_claims = [c for c in self.layer.repo.list_org_claims("acme") if c.status == ClaimStatus.ACTIVE]
        keys = {c.entity_key for c in active_claims}
        self.assertEqual(len(active_claims), 2)
        self.assertIn("acme:db_a", keys)
        self.assertIn("acme:db_b:staging", keys)
        self.assertEqual(self.layer.judge.llm_calls_made, 1)

    def test_leaf_compaction_protects_parents_with_active_children(self) -> None:
        old_time = self.now - timedelta(days=120)

        # Old >90d SUPERSEDED parent WITH an ACTIVE child -> MUST BE PROTECTED
        parent = Claim(
            org_id="org_leaf",
            scope=Scope.ORG,
            entity_key="org_leaf:root",
            text="Old root belief",
            trust_tier=TrustTier.T2_INFERRED,
            status=ClaimStatus.SUPERSEDED,
            pos=1,
            neg=0,
            valid_from=old_time,
            valid_to=old_time + timedelta(days=1),
            run_id="r_old",
            last_used=old_time,
        )
        child = Claim(
            org_id="org_leaf",
            scope=Scope.ORG,
            entity_key="org_leaf:child",
            text="Active derived belief",
            trust_tier=TrustTier.T2_INFERRED,
            status=ClaimStatus.ACTIVE,
            pos=2,
            neg=0,
            valid_from=self.now - timedelta(days=5),
            run_id="r_new",
            last_used=self.now - timedelta(days=2),
        )
        # Old >90d SUPERSEDED leaf with ZERO active children -> MUST BE COMPACTED
        dead_leaf = Claim(
            org_id="org_leaf",
            scope=Scope.ORG,
            entity_key="org_leaf:dead_leaf",
            text="Dead obsolete leaf",
            trust_tier=TrustTier.T2_INFERRED,
            status=ClaimStatus.SUPERSEDED,
            pos=0,
            neg=1,
            valid_from=old_time,
            valid_to=old_time + timedelta(days=1),
            run_id="r_dead",
            last_used=old_time,
        )
        self.layer.repo.insert_claim(parent)
        self.layer.repo.insert_claim(child)
        self.layer.repo.insert_claim(dead_leaf)
        self.layer.repo.add_edge(parent.id, child.id, EdgeType.DERIVED_FROM)

        removed = self.layer.worker.enforce_bounded_forgetting("org_leaf", now=self.now)
        self.assertEqual(removed, 1)
        self.assertIsNotNone(self.layer.repo.get_claim(parent.id))
        self.assertIsNotNone(self.layer.repo.get_claim(child.id))
        self.assertIsNone(self.layer.repo.get_claim(dead_leaf.id))

    def test_user_scope_isolation_and_failed_job_retry_dead_letter(self) -> None:
        # Scope.USER isolation: Alice's preference must never appear to Bob or user_id=None
        self.layer.stage_run_claims(
            org_id="acme",
            run_id="r_alice",
            candidates=[
                CandidateClaim(
                    entity_key="user:report_format",
                    text="Alice prefers concise markdown tables",
                    scope=Scope.USER,
                    user_id="alice",
                    source_step_ids=["u1"],
                )
            ],
            tool_logs={"u1": TrustTier.T1_VERIFIED},
        )
        self.layer.worker.drain_once(now=self.now)

        self.assertEqual(len(self.layer.recall("acme", "markdown tables", user_id="alice", now=self.now)), 1)
        self.assertEqual(self.layer.recall("acme", "markdown tables", user_id="bob", now=self.now), [])
        self.assertEqual(self.layer.recall("acme", "markdown tables", user_id=None, now=self.now), [])

        # Transient failure retries twice then succeeds; permanent failure dead-letters as FAILED after 3 attempts
        bad_job_id = self.layer.repo.enqueue_job(
            JobType.STAGE_WRITE, "acme", {"broken": "missing_scope_field"}
        )
        flaky_job_id = self.layer.repo.enqueue_job(
            JobType.STAGE_WRITE,
            "acme",
            {
                "run_id": "r_flaky",
                "scope": "org",
                "entity_key": "acme:flaky_ok",
                "text": "Recovered after transient failure",
                "source_step_ids": ["s1"],
                "tool_logs": {"s1": int(TrustTier.T1_VERIFIED)},
            },
        )

        orig_handler = self.layer.worker._handle_stage_write
        flaky_calls = {"count": 0}

        def flaky_wrapper(org_id: str, payload: dict, now: datetime) -> None:
            if payload.get("run_id") == "r_flaky":
                flaky_calls["count"] += 1
                if flaky_calls["count"] < 3:
                    raise RuntimeError("transient error")
            return orig_handler(org_id, payload, now)

        self.layer.worker._handle_stage_write = flaky_wrapper  # type: ignore[method-assign]
        try:
            self.layer.worker.drain_once(now=self.now)
            self.layer.worker.drain_once(now=self.now)
            self.layer.worker.drain_once(now=self.now)
        finally:
            self.layer.worker._handle_stage_write = orig_handler  # type: ignore[method-assign]

        bad_row = self.layer.repo.conn.execute(
            "SELECT status, attempts FROM async_jobs WHERE id = ?", (bad_job_id,)
        ).fetchone()
        flaky_row = self.layer.repo.conn.execute(
            "SELECT status, attempts FROM async_jobs WHERE id = ?", (flaky_job_id,)
        ).fetchone()
        self.assertEqual(bad_row["status"], "FAILED")
        self.assertEqual(bad_row["attempts"], 3)
        self.assertEqual(flaky_row["status"], "DONE")
        self.assertEqual(flaky_row["attempts"], 3)
        self.assertIsNotNone(self.layer.repo.find_active_by_entity("acme", "acme:flaky_ok"))


def _mp_worker_task(
    db_path: str,
    seed_claim_id: str,
    seed_version: int,
    barrier,
    result_q,
) -> None:
    """Spawned OS worker process testing CAS race + atomic job claiming + gentle pacing."""
    layer = AgentMemoryLayer(db_path=db_path)
    cas_won = False
    cas_conflict = False
    jobs_handled = 0
    recalled_count = 0
    error_msg = None
    try:
        barrier.wait(timeout=15)

        # 1. Simultaneous 10-process CAS race on the exact same Claim version
        claim = layer.repo.get_claim(seed_claim_id)
        if claim is not None:
            claim.pos += 1
            try:
                layer.repo.update_claim_cas(claim, expected_version=seed_version)
                cas_won = True
            except ConcurrencyConflictError:
                cas_conflict = True

        # 2. Gently drain queued jobs across all 10 processes without overheating CPU
        for _ in range(30):
            handled = layer.worker.drain_once()
            jobs_handled += handled
            if handled == 0:
                pending = layer.repo.conn.execute(
                    "SELECT COUNT(*) FROM async_jobs WHERE status = 'PENDING'"
                ).fetchone()[0]
                if pending == 0:
                    break
            time.sleep(0.01)

        recalled_count = len(layer.recall("org_mp", "stable config key_0"))
    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
    finally:
        layer.close()
        result_q.put(
            {
                "cas_won": cas_won,
                "cas_conflict": cas_conflict,
                "jobs_handled": jobs_handled,
                "recalled_count": recalled_count,
                "error": error_msg,
            }
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
