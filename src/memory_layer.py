"""Production hot-path memory layer integrating BM25, dense vectors, graph traversal, and async worker queues."""

from __future__ import annotations

from datetime import datetime
import math
from typing import Dict, List, Optional, Sequence

from src.indexes import Embedder, HashingNgramEmbedder, bm25_search_sqlite
from src.llm_adapters import LLMJudge, LocalHeuristicJudge
from src.models import (
    CandidateClaim,
    Claim,
    ClaimStatus,
    EdgeType,
    JobType,
    RecalledClaim,
    Scope,
    TrustTier,
    utc_now,
)
from src.storage import ClaimRepository
from src.worker import SleepTimeWorker


class AgentMemoryLayer:
    """High-level interface sitting between headless agent runs and the Epistemic Provenance Graph."""

    def __init__(
        self,
        db_path: str = ":memory:",
        embedder: Optional[Embedder] = None,
        judge: Optional[LLMJudge] = None,
        max_active_per_org: int = 10_000,
        k_anonymity_threshold: int = 10,
    ) -> None:
        self.embedder = embedder or HashingNgramEmbedder()
        self.judge = judge or LocalHeuristicJudge()
        self.repo = ClaimRepository(db_path=db_path, embedder=self.embedder)
        self.worker = SleepTimeWorker(
            repo=self.repo,
            judge=self.judge,
            max_active_per_org=max_active_per_org,
            k_anonymity_threshold=k_anonymity_threshold,
        )

    def close(self) -> None:
        self.worker.stop()
        self.repo.close()

    def register_verified_domain(self, org_id: str, domain: str) -> None:
        self.repo.register_verified_domain(org_id, domain)

    def recall(
        self,
        org_id: str,
        query_text: str,
        user_id: Optional[str] = None,
        token_budget: int = 4000,
        now: Optional[datetime] = None,
    ) -> List[RecalledClaim]:
        """Zero-LLM critical-path hybrid recall (BM25 + vector + 1-hop graph) packed <= 4k tokens."""
        now = now or utc_now()
        self.repo.sync_vector_cache()
        bm25_hits = bm25_search_sqlite(self.repo.conn, org_id=org_id, query_text=query_text, limit=50)
        q_vec = self.embedder.embed(query_text)
        vec_hits = self.repo.vector_index.search([org_id, "GLOBAL"], q_vec, top_k=50)

        candidate_ids = list(dict.fromkeys([*bm25_hits.keys(), *vec_hits.keys()]))
        if not candidate_ids:
            return []
        claims_map = self.repo.get_claims_batch(candidate_ids)

        ranked: List[RecalledClaim] = []
        for cid in candidate_ids:
            c = claims_map.get(cid)
            if c is None:
                continue
            if c.scope != Scope.GLOBAL and c.org_id != org_id:
                continue
            if c.scope == Scope.USER and (user_id is None or c.user_id != user_id):
                continue
            if c.status not in (ClaimStatus.ACTIVE, ClaimStatus.CONTESTED):
                continue

            bm25_s = bm25_hits.get(cid, 0.0)
            vec_s = vec_hits.get(cid, 0.0)
            sim = 0.45 * bm25_s + 0.55 * vec_s
            days = max((now - c.last_used).total_seconds() / 86400.0, 0.0)
            final_score = sim * c.confidence * math.exp(-0.02 * days)

            # 1-hop graph lookup for unverified CONTESTED_BY hints
            hints: List[str] = []
            if c.status == ClaimStatus.CONTESTED:
                hint_ids = self.repo.get_neighbors(c.id, EdgeType.CONTESTED_BY, direction="out")
                hint_map = self.repo.get_claims_batch(hint_ids)
                hints = [
                    h.text for h in hint_map.values() if h.status == ClaimStatus.QUARANTINED
                ]

            ranked.append(
                RecalledClaim(
                    claim=c,
                    score=final_score,
                    bm25_score=bm25_s,
                    vector_score=vec_s,
                    contested_hints=hints,
                )
            )

        ranked.sort(key=lambda r: r.score, reverse=True)
        packed: List[RecalledClaim] = []
        used_tokens = 0
        for item in ranked:
            hint_words = sum(len(h.split()) + 4 for h in item.contested_hints)
            est_tokens = max(12, int((len(item.claim.text.split()) + hint_words + 10) * 1.35))
            if used_tokens + est_tokens > token_budget:
                continue
            used_tokens += est_tokens
            packed.append(item)
        return packed

    def stage_run_claims(
        self,
        org_id: str,
        run_id: str,
        candidates: Sequence[CandidateClaim],
        tool_logs: Dict[str, TrustTier],
    ) -> List[int]:
        """Non-blocking staging of <=3 candidate claims onto the async worker queue (0 LLM calls)."""
        serialized_logs = {sid: int(tier) for sid, tier in tool_logs.items()}
        job_ids: List[int] = []
        for cand in list(candidates)[:3]:
            if cand.scope == Scope.TASK or not cand.source_step_ids:
                continue
            payload = {
                "run_id": run_id,
                "scope": cand.scope.value,
                "entity_key": cand.entity_key,
                "text": cand.text,
                "source_step_ids": list(cand.source_step_ids),
                "derived_from": list(cand.derived_from),
                "user_id": cand.user_id,
                "public_artifact_signature": cand.public_artifact_signature,
                "tool_logs": serialized_logs,
            }
            job_ids.append(self.repo.enqueue_job(JobType.STAGE_WRITE, org_id, payload))
        return job_ids

    def submit_human_feedback(
        self,
        org_id: str,
        run_id: str,
        report_text: str,
        human_note: str,
        recalled_claim_ids: Sequence[str],
    ) -> int:
        """Enqueue delayed 2% human feedback for async attribution and JTMS rollback."""
        return self.repo.enqueue_job(
            JobType.FEEDBACK_ROLLBACK,
            org_id,
            {
                "run_id": run_id,
                "report_text": report_text,
                "human_note": human_note,
                "recalled_claim_ids": list(recalled_claim_ids),
            },
        )

    def schedule_decay_compaction(self, org_id: str) -> int:
        return self.repo.enqueue_job(JobType.DECAY_COMPACT, org_id, {})

    def can_trigger_external_email(self, org_id: str, claim_id: str, email: str) -> bool:
        """Deterministic IFC action sink-gate blocking quarantined/untrusted external domains."""
        claim = self.repo.get_claim(claim_id)
        if claim is None or claim.org_id != org_id:
            return False
        if claim.trust_tier == TrustTier.T3_UNTRUSTED or claim.status == ClaimStatus.QUARANTINED:
            return False
        domain = email.split("@")[-1].lower()
        return self.repo.is_verified_domain(org_id, domain) or claim.trust_tier == TrustTier.T0_HUMAN

    @staticmethod
    def format_context_block(recalled: Sequence[RecalledClaim]) -> str:
        """Render recalled claims with [id] tags so downstream outputs can cite derived_from[]."""
        lines: List[str] = []
        for item in recalled:
            c = item.claim
            tag = f"[{c.id}] ({c.scope.value}, {c.status.value}, conf={c.confidence:.2f})"
            line = f"{tag} {c.entity_key}: {c.text}"
            if item.contested_hints:
                hints_str = "; ".join(item.contested_hints)
                line += f" | [UNVERIFIED CONTEST: {hints_str}]"
            lines.append(line)
        return "\n".join(lines)
