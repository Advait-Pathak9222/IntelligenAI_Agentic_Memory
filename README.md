# Epistemic Provenance Graph — Agentic Memory for Headless Agents

Architecture specification, minimal reference engine, and automated stress-test suite for persistent memory in unsupervised, multi-tenant headless AI agents.

## Problem Constraints

A headless agent runs without human supervision on a task queue, making memory an **epistemic belief-maintenance problem** rather than a pure retrieval problem:

| Dimension | Constraint |
|---|---|
| **Throughput & Orgs** | ~5,000 runs/day across 200 strictly isolated customer organizations (2–50 tool calls/run) |
| **Feedback Sparsity** | ~2% of outputs receive human feedback, arriving days later; 98% receive none |
| **Untrusted Inputs** | External emails, web pages, uploads, and support tickets may contain planted instructions |
| **Context & Lifetime** | <= 4,000 tokens of memory per step; bounded storage over 2+ years |
| **Latency & Cost Caps** | <= 300 ms p95 added to the critical path; <= 2 extra LLM calls/run on average (including background jobs) |
| **Model Constraint** | Black-box LLM only; no fine-tuning or weight updates |

## Architecture Overview

The design separates a deterministic **Hot Path** (`0` LLM calls, `<= 150 ms` p95) from an asynchronous **Sleep-Time Batch Lane** (`~0.08–0.30` LLM calls/run avg). Every memory is stored as one structured **`Claim`** node inside an org-partitioned **Provenance DAG** linked by `DERIVED_FROM`, `CONTESTED_BY`, and `SUPERSEDES` edges.

### 1. Unified Data Model (`Claim`)

| Field | Role |
|---|---|
| `id, org_id, scope` | UUID, hard tenant partition key, and scope (`task` \| `user` \| `org` \| `global`) |
| `entity_key, text` | Normalized subject key (e.g., `acme:prod_db`) and atomic claim (`<= 40` tokens) |
| `trust_tier` | Information-Flow Control (IFC) integrity: `T0_HUMAN > T1_VERIFIED > T2_INFERRED > T3_UNTRUSTED` |
| `status, pos, neg` | `QUARANTINED \| ACTIVE \| CONTESTED \| SUPERSEDED \| RETRACTED`; confidence `(pos + 1) / (pos + neg + 2)` |
| `valid_from, valid_to` | Bi-temporal validity window (closed on supersession instead of destructive overwrite) |
| `derived_from[], run_id` | Cited parent claim IDs and origin run for causal rollback; `version` (CAS) and `last_used` for decay |

### 2. Core Mechanisms

* **Zero-LLM Critical Path (Recall & Write Staging):** Hybrid BM25 + vector + 1-hop graph lookup scores `ACTIVE` and `CONTESTED` claims by `sim * confidence * exp(-0.02 * days)` and packs `<= 4,000` tokens tagged `[id]`. At run end, the agent's final output schema stages `<= 3` candidate claims citing `source_step_ids` and `derived_from[]`. Raw transcripts, `task` scratch, and ungrounded claims are dropped.
* **Non-Amplifying IFC Quarantine Gate:** Child trust is bounded by `min_trust(tool_logs, parents)` so LLM summarization never launders untrusted ticket/email text into trusted beliefs. `T3_UNTRUSTED` claims start `QUARANTINED`, cannot drive outbound actions, and may only attach a `CONTESTED_BY` edge to `T0–T2` claims rather than silently overwriting them.
* **Optimistic Concurrency & Batched Revision:** Commits enforce optimistic CAS (`version`) per `(org_id, entity_key)`. Exact matches increment `pos` with `0` LLM calls; collisions or contradicting `T1` tool errors queue a batched sleep-time LLM judge (`5 pairs/call`) returning `CONFIRM`, `SUPERSEDE`, `COEXIST`, or `REJECT`.
* **Bidirectional JTMS Error Containment:** When delayed 2% human feedback flags a report, `1` async LLM call attributes the fault among recalled claims. A deterministic graph walk traces `DERIVED_FROM` backward to unverified `T2`/`T3` roots (`RETRACTED`) and forward across all descendants, quarantining any claim lacking independent `T0`/`T1` support (`0` LLM calls).
* **Cross-Tenant Safety & Bounded Forgetting:** Tenant claims never leave `org_id`. Only PII-scrubbed public-artifact patterns independently observed across `k >= 10` orgs promote to `global` procedural playbooks with zero org links. Nightly Ebbinghaus pruning caps active claims at `10,000/org` and compacts `SUPERSEDED`/`RETRACTED` leaf nodes older than 90 days.

### 3. Cost & Latency Budget (5,000 Runs/Day)

| Workload | LLM Calls / Day | Critical-Path Latency |
|---|---|---|
| Hot-path recall & run-end write staging | `0` | `~70 ms` median / `<= 150 ms` p95 |
| Batched collision & staleness checks (~1,250/day, 5:1) | `~250` | `0 ms` (async) |
| Delayed 2% human feedback attribution (100/day) | `~100` | `0 ms` (async) |
| Global `k`-anonymous distillation & nightly merges | `~50` | `0 ms` (async) |
| **Total** | **`~400/day` (`0.08/run` avg)** | **`<= 150 ms` p95 (cap: `300 ms`)** |

## Repository Structure

```text
.
├── Agentic_Memory_Architecture_Submission.pdf   # Final <= 1,200-word submission PDF (with 2 SVG diagrams)
├── INTERVIEW_STUDY_GUIDE.md                     # Architecture defense guide, whiteboard math & Q&A
├── requirements.txt                             # Minimal runtime dependencies (numpy, httpx)
├── build/
│   ├── generate_pdf.py                          # HTML + SVG -> PDF builder and strict word-count auditor
│   └── submission.html                          # Rendered HTML source
├── src/
│   ├── __init__.py                              # Public package exports
│   ├── models.py                                # Claim, CandidateClaim, TrustTier, ClaimStatus, EdgeType
│   ├── indexes.py                               # SQLite FTS5 BM25, NumPy VectorIndex, Hashing/OpenAI embedders
│   ├── storage.py                               # SQLite WAL ClaimRepository, graph edges, CAS, async_jobs queue
│   ├── llm_adapters.py                          # Pluggable batched LLMJudge (OpenAI-compatible + local heuristic)
│   ├── worker.py                                # Async SleepTimeWorker (5:1 conflict batching, JTMS, decay)
│   └── memory_layer.py                          # AgentMemoryLayer hot-path recall & non-blocking staging API
└── tests/
    └── stress_test.py                           # Compliance, SQLite WAL persistence, async & scale tests
```

## Verification & Stress Tests

Requires Python 3.10+, `numpy`, `httpx` (and Google Chrome for PDF regeneration).

```bash
pip install -r requirements.txt

# 1. Regenerate the submission PDF and run the word-count audit
python3 build/generate_pdf.py

# 2. Run the full compliance and production stress-test suite (10 tests)
python3 -m unittest tests/stress_test.py -v
```

### What `tests/stress_test.py` Verifies

1. **Submission Compliance:** Exact 6 headings in order, per-section and total (`1,157 <= 1,200`) word limits, XML-valid SVG diagrams (`<= 2`), and zero unexpected attribution tokens.
2. **Persisted Hybrid Recall:** SQLite FTS5 BM25 + NumPy dense vector scores persist across database close/reopen and render `[id]`-tagged context blocks.
3. **Scenario 1 (Staleness):** `T3` Postgres-16 error contests `T1` Postgres-14 via `CONTESTED_BY` without silent overwrite; `T1` verification closes `valid_to`, activates Postgres 16, and writes a `SUPERSEDES` edge.
4. **Scenario 2 (Planted Memory):** `20` repeated `T3` Sybil tickets and summarization laundering attempts stay `QUARANTINED` and blocked at the outbound email sink-gate, while `T0` confirmation promotes valid updates.
5. **Scenario 3 (Compounding Error):** Delayed feedback triggers async attribution and bidirectional `DERIVED_FROM` rollback, retracting root `C_utc`, quarantining all `14` single-support derived claims, and preserving `T1`-backed claims with penalized confidence.
6. **Scenario 4 (Cross-Tenant Isolation):** Zero org-to-org leakage; public OSS bug patterns stay private at `9` orgs and promote to `global` at `k >= 10` (`30` orgs) with org identifiers scrubbed.
7. **Async Worker, CAS & Scale:** Background `asyncio` worker loop, SQLite optimistic CAS mismatch rejection, Ebbinghaus `10,000/org` capping (protecting `T0`), and multi-tenant critical-path latency + batched LLM budget checks.

## Known Trade-Offs

* **Silent `T1` Semantic Misreads:** If an internal `T1` tool returns valid JSON with ambiguous values (e.g., `04/05/2026`) and zero tool errors, repeated runs corroborate the wrong interpretation until a 2% human flag triggers DAG rollback. Catching silent semantic misreads per run would require a ground-truth oracle or per-read LLM audit, breaking both the `2.0` LLM-call and `300 ms` latency caps.
* **Single-Node Storage Backend:** `ClaimRepository` uses SQLite WAL + FTS5 + in-process NumPy vector matrices + a durable `async_jobs` table so the entire production architecture runs locally with zero external infrastructure; swapping in Qdrant/pgvector, Redis Streams, or Neo4j requires implementing the same repository/index interfaces.
