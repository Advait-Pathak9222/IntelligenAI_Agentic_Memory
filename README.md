# Epistemic Provenance Graph — Agentic Memory for Headless Agents

Architecture submission, reference implementation, and automated stress-test suite for designing persistent, self-correcting memory for unsupervised headless AI agents.

## Core Design

Every stored memory is modeled as a **taint-labeled, bi-temporal Claim node** inside an org-partitioned **Provenance DAG** (`DERIVED_FROM`, `CONTESTED_BY`, and `SUPERSEDES` edges):

- **Hot Path (0 LLM calls, <=150 ms p95):** Zero-LLM hybrid recall (BM25 + vector + 1-hop graph) packs `ACTIVE` and `CONTESTED` claims within a 4,000-token context cap, while run-end output schemas stage `<=3` candidate claims referencing `source_step_ids` and `derived_from[]`.
- **IFC Trust & Quarantine:** Four integrity tiers (`T0_HUMAN > T1_VERIFIED > T2_INFERRED > T3_UNTRUSTED`) enforce strict non-amplification (`trust_tier(child) = min_trust(parents)`). External `T3` claims start `QUARANTINED` and cannot silently overwrite higher-tier beliefs or authorize outbound external actions.
- **Sleep-Time Batch Lane (~0.08–0.30 LLM calls/run avg):** Exact corroborations increment evidence counts with zero LLM calls; entity collisions and semantic conflicts trigger batched sleep-time revision (`5 pairs/call`).
- **Bidirectional JTMS Rollback:** When delayed 2% human feedback flags a report, the graph walks `DERIVED_FROM` backward to unverified `T2`/`T3` roots and forward across all dependents to quarantine every unsupported downstream memory deterministically.
- **Cross-Tenant Safety & Bounded Lifetime:** Org data stays partitioned by `org_id`. Only PII-scrubbed public-artifact patterns seen independently by `k >= 10` orgs promote to `global` playbooks. Nightly Ebbinghaus pruning caps active claims at `10,000/org`.

## Repository Structure

```text
.
├── Agentic_Memory_Architecture_Submission.pdf   # Final <=1,200-word submission PDF (with 2 diagrams)
├── INTERVIEW_STUDY_GUIDE.md                     # Architecture defense guide & whiteboard math
├── build/
│   ├── generate_pdf.py                          # Reproducible HTML+SVG -> PDF builder & word-count auditor
│   └── submission.html                          # Rendered HTML source
├── src/
│   └── epistemic_memory.py                      # Minimal zero-dependency reference engine
└── tests/
    └── stress_test.py                           # Compliance, adversarial scenario & 5k-run scale tests
```

## Quickstart

Regenerate the submission PDF (requires Python 3.10+ and Google Chrome):

```bash
python3 build/generate_pdf.py
```

Run the full compliance and adversarial stress-test suite:

```bash
python3 -m unittest tests/stress_test.py -v
```
