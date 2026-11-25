# Interview Defense & Study Guide — Epistemic Provenance Graph

This guide accompanies [Agentic_Memory_Architecture_Submission.pdf](file:///Users/advaitpathak/Desktop/Placement%20Assignments/Intelligen-AI/Agentic_Memory_Architecture_Submission.pdf) and prepares you to defend every decision in a live engineering interview.

---

## 1. The 3-Minute Whiteboard Pitch

If the interviewer says *"Walk me through your design from scratch"*, use this structure:

1. **Reframe the problem (15 sec):**
   > "For a headless agent, memory isn't a retrieval problem—it's an **epistemic correctness** problem with no human watching. Claims go stale, contradict, get poisoned by untrusted emails/tickets, and compound across runs—all under a strict **300 ms p95 latency** cap and **2 LLM calls/run** cost cap."
2. **One data model (45 sec):**
   > "Instead of storing raw text chunks, every memory is one structured **Claim** node inside a **Provenance DAG** (`DERIVED_FROM` and `SUPERSEDES` edges). Each claim carries four things: a strict tenant/scope key (`org_id`, `scope`), a **non-amplifying trust tier** (`T0` human > `T1` verified API/tool > `T2` inferred > `T3` untrusted external text), a **bi-temporal window** (`valid_from`, `valid_to`), and lightweight **evidence counts** `(pos, neg)`."
3. **Two execution lanes (60 sec):**
   > "Because any LLM call takes >300 ms, the **Hot Path** makes **zero LLM calls**: before a run, hybrid BM25 + vector + graph search packs $\le 4,000$ tokens of `ACTIVE` claims in ~140 ms p95; at the end of the run, the agent emits $\le 3$ candidate claims inside its final output step (zero extra calls). Everything expensive happens offline in a **Sleep-Time Batch Lane**: deterministic rules handle exact matches and `T3` quarantine for free; only entity collisions or semantic contradictions trigger a batched LLM judge (5 pairs/call)."
4. **Why it handles all failure modes (45 sec):**
   > "Unverified external tickets (`T3`) start `QUARANTINED` and can contest—but never silently overwrite—`T0`/`T1` beliefs or authorize external domains. If delayed 2% human feedback flags a report days later, 1 LLM call pinpoints the bad root claim and a **0-LLM deterministic BFS** along `DERIVED_FROM` immediately quarantines every unsupported downstream memory."

---

## 2. How All 11 Requirements Map to One Minimal Model

| # | Requirement | Minimal Mechanism in Our Design | Why Nothing Extra Is Needed |
|---|---|---|---|
| 1 | **Write policy** | Final step emits $\le 3$ candidate claims with `source_step_ids`. Drops raw logs, `task` scratch, failed retries, ungrounded claims. | Zero extra write-extraction LLM calls on the hot path. |
| 2 | **Representation** | Single `Claim` struct + `DERIVED_FROM` / `SUPERSEDES` edges. | Avoids maintaining separate vector, KV, and profile schemas. |
| 3 | **Trust & provenance** | 4-tier IFC (`T0 > T1 > T2 > T3`) + non-amplification (`tier(child) = min_trust(parents)`). | Stops "provenance laundering" where an LLM summary makes an injected email look trusted. |
| 4 | **Revision** | Exact match $\to$ `pos++` (0 LLM). Collision/`sim > 0.82` $\to$ batched sleep-time LLM (`CONFIRM`, `SUPERSEDE`, `COEXIST`, `REJECT`). | Only ~25% of writes need an LLM; bi-temporal `valid_to` preserves history. |
| 5 | **Error containment** | Forward BFS on `DERIVED_FROM` when a root claim is `RETRACTED`; quarantine nodes with no alternate `T0`/`T1` path. | Instant, deterministic blast-radius rollback (0 LLM calls). |
| 6 | **Scope & isolation** | Strict `org_id` filter on `task`/`user`/`org`. `global` only accepts PII-scrubbed public-artifact patterns seen by $k \ge 10$ orgs. | Zero org leakage while still learning universal OSS fixes. |
| 7 | **Recall** | Zero-LLM hybrid BM25 + vector + graph lookup ranked by `sim * conf * exp(-0.02 * days)`, packed $\le 4\text{k}$ tokens (`T3` excluded from action rules). | Fast (<150 ms p95) and keeps untrusted claims out of action prompts. |
| 8 | **Forgetting** | Nightly Ebbinghaus pruning capped at 10k active claims/org; compact `SUPERSEDED`/`RETRACTED` leaves >90 days old. | Hard $O(1)$ ceiling (~2M active nodes across 200 orgs over 2+ years). |
| 9 | **Concurrency** | Short `(org_id, entity_key)` lock + optimistic CAS (`version`). Differing concurrent values queue a conflict pair. | Prevents silent last-write-wins corruption across parallel workers. |
| 10 | **Measurement** | 10% randomized memory-off holdout measuring repeat-task steps, tool retries, and 2% rejection rate. | Requires zero extra human labels beyond the existing 2% feedback. |
| 11 | **Cost & latency** | Hot path: 0 LLM calls, $\le 150\text{ ms}$ p95. Async: ~400 LLM calls/day total (**0.08/run avg**). | Massive headroom under the 300 ms p95 and 2.0 calls/run budgets. |

---

## 3. Whiteboard Math (Memorize These 4 Numbers)

1. **Daily Run Volume:** $5,000\text{ runs/day} \times 2\text{ max extra LLM calls/run} = 10,000\text{ memory LLM calls/day ceiling}$.
   * **Our usage:**
     * Hot-path recall & write staging: **$0$ calls/day**.
     * Collision/staleness checks (~25% of runs = $1,250$/day, batched 5:1): **$250$ calls/day**.
     * Delayed 2% feedback attribution ($2\% \times 5,000 = 100$ runs/day $\times 1$ call): **$100$ calls/day**.
     * Global $k$-anonymous distillation + nightly merges: **~$50$ calls/day**.
     * **Total = ~400 calls/day = $0.08$ calls/run** (even completely unbatched: $1,400/5,000 = 0.28 < 2.0$).
2. **Confidence Formula (Laplace / Beta Mean):**
   $$\text{conf} = \frac{\text{pos} + 1}{\text{pos} + \text{neg} + 2}$$
   * New unverified claim `(0,0)` starts at $0.50$ (or tier-weighted prior); 3 confirmations `(3,0)` gives $4/5 = 0.80$; 1 contradiction `(3,1)` drops to $4/6 = 0.67$. No LLM needed to score confidence.
3. **Storage Bound (2+ Years):**
   * Max $10,000\text{ active claims/org} \times 200\text{ orgs} = 2,000,000\text{ active rows}$ (~1–2 GB total with embeddings). Completely flat after steady state.
4. **Critical-Path Latency ($\le 300\text{ ms}$ p95):**
   * Hybrid vector + BM25 + 1-hop graph fetch (~60–90 ms p95) + deterministic scoring & 4k-token greedy pack (~15 ms) + async queue push (<10 ms) = **~140 ms p95** (and **0 ms** waiting on LLMs).

---

## 4. Tough Interview Questions & Winning Answers

### Q1: *"Why didn't you just use a vector database with timestamps and an LLM summarizer?"*
* **Answer:** Three reasons: (1) Vector similarity retrieves *both* "Postgres 14" and "Postgres 16" without knowing which is valid; (2) lossy LLM summarization strips provenance ("laundering" an untrusted email into a trusted system rule); and (3) running an LLM summarizer on every write/read violates the $\le 2$ LLM calls/run and $\le 300\text{ ms}$ budgets.

### Q2: *"In Scenario 2 (Planted email `billing-help@acme-support.co`), what if an attacker sends 20 tickets with that same email? Wouldn't corroboration promote it?"*
* **Answer:** **Great catch—and NO.** Cross-tier promotion requires **higher-tier corroboration (`T0` human or `T1` internal API/DKIM-verified domain)**, never $N$ copies of `T3` external tickets. Counting repeated `T3` inputs only updates frequency, never `trust_tier`. That's why we say `trust_tier(child) = min_trust(parents)` is a strict security invariant.

### Q3: *"In Scenario 1 (Postgres 14 $\to$ 16), what if Acme actually uses Postgres 14 in Billing and Postgres 16 in Analytics?"*
* **Answer:** That's exactly why the batched revision LLM has a **`COEXIST`** outcome alongside `SUPERSEDE`! If the evidence suggests two different scopes/environments rather than an upgrade, the judge splits `entity_key` into `acme:prod_db:billing` and `acme:prod_db:analytics` so both stay `ACTIVE` without fighting each other.

### Q4: *"What if the agent forgets or hallucinates while emitting the $\le 3$ structured candidate claims in its final step?"*
* **Answer:** (1) Enforce strict JSON schema / tool-call output on the final turn so it can't skip formatting; (2) the deterministic gate verifies that every `source_step_id` actually exists in that run's trace and checks `trust_tier` from the actual tool logs—so the agent **cannot lie about its own provenance** (e.g., claiming `T1` when the step was an external email read).

### Q5: *"In Scenario 3, when 1 report is flagged wrong, what if that run recalled 12 memories—how do you attribute the fault?"*
* **Answer:** That's the one place we explicitly spend **1 async LLM call** (`~100 calls/day` for the 2% feedback). We give the attribution LLM the flagged report, the human's correction note, and the $\le 12$ recalled claims from that `run_id`, and ask it to identify which claim(s) caused the wrong conclusion. Once identified, the forward blast-radius BFS across all 15 derived memories is **purely deterministic (0 LLM calls)**.

### Q6: *"In Scenario 4, could someone infer which orgs use a niche OSS library from the `global` playbook?"*
* **Answer:** First, $k \ge 10$ distinct orgs prevents single-tenant attribution ($k$-anonymity). Second, the global node stores **only** the scrubbed procedural rule and `org_count`, **zero `org_id` pointers**, and only triggers on public-artifact signatures (e.g., public PyPI/npm packages or public CVEs)—never proprietary internal identifiers. If even $k$-anonymized public library adoption is restricted by a strict customer contract, an org can opt out via a single `allow_global_contribute = false` flag while still reading `global` rules.

### Q7: *"What if two parallel workers update `acme:prod_db` at the exact same millisecond?"*
* **Answer:** Both acquire the short `(org_id, entity_key)` advisory lock and check `version` via Optimistic Concurrency Control (CAS). Worker A commits `v1 -> v2`. Worker B's `v1` CAS fails, reloads `v2`, and sees a collision on `entity_key`: if both wrote the same value, B just increments `pos`; if they wrote different values, B queues the pair for the sleep-time conflict judge instead of silently overwriting Worker A.

### Q8: *"Why did you say 'trust and provenance + error containment' is the hardest sub-problem instead of 'forgetting' or 'recall'?"*
* **Answer:** Recall and forgetting have well-understood engineering solutions (hybrid search + Ebbinghaus decay + hard caps). By contrast, **unsupervised epistemic correctness** has no local oracle: text alone cannot separate a malicious planted update from a real external update, and without causal lineage, a single silent error compounds indefinitely across future runs.

---

## 5. Build & Verification Notes
* **Submission PDF:** [Agentic_Memory_Architecture_Submission.pdf](file:///Users/advaitpathak/Desktop/Placement%20Assignments/Intelligen-AI/Agentic_Memory_Architecture_Submission.pdf)
* **Generator & Word-Count Script:** [generate_pdf.py](file:///Users/advaitpathak/Desktop/Placement%20Assignments/Intelligen-AI/build/generate_pdf.py)
* Verified word count: **1,101 words** (strictly under the **1,200-word** total limit and under every per-section limit), **2 diagrams**, **3 clean pages** with zero clipped elements or orphan headings.
