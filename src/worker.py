"""Asynchronous sleep-time batch worker processing staged writes, conflicts, rollbacks, and decay."""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timedelta, timezone
import math
import re
from typing import Dict, Iterable, List, Optional, Set, Tuple

from src.llm_adapters import LLMJudge
from src.models import (
    Claim,
    ClaimStatus,
    EdgeType,
    JobType,
    RevisionAction,
    Scope,
    TIER_PRIORS,
    TrustTier,
    utc_now,
)
from src.storage import ClaimRepository, ConcurrencyConflictError


PII_RE = re.compile(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+|\b\d{1,3}(?:\.\d{1,3}){3}\b)")


class SleepTimeWorker:
    """Off-critical-path worker consuming durable async_jobs with batched LLM calls."""

    def __init__(
        self,
        repo: ClaimRepository,
        judge: LLMJudge,
        max_active_per_org: int = 10_000,
        k_anonymity_threshold: int = 10,
        similarity_conflict_threshold: float = 0.82,
    ) -> None:
        self.repo = repo
        self.judge = judge
        self.max_active_per_org = max_active_per_org
        self.k_anonymity_threshold = k_anonymity_threshold
        self.similarity_conflict_threshold = similarity_conflict_threshold
        self._stop_event = asyncio.Event()

    async def run_loop(self, poll_interval_s: float = 0.05) -> None:
        """Continuously drain pending async jobs until stopped."""
        while not self._stop_event.is_set():
            processed = self.drain_once()
            if processed == 0:
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=poll_interval_s)
                except asyncio.TimeoutError:
                    pass

    def stop(self) -> None:
        self._stop_event.set()

    def drain_once(self, now: Optional[datetime] = None) -> int:
        """Process one pass across all pending job types; returns number of jobs handled."""
        now = now or utc_now()
        handled = 0

        # 1. Staged writes (deterministic gate + collision detection)
        for job in self.repo.claim_pending_jobs(JobType.STAGE_WRITE, limit=100):
            try:
                self._handle_stage_write(job["org_id"], job["payload"], now=now)
                self.repo.finish_job(job["id"], "DONE")
                handled += 1
            except Exception:
                self.repo.retry_or_fail_job(job["id"])

        # 2. Batched conflict revisions (up to 5 pairs per LLM call)
        conflict_jobs = self.repo.claim_pending_jobs(JobType.RESOLVE_CONFLICT, limit=50)
        if conflict_jobs:
            try:
                self._handle_batched_conflicts(conflict_jobs, now=now)
                for job in conflict_jobs:
                    self.repo.finish_job(job["id"], "DONE")
                    handled += 1
            except Exception:
                for job in conflict_jobs:
                    self.repo.retry_or_fail_job(job["id"])

        # 3. Delayed 2% human feedback attribution + bidirectional JTMS rollback
        for job in self.repo.claim_pending_jobs(JobType.FEEDBACK_ROLLBACK, limit=25):
            try:
                self._handle_feedback_rollback(job["payload"])
                self.repo.finish_job(job["id"], "DONE")
                handled += 1
            except Exception:
                self.repo.retry_or_fail_job(job["id"])

        # 4. Cross-tenant k-anonymous public-artifact distillation
        for job in self.repo.claim_pending_jobs(JobType.GLOBAL_DISTILL, limit=50):
            try:
                self._handle_global_distill(job["org_id"], job["payload"], now=now)
                self.repo.finish_job(job["id"], "DONE")
                handled += 1
            except Exception:
                self.repo.retry_or_fail_job(job["id"])

        # 5. Nightly bounded forgetting & leaf compaction
        for job in self.repo.claim_pending_jobs(JobType.DECAY_COMPACT, limit=50):
            try:
                self.enforce_bounded_forgetting(job["org_id"], now=now)
                self.repo.finish_job(job["id"], "DONE")
                handled += 1
            except Exception:
                self.repo.retry_or_fail_job(job["id"])

        return handled

    def compute_non_amplifying_tier(
        self, source_tiers: Iterable[TrustTier], parent_ids: Iterable[str]
    ) -> TrustTier:
        tiers = list(source_tiers)
        parents = self.repo.get_claims_batch(list(parent_ids))
        for p in parents.values():
            tiers.append(p.trust_tier)
        return max(tiers) if tiers else TrustTier.T3_UNTRUSTED

    def _handle_stage_write(self, org_id: str, payload: dict, now: datetime) -> None:
        scope = Scope(payload["scope"])
        if scope == Scope.TASK:
            return
        step_ids = payload.get("source_step_ids") or []
        if not step_ids:
            return  # Drop ungrounded claims lacking tool/step provenance

        tool_logs: Dict[str, int] = payload.get("tool_logs", {})
        source_tiers = [TrustTier(tool_logs.get(sid, TrustTier.T3_UNTRUSTED)) for sid in step_ids]
        parent_ids: List[str] = list(payload.get("derived_from") or [])
        tier = self.compute_non_amplifying_tier(source_tiers, parent_ids)
        pos, neg = TIER_PRIORS[tier]
        status = ClaimStatus.QUARANTINED if tier == TrustTier.T3_UNTRUSTED else ClaimStatus.ACTIVE

        entity_key = payload["entity_key"]
        text = payload["text"]
        run_id = payload["run_id"]
        user_id = payload.get("user_id")
        sig = payload.get("public_artifact_signature")

        vec = self.repo.embedder.embed(f"{entity_key} {text}")
        candidate = Claim(
            org_id=org_id,
            user_id=user_id,
            scope=scope,
            entity_key=entity_key,
            text=text,
            trust_tier=tier,
            status=status,
            pos=pos,
            neg=neg,
            valid_from=now,
            run_id=run_id,
            last_used=now,
        )
        outcome, committed, collision_id = self.repo.commit_or_corroborate_atomic(
            claim=candidate, vec=vec, parent_ids=parent_ids, now=now
        )
        if outcome == "CORROBORATED":
            return

        # Collision detection: exact entity_key mismatch OR high vector similarity > 0.82
        collision_ids: List[str] = []
        if collision_id is not None:
            collision_ids.append(collision_id)
        else:
            self.repo.sync_vector_cache()
            sim_hits = self.repo.vector_index.search([org_id], vec, top_k=5)
            for hit_id, sim in sim_hits.items():
                if hit_id != committed.id and sim > self.similarity_conflict_threshold:
                    hit_claim = self.repo.get_claim(hit_id)
                    if hit_claim is not None and hit_claim.text != text:
                        collision_ids.append(hit_id)
                        break

        if collision_ids:
            self.repo.enqueue_job(
                JobType.RESOLVE_CONFLICT,
                org_id,
                {"existing_id": collision_ids[0], "candidate_id": committed.id},
            )

        if sig:
            scrubbed = PII_RE.sub("<REDACTED>", text).replace(org_id, "<ORG>")
            self.repo.enqueue_job(
                JobType.GLOBAL_DISTILL,
                org_id,
                {"signature": sig, "scrubbed_rule": scrubbed},
            )

    def _handle_batched_conflicts(self, jobs: List[dict], now: datetime) -> None:
        pairs: List[Tuple[Claim, Claim]] = []
        for job in jobs:
            old = self.repo.get_claim(job["payload"]["existing_id"])
            new = self.repo.get_claim(job["payload"]["candidate_id"])
            if old is not None and new is not None:
                pairs.append((old, new))
        if not pairs:
            return

        decisions = self.judge.batched_revise_conflicts(pairs)
        for dec in decisions:
            for _ in range(3):
                old = self.repo.get_claim(dec.existing_id)
                new = self.repo.get_claim(dec.candidate_id)
                if old is None or new is None:
                    break
                try:
                    if dec.action == RevisionAction.REJECT:
                        new.status = ClaimStatus.RETRACTED
                        self.repo.update_claim_cas(new, expected_version=new.version)
                        break

                    # Cross-tier IFC invariant: T3 may contest T0-T2, never silently supersede it
                    if (
                        new.trust_tier == TrustTier.T3_UNTRUSTED
                        and old.trust_tier < TrustTier.T3_UNTRUSTED
                    ):
                        old.status = ClaimStatus.CONTESTED
                        old.neg += 1
                        self.repo.update_claim_cas(old, expected_version=old.version)
                        new.status = ClaimStatus.QUARANTINED
                        self.repo.update_claim_cas(new, expected_version=new.version)
                        self.repo.add_edge(old.id, new.id, EdgeType.CONTESTED_BY)
                        break

                    if dec.action == RevisionAction.SUPERSEDE:
                        self.repo.supersede_and_activate_atomic(
                            new_claim_id=new.id, fallback_old_id=old.id, now=now
                        )
                        break
                    if dec.action == RevisionAction.COEXIST:
                        if dec.split_entity_key:
                            new.entity_key = dec.split_entity_key
                        new.status = ClaimStatus.ACTIVE
                        self.repo.update_claim_cas(new, expected_version=new.version)
                        break
                    if dec.action == RevisionAction.CONFIRM:
                        old.pos += 1
                        self.repo.update_claim_cas(old, expected_version=old.version)
                        new.status = ClaimStatus.SUPERSEDED
                        self.repo.update_claim_cas(new, expected_version=new.version)
                        break
                    break
                except ConcurrencyConflictError:
                    continue

    def _handle_feedback_rollback(self, payload: dict) -> Set[str]:
        recalled_ids: List[str] = payload.get("recalled_claim_ids", [])
        recalled_map = self.repo.get_claims_batch(recalled_ids)
        recalled_list = [recalled_map[cid] for cid in recalled_ids if cid in recalled_map]
        if not recalled_list:
            return set()

        faulty_ids = self.judge.attribute_feedback(
            payload.get("report_text", ""),
            payload.get("human_note", ""),
            recalled_list,
        )
        all_affected: Set[str] = set()
        for fid in faulty_ids:
            all_affected |= self.retract_with_jtms_rollback(fid)
        return all_affected

    def retract_with_jtms_rollback(self, flagged_claim_id: str) -> Set[str]:
        """Deterministic bidirectional JTMS rollback (0 LLM calls)."""
        flagged = self.repo.get_claim(flagged_claim_id)
        if flagged is None:
            return set()

        # Walk backward along DERIVED_FROM to unverified T2/T3 root premises
        roots: Set[str] = set()
        back_q = deque([flagged_claim_id])
        seen_back: Set[str] = set()
        while back_q:
            curr_id = back_q.popleft()
            if curr_id in seen_back:
                continue
            seen_back.add(curr_id)
            curr = self.repo.get_claim(curr_id)
            if curr is None:
                continue
            parent_ids = self.repo.get_neighbors(curr_id, EdgeType.DERIVED_FROM, direction="in")
            parents = self.repo.get_claims_batch(parent_ids)
            unverified_parents = [
                pid for pid, p in parents.items() if p.trust_tier >= TrustTier.T2_INFERRED
            ]
            if not unverified_parents:
                if curr.trust_tier >= TrustTier.T2_INFERRED or curr_id == flagged_claim_id:
                    roots.add(curr_id)
            else:
                back_q.extend(unverified_parents)

        if not roots:
            roots.add(flagged_claim_id)

        affected: Set[str] = set()
        for rid in roots:
            rc = self.repo.get_claim(rid)
            if rc is not None and rc.status != ClaimStatus.RETRACTED:
                rc.status = ClaimStatus.RETRACTED
                self.repo.update_claim_cas(rc, expected_version=rc.version)
            affected.add(rid)

        # Forward BFS along DERIVED_FROM to quarantine unsupported subgraph
        fwd_q = deque(roots)
        while fwd_q:
            curr_id = fwd_q.popleft()
            for child_id in self.repo.get_neighbors(curr_id, EdgeType.DERIVED_FROM, direction="out"):
                child = self.repo.get_claim(child_id)
                if child is None or child.status in (ClaimStatus.RETRACTED, ClaimStatus.QUARANTINED):
                    continue
                parent_ids = self.repo.get_neighbors(child_id, EdgeType.DERIVED_FROM, direction="in")
                parents = self.repo.get_claims_batch(parent_ids)
                has_alt_verified = any(
                    pid not in (affected | roots)
                    and p.trust_tier <= TrustTier.T1_VERIFIED
                    and p.status == ClaimStatus.ACTIVE
                    for pid, p in parents.items()
                )
                if has_alt_verified:
                    child.neg += 1
                    self.repo.update_claim_cas(child, expected_version=child.version)
                else:
                    child.status = ClaimStatus.QUARANTINED
                    self.repo.update_claim_cas(child, expected_version=child.version)
                    affected.add(child_id)
                    fwd_q.append(child_id)
        return affected

    def _handle_global_distill(self, org_id: str, payload: dict, now: datetime) -> None:
        signature = payload["signature"]
        scrubbed = payload["scrubbed_rule"]
        distinct_orgs, samples = self.repo.record_global_observation(signature, org_id, scrubbed)
        if distinct_orgs < self.k_anonymity_threshold:
            return
        entity_key = f"oss:{signature}"
        if self.repo.find_active_by_entity("GLOBAL", entity_key) is not None:
            return
        distilled_text = self.judge.distill_global_rule(signature, samples)
        pos, neg = TIER_PRIORS[TrustTier.T1_VERIFIED]
        global_claim = Claim(
            org_id="GLOBAL",
            scope=Scope.GLOBAL,
            entity_key=entity_key,
            text=distilled_text,
            trust_tier=TrustTier.T1_VERIFIED,
            status=ClaimStatus.ACTIVE,
            pos=pos,
            neg=neg,
            valid_from=now,
            run_id="k_anon_batch",
            last_used=now,
        )
        self.repo.insert_claim(global_claim)

    def enforce_bounded_forgetting(self, org_id: str, now: Optional[datetime] = None) -> int:
        now = now or utc_now()
        removed = 0
        org_claims = self.repo.list_org_claims(org_id)

        # Compact >90d SUPERSEDED/RETRACTED leaf nodes with zero active dependents
        for c in org_claims:
            if c.status in (ClaimStatus.SUPERSEDED, ClaimStatus.RETRACTED):
                age_days = (now - c.last_used).total_seconds() / 86400.0
                if age_days <= 90.0:
                    continue
                child_ids = self.repo.get_neighbors(c.id, EdgeType.DERIVED_FROM, direction="out")
                children = self.repo.get_claims_batch(child_ids)
                if not any(
                    ch.status in (ClaimStatus.ACTIVE, ClaimStatus.CONTESTED)
                    for ch in children.values()
                ):
                    self.repo.delete_claim(c.id, c.org_id)
                    removed += 1

        # Cap active claims per org via Ebbinghaus salience (protecting T0_HUMAN)
        refreshed = self.repo.list_org_claims(org_id)
        total_active = sum(
            1 for c in refreshed if c.status in (ClaimStatus.ACTIVE, ClaimStatus.CONTESTED)
        )
        if total_active > self.max_active_per_org:
            excess = total_active - self.max_active_per_org
            evictable = [
                c
                for c in refreshed
                if c.status in (ClaimStatus.ACTIVE, ClaimStatus.CONTESTED, ClaimStatus.QUARANTINED)
                and c.trust_tier > TrustTier.T0_HUMAN
            ]
            evictable.sort(
                key=lambda c: c.confidence
                * math.exp(-0.02 * max((now - c.last_used).total_seconds() / 86400.0, 0.0))
                * math.log1p(c.uses)
            )
            for c in evictable[:excess]:
                self.repo.delete_claim(c.id, c.org_id)
                removed += 1
        return removed
