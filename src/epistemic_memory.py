"""Minimal reference implementation of the Epistemic Provenance Graph memory layer."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, IntEnum
import math
import re
from typing import Dict, Iterable, List, Optional, Set, Tuple
import uuid


class TrustTier(IntEnum):
    """Information-Flow Control (IFC) integrity hierarchy (lower int = higher trust)."""

    T0_HUMAN = 0
    T1_VERIFIED = 1
    T2_INFERRED = 2
    T3_UNTRUSTED = 3


class ClaimStatus(str, Enum):
    QUARANTINED = "QUARANTINED"
    ACTIVE = "ACTIVE"
    CONTESTED = "CONTESTED"
    SUPERSEDED = "SUPERSEDED"
    RETRACTED = "RETRACTED"


class Scope(str, Enum):
    TASK = "task"
    USER = "user"
    ORG = "org"
    GLOBAL = "global"


TIER_PRIORS: Dict[TrustTier, Tuple[int, int]] = {
    TrustTier.T0_HUMAN: (8, 0),
    TrustTier.T1_VERIFIED: (3, 0),
    TrustTier.T2_INFERRED: (0, 0),
    TrustTier.T3_UNTRUSTED: (0, 2),
}


@dataclass
class Claim:
    org_id: str
    scope: Scope
    entity_key: str
    text: str
    trust_tier: TrustTier
    status: ClaimStatus
    pos: int
    neg: int
    valid_from: datetime
    run_id: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    valid_to: Optional[datetime] = None
    derived_from: List[str] = field(default_factory=list)
    contested_by: List[str] = field(default_factory=list)
    supersedes: Optional[str] = None
    version: int = 1
    last_used: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    uses: int = 1

    @property
    def confidence(self) -> float:
        return (self.pos + 1) / (self.pos + self.neg + 2)


class ConcurrencyConflictError(Exception):
    """Raised on optimistic CAS version mismatch for concurrent writes."""


class EpistemicMemoryStore:
    """Org-partitioned provenance DAG implementing IFC quarantine, CAS, and JTMS rollback."""

    def __init__(
        self,
        max_active_per_org: int = 10_000,
        k_anonymity_threshold: int = 10,
    ) -> None:
        self.claims: Dict[str, Claim] = {}
        self.children_index: Dict[str, Set[str]] = {}
        self.org_allowlists: Dict[str, Set[str]] = {}
        self.global_signatures: Dict[str, Set[str]] = {}
        self.global_signature_text: Dict[str, str] = {}
        self.max_active_per_org = max_active_per_org
        self.k_anonymity_threshold = k_anonymity_threshold
        self.conflict_queue: List[Tuple[str, str]] = []

    def register_verified_domain(self, org_id: str, domain: str) -> None:
        self.org_allowlists.setdefault(org_id, set()).add(domain.lower())

    def compute_non_amplifying_tier(
        self,
        source_tiers: Iterable[TrustTier],
        parent_ids: Iterable[str],
    ) -> TrustTier:
        """IFC non-amplification: child trust cannot exceed worst input or parent trust."""
        tiers = list(source_tiers)
        for pid in parent_ids:
            if pid in self.claims:
                tiers.append(self.claims[pid].trust_tier)
        return max(tiers) if tiers else TrustTier.T3_UNTRUSTED

    def stage_and_commit(
        self,
        org_id: str,
        scope: Scope,
        entity_key: str,
        text: str,
        source_tiers: Iterable[TrustTier],
        run_id: str,
        derived_from: Optional[List[str]] = None,
        now: Optional[datetime] = None,
        expected_version: Optional[int] = None,
    ) -> Claim:
        """Deterministic write gate + CAS commit; queues collisions instead of silent overwrite."""
        if scope == Scope.TASK:
            raise ValueError("Task-scoped scratch is deliberately excluded from persistent memory.")
        now = now or datetime.now(timezone.utc)
        parent_ids = list(derived_from or [])
        tier = self.compute_non_amplifying_tier(source_tiers, parent_ids)
        pos, neg = TIER_PRIORS[tier]
        status = ClaimStatus.QUARANTINED if tier == TrustTier.T3_UNTRUSTED else ClaimStatus.ACTIVE

        existing_active = self._find_active_by_entity(org_id, entity_key)
        if expected_version is not None and existing_active is not None:
            if existing_active.version != expected_version:
                raise ConcurrencyConflictError(
                    f"CAS mismatch on {org_id}/{entity_key}: expected v{expected_version}, got v{existing_active.version}"
                )

        # Exact match corroboration (0 LLM calls)
        if existing_active is not None and existing_active.text == text:
            if existing_active.trust_tier <= tier:
                existing_active.pos += 1
                existing_active.last_used = now
                existing_active.uses += 1
                existing_active.version += 1
                return existing_active

        claim = Claim(
            org_id=org_id,
            scope=scope,
            entity_key=entity_key,
            text=text,
            trust_tier=tier,
            status=status,
            pos=pos,
            neg=neg,
            valid_from=now,
            run_id=run_id,
            derived_from=parent_ids,
            last_used=now,
        )
        self.claims[claim.id] = claim
        for pid in parent_ids:
            self.children_index.setdefault(pid, set()).add(claim.id)

        if existing_active is not None and existing_active.text != text:
            self.conflict_queue.append((existing_active.id, claim.id))

        return claim

    def resolve_conflict(
        self,
        existing_id: str,
        candidate_id: str,
        decision: str,
        now: Optional[datetime] = None,
    ) -> None:
        """Apply sleep-time revision verdict while enforcing cross-tier non-overwrite rules."""
        now = now or datetime.now(timezone.utc)
        old = self.claims[existing_id]
        new = self.claims[candidate_id]

        if decision == "REJECT":
            new.status = ClaimStatus.RETRACTED
            return

        if new.trust_tier == TrustTier.T3_UNTRUSTED and old.trust_tier < TrustTier.T3_UNTRUSTED:
            # T3 may never silently supersede T0-T2; mark contested and link hint.
            old.status = ClaimStatus.CONTESTED
            old.neg += 1
            old.version += 1
            if new.id not in old.contested_by:
                old.contested_by.append(new.id)
            new.status = ClaimStatus.QUARANTINED
            return

        if decision == "SUPERSEDE":
            old.status = ClaimStatus.SUPERSEDED
            old.valid_to = now
            old.version += 1
            new.status = ClaimStatus.ACTIVE
            new.supersedes = old.id
            new.version += 1
        elif decision == "COEXIST":
            old.status = ClaimStatus.ACTIVE
            new.status = ClaimStatus.ACTIVE
        elif decision == "CONFIRM":
            old.pos += 1
            old.version += 1
            new.status = ClaimStatus.SUPERSEDED

    def can_trigger_external_email(self, org_id: str, claim_id: str, email: str) -> bool:
        """Deterministic action sink-gate for outbound communications."""
        claim = self.claims[claim_id]
        if claim.org_id != org_id:
            return False
        domain = email.split("@")[-1].lower()
        allowlist = self.org_allowlists.get(org_id, set())
        if claim.trust_tier == TrustTier.T3_UNTRUSTED or claim.status == ClaimStatus.QUARANTINED:
            return False
        return domain in allowlist or claim.trust_tier == TrustTier.T0_HUMAN

    def recall(
        self,
        org_id: str,
        query_terms: Iterable[str],
        now: Optional[datetime] = None,
        token_budget: int = 4000,
    ) -> List[Tuple[Claim, List[str]]]:
        """Zero-LLM org-isolated recall returning claims + unverified CONTESTED_BY hints."""
        now = now or datetime.now(timezone.utc)
        terms = {t.lower() for t in query_terms if t}
        scored: List[Tuple[float, Claim]] = []

        for c in self.claims.values():
            if c.scope == Scope.GLOBAL:
                pass
            elif c.org_id != org_id:
                continue
            if c.status not in (ClaimStatus.ACTIVE, ClaimStatus.CONTESTED):
                continue
            tokens = set(re.findall(r"[a-z0-9_.:-]+", f"{c.entity_key} {c.text}".lower()))
            overlap = len(terms & tokens) / max(len(terms), 1)
            sim = 0.2 + 0.8 * overlap
            days = max((now - c.last_used).total_seconds() / 86400.0, 0.0)
            score = sim * c.confidence * math.exp(-0.02 * days)
            scored.append((score, c))

        scored.sort(key=lambda item: item[0], reverse=True)
        packed: List[Tuple[Claim, List[str]]] = []
        used_tokens = 0
        for _, claim in scored:
            hints = [
                self.claims[cid].text
                for cid in claim.contested_by
                if cid in self.claims and self.claims[cid].status == ClaimStatus.QUARANTINED
            ]
            est_tokens = max(10, len(claim.text.split()) + sum(len(h.split()) for h in hints) + 8)
            if used_tokens + est_tokens > token_budget:
                continue
            used_tokens += est_tokens
            claim.last_used = now
            claim.uses += 1
            packed.append((claim, hints))
        return packed

    def retract_with_jtms_rollback(self, flagged_claim_id: str) -> Set[str]:
        """Bidirectional JTMS rollback: walk back to T2/T3 roots, then forward-quarantine subgraph."""
        if flagged_claim_id not in self.claims:
            return set()

        # Walk backward to unverified (T2/T3) root premises
        roots: Set[str] = set()
        back_q = deque([flagged_claim_id])
        seen_back: Set[str] = set()
        while back_q:
            curr_id = back_q.popleft()
            if curr_id in seen_back or curr_id not in self.claims:
                continue
            seen_back.add(curr_id)
            curr = self.claims[curr_id]
            unverified_parents = [
                pid
                for pid in curr.derived_from
                if pid in self.claims and self.claims[pid].trust_tier >= TrustTier.T2_INFERRED
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
            self.claims[rid].status = ClaimStatus.RETRACTED
            self.claims[rid].version += 1
            affected.add(rid)

        # Forward BFS to quarantine descendants without independent T0/T1 support
        fwd_q = deque(roots)
        while fwd_q:
            curr_id = fwd_q.popleft()
            for child_id in self.children_index.get(curr_id, set()):
                child = self.claims[child_id]
                if child.status in (ClaimStatus.RETRACTED, ClaimStatus.QUARANTINED):
                    continue
                if self._has_independent_verified_support(child, retracted_or_bad=affected | roots):
                    child.neg += 1
                    child.version += 1
                else:
                    child.status = ClaimStatus.QUARANTINED
                    child.version += 1
                    affected.add(child_id)
                    fwd_q.append(child_id)
        return affected

    def _has_independent_verified_support(self, claim: Claim, retracted_or_bad: Set[str]) -> bool:
        valid_verified_parents = [
            pid
            for pid in claim.derived_from
            if pid not in retracted_or_bad
            and pid in self.claims
            and self.claims[pid].trust_tier <= TrustTier.T1_VERIFIED
            and self.claims[pid].status == ClaimStatus.ACTIVE
        ]
        return len(valid_verified_parents) > 0

    def record_public_oss_pattern(self, org_id: str, signature: str, scrubbed_rule: str) -> Optional[Claim]:
        """Distill PII-scrubbed public OSS fixes to global scope once k distinct orgs observe them."""
        orgs = self.global_signatures.setdefault(signature, set())
        orgs.add(org_id)
        self.global_signature_text[signature] = scrubbed_rule
        if len(orgs) >= self.k_anonymity_threshold:
            existing = self._find_active_by_entity("GLOBAL", f"oss:{signature}")
            if existing is not None:
                return existing
            return self.stage_and_commit(
                org_id="GLOBAL",
                scope=Scope.GLOBAL,
                entity_key=f"oss:{signature}",
                text=scrubbed_rule,
                source_tiers=[TrustTier.T1_VERIFIED],
                run_id="k_anon_batch",
            )
        return None

    def enforce_bounded_forgetting(self, org_id: str, now: Optional[datetime] = None) -> int:
        """Compact >90d dead leaves and cap active claims per org via Ebbinghaus salience."""
        now = now or datetime.now(timezone.utc)
        removed = 0

        # Compact SUPERSEDED / RETRACTED leaf nodes older than 90 days
        leaf_ids = []
        for cid, c in list(self.claims.items()):
            if c.org_id != org_id:
                continue
            if c.status in (ClaimStatus.SUPERSEDED, ClaimStatus.RETRACTED):
                age_days = (now - c.last_used).total_seconds() / 86400.0
                has_active_children = any(
                    self.claims[ch].status in (ClaimStatus.ACTIVE, ClaimStatus.CONTESTED)
                    for ch in self.children_index.get(cid, set())
                    if ch in self.claims
                )
                if age_days > 90.0 and not has_active_children:
                    leaf_ids.append(cid)
        for cid in leaf_ids:
            del self.claims[cid]
            removed += 1

        # Cap active claims per org (protecting T0_HUMAN)
        active_evictable = [
            c
            for c in self.claims.values()
            if c.org_id == org_id
            and c.status in (ClaimStatus.ACTIVE, ClaimStatus.CONTESTED, ClaimStatus.QUARANTINED)
            and c.trust_tier > TrustTier.T0_HUMAN
        ]
        total_active = sum(
            1
            for c in self.claims.values()
            if c.org_id == org_id and c.status in (ClaimStatus.ACTIVE, ClaimStatus.CONTESTED)
        )
        if total_active > self.max_active_per_org:
            excess = total_active - self.max_active_per_org
            active_evictable.sort(
                key=lambda c: c.confidence
                * math.exp(-0.02 * max((now - c.last_used).total_seconds() / 86400.0, 0.0))
                * math.log1p(c.uses)
            )
            for c in active_evictable[:excess]:
                del self.claims[c.id]
                removed += 1
        return removed

    def _find_active_by_entity(self, org_id: str, entity_key: str) -> Optional[Claim]:
        for c in self.claims.values():
            if c.org_id == org_id and c.entity_key == entity_key and c.status == ClaimStatus.ACTIVE:
                return c
        return None
