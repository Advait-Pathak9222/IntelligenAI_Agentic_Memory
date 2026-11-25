"""Core domain models and enums for the Epistemic Provenance Graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, IntEnum
from typing import Dict, List, Optional, Tuple
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


class EdgeType(str, Enum):
    DERIVED_FROM = "DERIVED_FROM"
    CONTESTED_BY = "CONTESTED_BY"
    SUPERSEDES = "SUPERSEDES"


class RevisionAction(str, Enum):
    CONFIRM = "CONFIRM"
    SUPERSEDE = "SUPERSEDE"
    COEXIST = "COEXIST"
    REJECT = "REJECT"


class JobType(str, Enum):
    STAGE_WRITE = "STAGE_WRITE"
    RESOLVE_CONFLICT = "RESOLVE_CONFLICT"
    FEEDBACK_ROLLBACK = "FEEDBACK_ROLLBACK"
    GLOBAL_DISTILL = "GLOBAL_DISTILL"
    DECAY_COMPACT = "DECAY_COMPACT"


TIER_PRIORS: Dict[TrustTier, Tuple[int, int]] = {
    TrustTier.T0_HUMAN: (8, 0),
    TrustTier.T1_VERIFIED: (3, 0),
    TrustTier.T2_INFERRED: (0, 0),
    TrustTier.T3_UNTRUSTED: (0, 2),
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


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
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    user_id: Optional[str] = None
    valid_to: Optional[datetime] = None
    version: int = 1
    last_used: datetime = field(default_factory=utc_now)
    uses: int = 1

    @property
    def confidence(self) -> float:
        return (self.pos + 1) / (self.pos + self.neg + 2)


@dataclass
class CandidateClaim:
    """Structured candidate emitted at run end (<=3 per run) referencing tool step provenance."""

    entity_key: str
    text: str
    scope: Scope = Scope.ORG
    source_step_ids: List[str] = field(default_factory=list)
    derived_from: List[str] = field(default_factory=list)
    user_id: Optional[str] = None
    public_artifact_signature: Optional[str] = None


@dataclass
class RecalledClaim:
    claim: Claim
    score: float
    bm25_score: float
    vector_score: float
    contested_hints: List[str] = field(default_factory=list)


@dataclass
class RevisionDecision:
    existing_id: str
    candidate_id: str
    action: RevisionAction
    split_entity_key: Optional[str] = None
    reason: str = ""
