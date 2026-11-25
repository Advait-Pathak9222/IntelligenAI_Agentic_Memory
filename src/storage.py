"""SQLite WAL-backed persistence, FTS5 BM25 sync, vector cache, and durable job queue."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import sqlite3
import threading
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from src.indexes import Embedder, VectorIndex
from src.models import Claim, ClaimStatus, EdgeType, JobType, Scope, TrustTier


class ConcurrencyConflictError(Exception):
    """Raised on optimistic CAS version mismatch."""


def _dt_to_iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt is not None else None


def _iso_to_dt(raw: Optional[str]) -> Optional[datetime]:
    return datetime.fromisoformat(raw) if raw is not None else None


class ClaimRepository:
    """Thread-safe SQLite WAL repository syncing structured claims, FTS5, vectors, graph edges, and jobs."""

    def __init__(self, db_path: str, embedder: Embedder) -> None:
        self.db_path = db_path
        self.embedder = embedder
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30.0)
        self.conn.row_factory = sqlite3.Row
        self._configure_pragmas()
        self._init_schema()
        self.vector_index = VectorIndex(dim=embedder.dim)
        self._hydrate_vector_index()

    def _configure_pragmas(self) -> None:
        with self._lock:
            self.conn.execute("PRAGMA journal_mode=WAL;")
            self.conn.execute("PRAGMA synchronous=NORMAL;")
            self.conn.execute("PRAGMA foreign_keys=ON;")
            self.conn.execute("PRAGMA busy_timeout=5000;")

    def _init_schema(self) -> None:
        with self._lock:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS claims (
                    id TEXT PRIMARY KEY,
                    org_id TEXT NOT NULL,
                    user_id TEXT,
                    scope TEXT NOT NULL,
                    entity_key TEXT NOT NULL,
                    text TEXT NOT NULL,
                    trust_tier INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    pos INTEGER NOT NULL,
                    neg INTEGER NOT NULL,
                    valid_from TEXT NOT NULL,
                    valid_to TEXT,
                    run_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    last_used TEXT NOT NULL,
                    uses INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_claims_org_entity_status
                    ON claims(org_id, entity_key, status);
                CREATE INDEX IF NOT EXISTS idx_claims_scope_status
                    ON claims(scope, status);

                CREATE VIRTUAL TABLE IF NOT EXISTS claims_fts USING fts5(
                    claim_id UNINDEXED,
                    org_id UNINDEXED,
                    entity_key,
                    text
                );

                CREATE TABLE IF NOT EXISTS claim_vectors (
                    claim_id TEXT PRIMARY KEY,
                    org_id TEXT NOT NULL,
                    vec BLOB NOT NULL,
                    FOREIGN KEY (claim_id) REFERENCES claims(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS claim_edges (
                    src_id TEXT NOT NULL,
                    dst_id TEXT NOT NULL,
                    edge_type TEXT NOT NULL,
                    PRIMARY KEY (src_id, dst_id, edge_type)
                );
                CREATE INDEX IF NOT EXISTS idx_edges_dst ON claim_edges(dst_id, edge_type);

                CREATE TABLE IF NOT EXISTS org_allowlists (
                    org_id TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    PRIMARY KEY (org_id, domain)
                );

                CREATE TABLE IF NOT EXISTS global_observations (
                    signature TEXT NOT NULL,
                    org_id TEXT NOT NULL,
                    scrubbed_rule TEXT NOT NULL,
                    PRIMARY KEY (signature, org_id)
                );

                CREATE TABLE IF NOT EXISTS async_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_type TEXT NOT NULL,
                    org_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'PENDING',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_status ON async_jobs(status, job_type, id);
                """
            )
            self.conn.commit()

    def _hydrate_vector_index(self) -> None:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT v.claim_id, v.org_id, v.vec
                FROM claim_vectors AS v
                JOIN claims AS c ON c.id = v.claim_id
                WHERE c.status IN ('ACTIVE', 'CONTESTED')
                """
            ).fetchall()
            for r in rows:
                vec = np.frombuffer(r["vec"], dtype=np.float32)
                self.vector_index.upsert(r["org_id"], r["claim_id"], vec)

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def register_verified_domain(self, org_id: str, domain: str) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT OR IGNORE INTO org_allowlists(org_id, domain) VALUES (?, ?)",
                (org_id, domain.lower()),
            )
            self.conn.commit()

    def is_verified_domain(self, org_id: str, domain: str) -> bool:
        with self._lock:
            row = self.conn.execute(
                "SELECT 1 FROM org_allowlists WHERE org_id = ? AND domain = ?",
                (org_id, domain.lower()),
            ).fetchone()
            return row is not None

    def get_claim(self, claim_id: str) -> Optional[Claim]:
        with self._lock:
            row = self.conn.execute("SELECT * FROM claims WHERE id = ?", (claim_id,)).fetchone()
            return self._row_to_claim(row) if row is not None else None

    def get_claims_batch(self, claim_ids: Sequence[str]) -> Dict[str, Claim]:
        if not claim_ids:
            return {}
        placeholders = ",".join("?" for _ in claim_ids)
        with self._lock:
            rows = self.conn.execute(
                f"SELECT * FROM claims WHERE id IN ({placeholders})", tuple(claim_ids)
            ).fetchall()
            return {r["id"]: self._row_to_claim(r) for r in rows}

    def find_active_by_entity(self, org_id: str, entity_key: str) -> Optional[Claim]:
        with self._lock:
            row = self.conn.execute(
                """
                SELECT * FROM claims
                WHERE org_id = ? AND entity_key = ? AND status = 'ACTIVE'
                ORDER BY valid_from DESC LIMIT 1
                """,
                (org_id, entity_key),
            ).fetchone()
            return self._row_to_claim(row) if row is not None else None

    def insert_claim(self, claim: Claim, vec: Optional[np.ndarray] = None) -> None:
        if vec is None:
            vec = self.embedder.embed(f"{claim.entity_key} {claim.text}")
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO claims(
                    id, org_id, user_id, scope, entity_key, text, trust_tier, status,
                    pos, neg, valid_from, valid_to, run_id, version, last_used, uses
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    claim.id,
                    claim.org_id,
                    claim.user_id,
                    claim.scope.value,
                    claim.entity_key,
                    claim.text,
                    int(claim.trust_tier),
                    claim.status.value,
                    claim.pos,
                    claim.neg,
                    _dt_to_iso(claim.valid_from),
                    _dt_to_iso(claim.valid_to),
                    claim.run_id,
                    claim.version,
                    _dt_to_iso(claim.last_used),
                    claim.uses,
                ),
            )
            self.conn.execute(
                "INSERT INTO claims_fts(claim_id, org_id, entity_key, text) VALUES (?, ?, ?, ?)",
                (claim.id, claim.org_id, claim.entity_key, claim.text),
            )
            self.conn.execute(
                "INSERT INTO claim_vectors(claim_id, org_id, vec) VALUES (?, ?, ?)",
                (claim.id, claim.org_id, vec.astype(np.float32).tobytes()),
            )
            self.conn.commit()
            if claim.status in (ClaimStatus.ACTIVE, ClaimStatus.CONTESTED):
                self.vector_index.upsert(claim.org_id, claim.id, vec)

    def update_claim_cas(self, claim: Claim, expected_version: int) -> Claim:
        """Optimistic CAS commit; raises ConcurrencyConflictError if modified concurrently."""
        new_version = expected_version + 1
        with self._lock:
            cur = self.conn.execute(
                """
                UPDATE claims
                SET entity_key = ?, status = ?, pos = ?, neg = ?, valid_to = ?,
                    last_used = ?, uses = ?, version = ?
                WHERE id = ? AND version = ?
                """,
                (
                    claim.entity_key,
                    claim.status.value,
                    claim.pos,
                    claim.neg,
                    _dt_to_iso(claim.valid_to),
                    _dt_to_iso(claim.last_used),
                    claim.uses,
                    new_version,
                    claim.id,
                    expected_version,
                ),
            )
            if cur.rowcount != 1:
                self.conn.rollback()
                raise ConcurrencyConflictError(
                    f"CAS conflict on claim {claim.id}: expected v{expected_version}"
                )
            self.conn.commit()
            claim.version = new_version
            if claim.status in (ClaimStatus.ACTIVE, ClaimStatus.CONTESTED):
                row = self.conn.execute(
                    "SELECT vec FROM claim_vectors WHERE claim_id = ?", (claim.id,)
                ).fetchone()
                if row is not None:
                    self.vector_index.upsert(
                        claim.org_id, claim.id, np.frombuffer(row["vec"], dtype=np.float32)
                    )
            else:
                self.vector_index.remove(claim.org_id, claim.id)
            return claim

    def add_edge(self, src_id: str, dst_id: str, edge_type: EdgeType) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT OR IGNORE INTO claim_edges(src_id, dst_id, edge_type) VALUES (?, ?, ?)",
                (src_id, dst_id, edge_type.value),
            )
            self.conn.commit()

    def get_neighbors(self, node_id: str, edge_type: EdgeType, direction: str = "out") -> List[str]:
        with self._lock:
            if direction == "out":
                rows = self.conn.execute(
                    "SELECT dst_id FROM claim_edges WHERE src_id = ? AND edge_type = ?",
                    (node_id, edge_type.value),
                ).fetchall()
            else:
                rows = self.conn.execute(
                    "SELECT src_id FROM claim_edges WHERE dst_id = ? AND edge_type = ?",
                    (node_id, edge_type.value),
                ).fetchall()
            return [r[0] for r in rows]

    def delete_claim(self, claim_id: str, org_id: str) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM claims_fts WHERE claim_id = ?", (claim_id,))
            self.conn.execute("DELETE FROM claim_edges WHERE src_id = ? OR dst_id = ?", (claim_id, claim_id))
            self.conn.execute("DELETE FROM claims WHERE id = ?", (claim_id,))
            self.conn.commit()
            self.vector_index.remove(org_id, claim_id)

    def enqueue_job(self, job_type: JobType, org_id: str, payload: dict) -> int:
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock:
            cur = self.conn.execute(
                """
                INSERT INTO async_jobs(job_type, org_id, payload_json, status, created_at, updated_at)
                VALUES (?, ?, ?, 'PENDING', ?, ?)
                """,
                (job_type.value, org_id, json.dumps(payload), now_iso, now_iso),
            )
            self.conn.commit()
            return int(cur.lastrowid)

    def claim_pending_jobs(self, job_type: Optional[JobType] = None, limit: int = 20) -> List[dict]:
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock:
            if job_type is not None:
                rows = self.conn.execute(
                    """
                    SELECT * FROM async_jobs
                    WHERE status = 'PENDING' AND job_type = ?
                    ORDER BY id ASC LIMIT ?
                    """,
                    (job_type.value, limit),
                ).fetchall()
            else:
                rows = self.conn.execute(
                    "SELECT * FROM async_jobs WHERE status = 'PENDING' ORDER BY id ASC LIMIT ?",
                    (limit,),
                ).fetchall()
            if not rows:
                return []
            ids = [r["id"] for r in rows]
            placeholders = ",".join("?" for _ in ids)
            self.conn.execute(
                f"UPDATE async_jobs SET status = 'RUNNING', attempts = attempts + 1, updated_at = ? WHERE id IN ({placeholders})",
                (now_iso, *ids),
            )
            self.conn.commit()
            return [
                {
                    "id": r["id"],
                    "job_type": JobType(r["job_type"]),
                    "org_id": r["org_id"],
                    "payload": json.loads(r["payload_json"]),
                }
                for r in rows
            ]

    def finish_job(self, job_id: int, status: str = "DONE") -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self.conn.execute(
                "UPDATE async_jobs SET status = ?, updated_at = ? WHERE id = ?",
                (status, now_iso, job_id),
            )
            self.conn.commit()

    def record_global_observation(self, signature: str, org_id: str, scrubbed_rule: str) -> Tuple[int, List[str]]:
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO global_observations(signature, org_id, scrubbed_rule) VALUES (?, ?, ?)",
                (signature, org_id, scrubbed_rule),
            )
            self.conn.commit()
            rows = self.conn.execute(
                "SELECT org_id, scrubbed_rule FROM global_observations WHERE signature = ?",
                (signature,),
            ).fetchall()
            return len(rows), [r["scrubbed_rule"] for r in rows]

    def list_org_claims(self, org_id: str) -> List[Claim]:
        with self._lock:
            rows = self.conn.execute("SELECT * FROM claims WHERE org_id = ?", (org_id,)).fetchall()
            return [self._row_to_claim(r) for r in rows]

    def _row_to_claim(self, r: sqlite3.Row) -> Claim:
        return Claim(
            id=r["id"],
            org_id=r["org_id"],
            user_id=r["user_id"],
            scope=Scope(r["scope"]),
            entity_key=r["entity_key"],
            text=r["text"],
            trust_tier=TrustTier(r["trust_tier"]),
            status=ClaimStatus(r["status"]),
            pos=int(r["pos"]),
            neg=int(r["neg"]),
            valid_from=_iso_to_dt(r["valid_from"]),  # type: ignore[arg-type]
            valid_to=_iso_to_dt(r["valid_to"]),
            run_id=r["run_id"],
            version=int(r["version"]),
            last_used=_iso_to_dt(r["last_used"]),  # type: ignore[arg-type]
            uses=int(r["uses"]),
        )
