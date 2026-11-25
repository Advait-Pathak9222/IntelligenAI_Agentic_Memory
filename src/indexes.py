"""Hybrid BM25 (SQLite FTS5), dense NumPy vector index, and graph adjacency helpers."""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Set, Tuple

import httpx
import numpy as np

from src.models import EdgeType


TOKEN_RE = re.compile(r"[a-z0-9_.:@/-]+")


class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> np.ndarray: ...

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray: ...


class HashingNgramEmbedder:
    """Fast deterministic local embedder combining word tokens and char trigrams."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        lowered = text.lower()
        tokens = TOKEN_RE.findall(lowered)
        for tok in tokens:
            self._accumulate(vec, f"w:{tok}", weight=1.5)
            padded = f"^{tok}$"
            for i in range(max(len(padded) - 2, 0)):
                self._accumulate(vec, f"c:{padded[i:i+3]}", weight=0.4)
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        return vec

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self.embed(t) for t in texts])

    def _accumulate(self, vec: np.ndarray, feature: str, weight: float) -> None:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        h = int.from_bytes(digest, "little")
        idx = h % self.dim
        sign = 1.0 if ((h >> 63) & 1) == 0 else -1.0
        vec[idx] += sign * weight


class OpenAIEmbedder:
    """Production dense embedder calling any OpenAI-compatible embeddings endpoint."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dim: int = 1536,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 10.0,
    ) -> None:
        self.model = model
        self.dim = dim
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "input": list(texts)}
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.base_url}/embeddings", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()["data"]
        arr = np.array([row["embedding"] for row in data], dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


class VectorIndex:
    """Org-partitioned dense cosine index backed by SQLite BLOB persistence."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._org_ids: Dict[str, List[str]] = {}
        self._org_vecs: Dict[str, np.ndarray] = {}

    def upsert(self, org_id: str, claim_id: str, vec: np.ndarray) -> None:
        ids = self._org_ids.setdefault(org_id, [])
        vec_row = vec.astype(np.float32).reshape(1, -1)
        if claim_id in ids:
            idx = ids.index(claim_id)
            self._org_vecs[org_id][idx] = vec_row[0]
            return
        ids.append(claim_id)
        if org_id not in self._org_vecs or self._org_vecs[org_id].size == 0:
            self._org_vecs[org_id] = vec_row
        else:
            self._org_vecs[org_id] = np.vstack([self._org_vecs[org_id], vec_row])

    def remove(self, org_id: str, claim_id: str) -> None:
        ids = self._org_ids.get(org_id)
        if not ids or claim_id not in ids:
            return
        idx = ids.index(claim_id)
        ids.pop(idx)
        mat = self._org_vecs[org_id]
        if len(ids) == 0:
            self._org_vecs[org_id] = np.zeros((0, self.dim), dtype=np.float32)
        else:
            self._org_vecs[org_id] = np.delete(mat, idx, axis=0)

    def search(self, org_ids: Sequence[str], query_vec: np.ndarray, top_k: int = 50) -> Dict[str, float]:
        q = query_vec.astype(np.float32)
        scores: Dict[str, float] = {}
        for oid in org_ids:
            ids = self._org_ids.get(oid)
            mat = self._org_vecs.get(oid)
            if not ids or mat is None or mat.size == 0:
                continue
            sims = mat @ q
            k = min(top_k, len(ids))
            if k == len(ids):
                top_idx = np.argsort(-sims)
            else:
                part = np.argpartition(-sims, k - 1)[:k]
                top_idx = part[np.argsort(-sims[part])]
            for i in top_idx:
                scores[ids[int(i)]] = max(0.0, float(sims[int(i)]))
        return scores


def build_fts5_query(query_text: str) -> str:
    """Convert raw query text into a safe FTS5 OR query without syntax errors."""
    tokens = TOKEN_RE.findall(query_text.lower())
    cleaned = [t.replace('"', "") for t in tokens if len(t) >= 2]
    if not cleaned:
        return ""
    unique = list(dict.fromkeys(cleaned))[:16]
    return " OR ".join(f'"{tok}"' for tok in unique)


def bm25_search_sqlite(
    conn: sqlite3.Connection,
    org_id: str,
    query_text: str,
    limit: int = 50,
) -> Dict[str, float]:
    """Query SQLite FTS5 index with org/global isolation and return normalized [0, 1] BM25 scores."""
    fts_q = build_fts5_query(query_text)
    if not fts_q:
        return {}
    rows = conn.execute(
        """
        SELECT f.claim_id, bm25(claims_fts) AS rank_score
        FROM claims_fts AS f
        JOIN claims AS c ON c.id = f.claim_id
        WHERE claims_fts MATCH ?
          AND (c.org_id = ? OR c.scope = 'global')
          AND c.status IN ('ACTIVE', 'CONTESTED')
        ORDER BY rank_score ASC
        LIMIT ?
        """,
        (fts_q, org_id, limit),
    ).fetchall()
    if not rows:
        return {}
    # SQLite bm25() returns negative values where more negative = stronger match
    raw = {r[0]: abs(float(r[1])) for r in rows}
    max_val = max(raw.values()) or 1.0
    return {cid: val / max_val for cid, val in raw.items()}


def fetch_edges_from(conn: sqlite3.Connection, src_id: str, edge_type: EdgeType) -> List[str]:
    rows = conn.execute(
        "SELECT dst_id FROM claim_edges WHERE src_id = ? AND edge_type = ?",
        (src_id, edge_type.value),
    ).fetchall()
    return [r[0] for r in rows]


def fetch_edges_to(conn: sqlite3.Connection, dst_id: str, edge_type: EdgeType) -> List[str]:
    rows = conn.execute(
        "SELECT src_id FROM claim_edges WHERE dst_id = ? AND edge_type = ?",
        (dst_id, edge_type.value),
    ).fetchall()
    return [r[0] for r in rows]
