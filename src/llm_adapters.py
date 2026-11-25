"""Pluggable LLM judge adapters for batched sleep-time revision and feedback attribution."""

from __future__ import annotations

import json
import os
from typing import List, Optional, Protocol, Sequence, Tuple

import httpx

from src.models import Claim, RevisionAction, RevisionDecision


class LLMJudge(Protocol):
    llm_calls_made: int

    def batched_revise_conflicts(
        self, pairs: Sequence[Tuple[Claim, Claim]]
    ) -> List[RevisionDecision]: ...

    def attribute_feedback(
        self, report_text: str, human_note: str, recalled_claims: Sequence[Claim]
    ) -> List[str]: ...

    def distill_global_rule(self, signature: str, scrubbed_samples: Sequence[str]) -> str: ...


class LocalHeuristicJudge:
    """Offline judge simulating batched LLM calls and tracking exact call counts."""

    def __init__(self) -> None:
        self.llm_calls_made = 0

    def batched_revise_conflicts(
        self, pairs: Sequence[Tuple[Claim, Claim]]
    ) -> List[RevisionDecision]:
        if not pairs:
            return []
        # Counts 1 LLM call per batch (up to 5 pairs/call)
        self.llm_calls_made += (len(pairs) + 4) // 5
        decisions: List[RevisionDecision] = []
        for existing, candidate in pairs:
            if existing.text.strip().lower() == candidate.text.strip().lower():
                decisions.append(
                    RevisionDecision(
                        existing_id=existing.id,
                        candidate_id=candidate.id,
                        action=RevisionAction.CONFIRM,
                        reason="Exact semantic equivalence",
                    )
                )
            elif "staging" in candidate.text.lower() and "prod" in existing.text.lower():
                decisions.append(
                    RevisionDecision(
                        existing_id=existing.id,
                        candidate_id=candidate.id,
                        action=RevisionAction.COEXIST,
                        split_entity_key=f"{candidate.entity_key}:staging",
                        reason="Distinct environment contexts",
                    )
                )
            else:
                decisions.append(
                    RevisionDecision(
                        existing_id=existing.id,
                        candidate_id=candidate.id,
                        action=RevisionAction.SUPERSEDE,
                        reason="Newer observation conflicts with prior state",
                    )
                )
        return decisions

    def attribute_feedback(
        self, report_text: str, human_note: str, recalled_claims: Sequence[Claim]
    ) -> List[str]:
        self.llm_calls_made += 1
        note_tokens = {tok.lower() for tok in human_note.replace(":", " ").split() if len(tok) > 2}
        matched: List[str] = []
        for claim in recalled_claims:
            c_tokens = {tok.lower() for tok in f"{claim.entity_key} {claim.text}".split()}
            if note_tokens & c_tokens:
                matched.append(claim.id)
        if not matched and recalled_claims:
            # Fallback to the lowest-trust recalled claim
            sorted_claims = sorted(recalled_claims, key=lambda c: (-int(c.trust_tier), c.confidence))
            matched.append(sorted_claims[0].id)
        return matched

    def distill_global_rule(self, signature: str, scrubbed_samples: Sequence[str]) -> str:
        self.llm_calls_made += 1
        if scrubbed_samples:
            return scrubbed_samples[0]
        return f"Verified global procedural fix for public artifact signature {signature}."


class OpenAICompatibleJudge:
    """Production LLM judge calling any OpenAI-compatible Chat Completions endpoint."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 20.0,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.transport = transport
        self.llm_calls_made = 0

    def _chat_json(self, system_prompt: str, user_prompt: str) -> dict:
        self.llm_calls_made += 1
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        with httpx.Client(timeout=self.timeout, transport=self.transport) as client:
            resp = client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
        return json.loads(content)

    def batched_revise_conflicts(
        self, pairs: Sequence[Tuple[Claim, Claim]]
    ) -> List[RevisionDecision]:
        results: List[RevisionDecision] = []
        for i in range(0, len(pairs), 5):
            chunk = pairs[i : i + 5]
            items = [
                {
                    "existing_id": old.id,
                    "existing_text": old.text,
                    "existing_valid_from": old.valid_from.isoformat(),
                    "candidate_id": new.id,
                    "candidate_text": new.text,
                    "candidate_valid_from": new.valid_from.isoformat(),
                }
                for old, new in chunk
            ]
            data = self._chat_json(
                "You resolve belief conflicts in an agent memory store. "
                "Return JSON {'decisions': [{'existing_id','candidate_id','action','split_entity_key','reason'}]} "
                "where action is one of CONFIRM, SUPERSEDE, COEXIST, REJECT.",
                json.dumps({"pairs": items}),
            )
            for row in data.get("decisions", []):
                results.append(
                    RevisionDecision(
                        existing_id=row["existing_id"],
                        candidate_id=row["candidate_id"],
                        action=RevisionAction(row["action"]),
                        split_entity_key=row.get("split_entity_key"),
                        reason=row.get("reason", ""),
                    )
                )
        return results

    def attribute_feedback(
        self, report_text: str, human_note: str, recalled_claims: Sequence[Claim]
    ) -> List[str]:
        payload = {
            "report_text": report_text,
            "human_note": human_note,
            "recalled_claims": [{"id": c.id, "entity_key": c.entity_key, "text": c.text} for c in recalled_claims],
        }
        data = self._chat_json(
            "Identify which recalled_claims caused the error described in human_note. "
            "Return JSON {'faulty_claim_ids': [...]}.",
            json.dumps(payload),
        )
        return list(data.get("faulty_claim_ids", []))

    def distill_global_rule(self, signature: str, scrubbed_samples: Sequence[str]) -> str:
        data = self._chat_json(
            "Synthesize one concise, PII-free procedural fix rule from scrubbed public-artifact samples. "
            "Return JSON {'rule': '...'}",
            json.dumps({"signature": signature, "samples": list(scrubbed_samples[:10])}),
        )
        return str(data["rule"])
