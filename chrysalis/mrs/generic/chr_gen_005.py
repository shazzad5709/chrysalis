from __future__ import annotations

import random
import re
from typing import Any

from chrysalis.config import SCORE_TOLERANCE_AIRTIGHT
from chrysalis.mrs.base import BaseMR

TRAILING_PUNCTUATION = ".,!?;:\"'()"
URL_EMAIL_PATTERN = re.compile(r"(@|https?://|www\.)", re.IGNORECASE)


def _clean_token(token: str) -> tuple[str, str]:
    stripped = token.rstrip(TRAILING_PUNCTUATION)
    suffix = token[len(stripped) :]
    return stripped, suffix


def _eligible_indices(tokens: list[str]) -> list[int]:
    eligible: list[int] = []
    for index, token in enumerate(tokens):
        clean, _ = _clean_token(token)
        if len(clean) < 4:
            continue
        if clean.isdigit():
            continue
        if URL_EMAIL_PATTERN.search(clean):
            continue
        eligible.append(index)
    return eligible


def _inject_token(token: str, rng: random.Random) -> str | None:
    clean, suffix = _clean_token(token)
    if len(clean) < 4:
        return None
    position = rng.randint(1, len(clean) - 2)
    return f"{clean[:position]} {clean[position:]}{suffix}"


def _transform_text(text: str, seed: int) -> str | None:
    tokens = text.split()
    eligible = _eligible_indices(tokens)
    if not eligible:
        return None

    rng = random.Random(seed)
    selection_count = max(1, round(len(eligible) * 0.3))
    selected = set(rng.sample(eligible, k=min(selection_count, len(eligible))))

    transformed_tokens: list[str] = []
    changed = False
    for index, token in enumerate(tokens):
        if index not in selected:
            transformed_tokens.append(token)
            continue

        injected = _inject_token(token, rng)
        if injected is None:
            transformed_tokens.append(token)
            continue
        transformed_tokens.append(injected)
        changed = True

    if not changed:
        return None
    return " ".join(transformed_tokens)


class CHRGEN005(BaseMR):
    @property
    def mr_id(self) -> str:
        return "CHR-GEN-005"

    @property
    def subtasks(self) -> list[str]:
        return ["SA", "NLI", "TOPIC"]

    def verify_airtight(self, source: str | dict, followup: str | dict) -> bool:
        if isinstance(source, dict) and isinstance(followup, dict):
            source_joined = f"{source.get('premise', '')}{source.get('hypothesis', '')}"
            followup_joined = f"{followup.get('premise', '')}{followup.get('hypothesis', '')}"
            return followup_joined.replace(" ", "") == source_joined.replace(" ", "")
        return str(followup).replace(" ", "") == str(source).replace(" ", "")

    def transform(self, source_input: str | dict, seed: int = 42) -> str | dict | None:
        self.clear_skip_reason()
        if isinstance(source_input, dict):
            premise = source_input.get("premise", "")
            hypothesis = source_input.get("hypothesis", "")
            transformed_premise = _transform_text(premise, seed)
            transformed_hypothesis = _transform_text(hypothesis, seed)
            if transformed_premise is None:
                self.set_skip_reason("no_eligible_tokens_in_premise")
                return None
            if transformed_hypothesis is None:
                self.set_skip_reason("no_eligible_tokens_in_hypothesis")
                return None
            return {"premise": transformed_premise, "hypothesis": transformed_hypothesis}
        transformed = _transform_text(source_input, seed)
        if transformed is None:
            self.set_skip_reason("no_eligible_tokens")
            return None
        return transformed

    def check_pass(self, source_output: dict, followup_output: dict) -> bool:
        return (
            abs(followup_output["score"] - source_output["score"]) < SCORE_TOLERANCE_AIRTIGHT
            and followup_output["label"] == source_output["label"]
        )

    def automated_checks(self, source_input: str | dict, followup_input: str | dict) -> bool:
        if not self.verify_airtight(source_input, followup_input):
            return False

        if isinstance(source_input, dict) and isinstance(followup_input, dict):
            source_text = f"{source_input.get('premise', '')} {source_input.get('hypothesis', '')}".strip()
            followup_text = f"{followup_input.get('premise', '')} {followup_input.get('hypothesis', '')}".strip()
        else:
            source_text = str(source_input)
            followup_text = str(followup_input)

        if len(followup_text.split()) <= len(source_text.split()):
            return False
        if source_text == followup_text:
            return False
        return True
