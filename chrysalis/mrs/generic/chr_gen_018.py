from __future__ import annotations

import logging

from chrysalis.config import SCORE_TOLERANCE_AIRTIGHT
from chrysalis.mrs.base import BaseMR

logger = logging.getLogger(__name__)


def _transform_text(text: str, variant: str) -> str | None:
    if variant == "uppercase":
        return None if text == text.upper() else text.upper()
    if variant == "lowercase":
        return None if text == text.lower() else text.lower()
    raise ValueError(f"Unsupported variant: {variant}")


class CHRGEN018(BaseMR):
    @property
    def mr_id(self) -> str:
        return "CHR-GEN-018"

    @property
    def subtasks(self) -> list[str]:
        return ["SA", "NLI", "TOPIC"]

    def check_tokenizer_casing(self, tokenizer) -> bool:
        sample = "Case Check"
        original = tokenizer.encode(sample, add_special_tokens=False)
        upper = tokenizer.encode(sample.upper(), add_special_tokens=False)
        differs = original != upper
        if not differs:
            logger.warning("Tokenizer appears uncased; CHR-GEN-018 should be excluded.")
        return differs

    def transform(
        self,
        source_input: str | dict,
        seed: int = 42,
        variant: str = "uppercase",
    ) -> str | dict | None:
        del seed
        self.clear_skip_reason()
        if isinstance(source_input, dict):
            premise = _transform_text(source_input.get("premise", ""), variant)
            hypothesis = _transform_text(source_input.get("hypothesis", ""), variant)
            if premise is None and hypothesis is None:
                self.set_skip_reason(f"already_{variant}")
                return None
            return {
                "premise": source_input.get("premise", "") if premise is None else premise,
                "hypothesis": source_input.get("hypothesis", "") if hypothesis is None else hypothesis,
                "variant": variant,
            }
        transformed = _transform_text(source_input, variant)
        if transformed is None:
            self.set_skip_reason(f"already_{variant}")
            return None
        return transformed

    def transform_both(self, source_input: str | dict) -> list[dict | str | None]:
        return [
            self.transform(source_input, variant="uppercase"),
            self.transform(source_input, variant="lowercase"),
        ]

    def check_pass(self, source_output: dict, followup_output: dict) -> bool:
        return (
            abs(followup_output["score"] - source_output["score"]) < SCORE_TOLERANCE_AIRTIGHT
            and followup_output["label"] == source_output["label"]
        )

    def automated_checks(self, source_input: str | dict, followup_input: str | dict) -> bool:
        if isinstance(source_input, dict) and isinstance(followup_input, dict):
            source_text = f"{source_input.get('premise', '')} {source_input.get('hypothesis', '')}"
            followup_text = f"{followup_input.get('premise', '')} {followup_input.get('hypothesis', '')}"
        else:
            source_text = str(source_input)
            followup_text = str(followup_input)

        if len(source_text) != len(followup_text):
            return False

        changed = False
        for source_char, followup_char in zip(source_text, followup_text):
            if source_char.isalpha():
                if source_char == followup_char:
                    continue
                changed = True
            elif source_char != followup_char:
                return False

        return changed
