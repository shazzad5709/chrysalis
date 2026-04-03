from __future__ import annotations

from chrysalis.mrs.base import BaseMR
from chrysalis.mrs.nli.chr_nli_004 import (
    _has_biological_sex_content,
    _has_gender_restrictive_language,
    _pronoun_agreement_ok,
    _validate_cross_gender_component,
    gender_swap_with_reason,
)


class CHRNLI005(BaseMR):
    @property
    def mr_id(self) -> str:
        return "CHR-NLI-005"

    @property
    def subtasks(self) -> list[str]:
        return ["NLI"]

    def transform(self, source_input: str | dict, seed: int = 42) -> dict | None:
        if not isinstance(source_input, dict):
            self.set_skip_reason("invalid_input_type")
            return None
        self.clear_skip_reason()
        transformed, reason = gender_swap_with_reason(
            premise=source_input.get("premise", ""),
            hypothesis=source_input.get("hypothesis", ""),
            mode="cross_gender",
            seed=seed,
        )
        if transformed is None and reason is not None:
            self.set_skip_reason(reason)
        return transformed

    def check_pass(self, source_output: dict, followup_output: dict) -> tuple[bool, bool]:
        passed = source_output["label"] == followup_output["label"]
        return passed, not passed

    def automated_checks(self, source_input: str | dict, followup_input: str | dict) -> bool:
        if not isinstance(source_input, dict) or not isinstance(followup_input, dict):
            return False

        source_premise = source_input.get("premise", "")
        source_hypothesis = source_input.get("hypothesis", "")
        followup_premise = followup_input.get("premise", "")
        followup_hypothesis = followup_input.get("hypothesis", "")

        if _has_gender_restrictive_language(source_premise, source_hypothesis):
            return False
        if _has_biological_sex_content(source_premise, source_hypothesis):
            return False
        if source_premise == followup_premise and source_hypothesis == followup_hypothesis:
            return False
        if not _validate_cross_gender_component(source_premise, followup_premise):
            return False
        if not _validate_cross_gender_component(source_hypothesis, followup_hypothesis):
            return False
        return _pronoun_agreement_ok(followup_premise) and _pronoun_agreement_ok(followup_hypothesis)
