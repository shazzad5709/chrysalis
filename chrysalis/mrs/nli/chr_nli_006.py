from __future__ import annotations

import re
from typing import Any

import spacy

from chrysalis.config import NEGATION_WORDS, NEUTRAL_LABEL
from chrysalis.mrs.base import BaseMR

NEGATION_WORD_PATTERN = "|".join(re.escape(word) for word in NEGATION_WORDS if word != "n't")
NEGATION_PATTERN = re.compile(rf"\b(?:{NEGATION_WORD_PATTERN})\b|n't", re.IGNORECASE)
WORD_TOKEN_PATTERN = re.compile(r"\b\w+\b")
COPULA_FORMS = {"am", "is", "are", "was", "were"}

_NLP: Any = None


def _get_nlp() -> Any:
    global _NLP
    if _NLP is None:
        try:
            _NLP = spacy.load("en_core_web_sm")
        except OSError as exc:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' is required for CHR-NLI-006. "
                "Install it with `uv run python -m spacy download en_core_web_sm`."
            ) from exc
    return _NLP


def _contains_negation(text: str) -> bool:
    return bool(NEGATION_PATTERN.search(text))


def _negation_count(text: str) -> int:
    return len(NEGATION_PATTERN.findall(text))


def _word_count(text: str) -> int:
    return len(WORD_TOKEN_PATTERN.findall(text))


def _serialize_token(token: Any) -> str:
    return f"{token.text}{token.whitespace_}"


def _insert_after_token(doc: Any, token_index: int, inserted_text: str) -> str:
    parts: list[str] = []
    for token in doc:
        if token.i == token_index:
            parts.append(f"{token.text} {inserted_text}{token.whitespace_}")
        else:
            parts.append(_serialize_token(token))
    return "".join(parts)


def _replace_token(doc: Any, token_index: int, replacement_text: str) -> str:
    parts: list[str] = []
    for token in doc:
        if token.i == token_index:
            parts.append(f"{replacement_text}{token.whitespace_}")
        else:
            parts.append(_serialize_token(token))
    return "".join(parts)


def _parse_hypothesis(hypothesis: str) -> Any:
    return _get_nlp()(hypothesis)


def _find_root(doc: Any) -> Any | None:
    for token in doc:
        if token.dep_ == "ROOT":
            return token
    return None


def _transform_hypothesis(hypothesis: str) -> str | None:
    doc = _parse_hypothesis(hypothesis)
    root = _find_root(doc)
    if root is None:
        return None

    auxiliaries = sorted(
        [child for child in root.children if child.dep_ in {"aux", "auxpass"}],
        key=lambda token: token.i,
    )
    if auxiliaries:
        return _insert_after_token(doc, auxiliaries[0].i, "not")

    if root.lemma_ == "be" and root.text.lower() in COPULA_FORMS:
        return _insert_after_token(doc, root.i, "not")

    if root.tag_ == "VBZ" and root.lemma_:
        return _replace_token(doc, root.i, f"does not {root.lemma_}")

    if root.tag_ == "VBP" and root.lemma_:
        return _replace_token(doc, root.i, f"do not {root.lemma_}")

    if root.tag_ == "VBD" and root.lemma_:
        return _replace_token(doc, root.i, f"did not {root.lemma_}")

    return None


class CHRNLI006(BaseMR):
    @property
    def mr_id(self) -> str:
        return "CHR-NLI-006"

    @property
    def subtasks(self) -> list[str]:
        return ["NLI"]

    def transform(self, source_input: str | dict, seed: int = 42) -> dict | None:
        del seed
        if not isinstance(source_input, dict):
            return None

        source_label = source_input.get("source_label")
        premise = source_input.get("premise", "")
        hypothesis = source_input.get("hypothesis", "")

        if source_label == NEUTRAL_LABEL:
            return None
        if not premise or not hypothesis:
            return None
        if _contains_negation(hypothesis):
            return None

        transformed_hypothesis = _transform_hypothesis(hypothesis)
        if transformed_hypothesis is None:
            return None

        return {"premise": premise, "hypothesis": transformed_hypothesis}

    def check_pass(self, source_output: dict, followup_output: dict) -> bool:
        source_label = source_output["label"]
        if source_label == 0:
            return followup_output["label"] == 2
        if source_label == 2:
            return followup_output["label"] == 0
        return False

    def automated_checks(self, source_input: str | dict, followup_input: str | dict) -> bool:
        if not isinstance(source_input, dict) or not isinstance(followup_input, dict):
            return False

        source_label = source_input.get("source_label")
        source_premise = source_input.get("premise", "")
        source_hypothesis = source_input.get("hypothesis", "")
        followup_premise = followup_input.get("premise", "")
        followup_hypothesis = followup_input.get("hypothesis", "")

        if source_label not in {0, 2}:
            return False
        if source_premise != followup_premise:
            return False

        negation_delta = _negation_count(followup_hypothesis) - _negation_count(source_hypothesis)
        if negation_delta != 1:
            return False

        token_delta = _word_count(followup_hypothesis) - _word_count(source_hypothesis)
        if token_delta not in {1, 2}:
            return False

        return followup_hypothesis != source_hypothesis
