from __future__ import annotations

import re
from typing import Any

import spacy

from chrysalis.config import NEGATION_WORDS
from chrysalis.mrs.base import BaseMR

NEGATION_WORD_PATTERN = "|".join(re.escape(word) for word in NEGATION_WORDS if word != "n't")
NEGATION_PATTERN = re.compile(rf"\b(?:{NEGATION_WORD_PATTERN})\b|n't", re.IGNORECASE)
WORD_TOKEN_PATTERN = re.compile(r"\b\w+\b")
COPULA_FORMS = {"am", "is", "are", "was", "were", "'s", "'re", "'m"}

_NLP: Any = None


def _get_nlp() -> Any:
    global _NLP
    if _NLP is None:
        try:
            _NLP = spacy.load("en_core_web_sm")
        except OSError as exc:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' is required for CHR-SA-001. "
                "Install it with `uv run python -m spacy download en_core_web_sm`."
            ) from exc
    return _NLP


def _get_text(source_input: str | dict) -> str:
    if isinstance(source_input, dict):
        return source_input.get("text", "")
    return source_input


def _get_label(source_input: str | dict) -> int | None:
    if isinstance(source_input, dict):
        return source_input.get("source_label")
    return None


def _is_neutral_label(label: Any) -> bool:
    if label is None:
        return False
    if isinstance(label, str):
        return label.lower() == "neutral"
    return label not in {0, 1}


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


def _find_root(doc: Any) -> Any | None:
    for token in doc:
        if token.dep_ == "ROOT":
            return token
    return None


def _transform_text(text: str) -> str | None:
    doc = _get_nlp()(text)
    root = _find_root(doc)
    if root is None:
        return None

    auxiliaries = sorted(
        [child for child in root.children if child.dep_ in {"aux", "auxpass"}],
        key=lambda token: token.i,
    )
    if auxiliaries:
        return _insert_after_token(doc, auxiliaries[0].i, "not")

    if root.lemma_ == "be" and (root.text.lower() in COPULA_FORMS or root.tag_ in {"VBZ", "VBP", "VBD"}):
        return _insert_after_token(doc, root.i, "not")

    if root.tag_ == "VBZ" and root.lemma_:
        return _replace_token(doc, root.i, f"does not {root.lemma_}")

    if root.tag_ == "VBP" and root.lemma_:
        return _replace_token(doc, root.i, f"do not {root.lemma_}")

    if root.tag_ == "VBD" and root.lemma_:
        return _replace_token(doc, root.i, f"did not {root.lemma_}")

    if root.tag_ == "VB":
        return _replace_token(doc, root.i, f"not {root.text}")

    return None


class CHRSA001(BaseMR):
    @property
    def mr_id(self) -> str:
        return "CHR-SA-001"

    @property
    def subtasks(self) -> list[str]:
        return ["SA"]

    def transform(self, source_input: str | dict, seed: int = 42) -> str | None:
        del seed
        text = _get_text(source_input)
        source_label = _get_label(source_input)

        if not text.strip():
            return None
        if _is_neutral_label(source_label):
            return None
        if _contains_negation(text):
            return None

        return _transform_text(text)

    def check_pass(self, source_output: dict, followup_output: dict) -> bool:
        source_label = source_output["label"]
        source_score = source_output["score"]
        followup_label = followup_output["label"]
        followup_score = followup_output["score"]

        if source_label == 1 and followup_score < 0.5:
            return True
        if source_label == 0 and followup_score > 0.5:
            return True
        return followup_label != source_label or (
            source_label == 1 and followup_score < source_score
        ) or (
            source_label == 0 and followup_score > source_score
        )

    def automated_checks(self, source_input: str | dict, followup_input: str | dict) -> bool:
        source_text = _get_text(source_input)
        followup_text = _get_text(followup_input)
        source_label = _get_label(source_input)

        if _is_neutral_label(source_label):
            return False
        if source_text == followup_text:
            return False
        if _negation_count(followup_text) - _negation_count(source_text) != 1:
            return False
        token_delta = _word_count(followup_text) - _word_count(source_text)
        return token_delta in {1, 2}
