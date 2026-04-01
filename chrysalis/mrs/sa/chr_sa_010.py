from __future__ import annotations

import re
from typing import Any

import spacy

from chrysalis.mrs.base import BaseMR

TARGET_TAGS = {"NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS"}
WORD_TOKEN_PATTERN = re.compile(r"\b\w+\b")

_NLP: Any = None


def _get_nlp() -> Any:
    global _NLP
    if _NLP is None:
        try:
            _NLP = spacy.load("en_core_web_sm")
        except OSError as exc:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' is required for CHR-SA-010. "
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


def _word_count(text: str) -> int:
    return len(WORD_TOKEN_PATTERN.findall(text))


def _uppercase_targets(text: str) -> tuple[str, bool]:
    doc = _get_nlp()(text)
    force_upper: set[int] = set()
    for index in range(1, len(doc) - 1):
        token = doc[index]
        left = doc[index - 1]
        right = doc[index + 1]
        if (
            token.text == "-"
            and left.text.isalpha()
            and right.text.isalpha()
            and (left.tag_ in TARGET_TAGS or right.tag_ in TARGET_TAGS or right.dep_ == "amod")
        ):
            force_upper.update({index - 1, index + 1})

    parts: list[str] = []
    changed = False
    for token in doc:
        if (token.tag_ in TARGET_TAGS or token.i in force_upper) and not token.text.isupper():
            parts.append(f"{token.text.upper()}{token.whitespace_}")
            changed = True
        else:
            parts.append(f"{token.text}{token.whitespace_}")
    return "".join(parts), changed


class CHRSA010(BaseMR):
    @property
    def mr_id(self) -> str:
        return "CHR-SA-010"

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

        transformed, changed = _uppercase_targets(text)
        return transformed if changed else None

    def check_pass(self, source_output: dict, followup_output: dict) -> bool:
        source_label = source_output["label"]
        if source_label == 1 and followup_output["score"] >= source_output["score"]:
            return True
        if source_label == 0 and followup_output["score"] <= source_output["score"]:
            return True
        return followup_output["label"] == source_label

    def automated_checks(self, source_input: str | dict, followup_input: str | dict) -> bool:
        source_text = _get_text(source_input)
        followup_text = _get_text(followup_input)
        expected, changed = _uppercase_targets(source_text)
        if not changed:
            return False
        if _word_count(source_text) != _word_count(followup_text):
            return False
        return expected == followup_text
