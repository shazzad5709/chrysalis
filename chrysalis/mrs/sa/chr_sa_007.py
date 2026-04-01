from __future__ import annotations

import re
from typing import Any

import spacy

from chrysalis.config import ABBREVIATION_SAFELIST
from chrysalis.mrs.base import BaseMR

WORD_TOKEN_PATTERN = re.compile(r"\b\w+\b")
INITIALS_PATTERN = re.compile(ABBREVIATION_SAFELIST["initials_pattern"])
MULTI_INITIALS_PATTERN = re.compile(r"(?:[A-Za-z]\.){2,}$")
ABBREVIATIONS = set(
    ABBREVIATION_SAFELIST["titles"]
    + ABBREVIATION_SAFELIST["geographic"]
    + ABBREVIATION_SAFELIST["misc"]
    + ["Ms."]
)
ABBREVIATIONS_LOWER = {value.lower() for value in ABBREVIATIONS}

_SENTENCIZER: Any = None


def _get_sentencizer() -> Any:
    global _SENTENCIZER
    if _SENTENCIZER is None:
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        _SENTENCIZER = nlp
    return _SENTENCIZER


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


def _ends_with_strong_punctuation(text: str) -> bool:
    stripped = text.rstrip()
    return stripped.endswith("!") or stripped.endswith("?") or stripped.endswith("...")


def _sentence_final_period_indices(text: str) -> list[int]:
    indices: list[int] = []
    doc = _get_sentencizer()(text)
    for sent in doc.sents:
        sent_text = text[sent.start_char : sent.end_char]
        stripped = sent_text.rstrip()
        if not stripped.endswith("."):
            continue

        period_index = sent.start_char + len(stripped) - 1
        previous_match = re.search(r"([A-Za-z]+)\.$", stripped)
        if previous_match is None:
            continue

        token_with_period = f"{previous_match.group(1)}."
        if (
            token_with_period.lower() in ABBREVIATIONS_LOWER
            or INITIALS_PATTERN.fullmatch(token_with_period)
            or MULTI_INITIALS_PATTERN.search(stripped)
        ):
            continue

        indices.append(period_index)
    return indices


class CHRSA007(BaseMR):
    @property
    def mr_id(self) -> str:
        return "CHR-SA-007"

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
        if "..." in text or _ends_with_strong_punctuation(text):
            return None

        indices = _sentence_final_period_indices(text)
        if not indices:
            return None

        chars = list(text)
        for index in indices:
            chars[index] = "!"
        return "".join(chars)

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

        if source_text == followup_text:
            return False
        if _word_count(source_text) != _word_count(followup_text):
            return False
        if followup_text.count("!") <= source_text.count("!"):
            return False
        if re.search(r"\b(?:Dr|Mr|Mrs|Prof|Sr|Jr)\.!", followup_text):
            return False
        if re.search(r"\b[A-Za-z]\.!", followup_text):
            return False
        return re.search(r"\b(?:[A-Za-z]\.)+[A-Za-z]!", followup_text) is None
