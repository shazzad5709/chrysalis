from __future__ import annotations

import random
import re
from typing import Any

import spacy

from chrysalis.config import INTENSIFIER_POOL
from chrysalis.mrs.base import BaseMR

WORD_TOKEN_PATTERN = re.compile(r"\b\w+\b")
NEGATION_SCOPE_WORDS = {"not", "n't", "never", "no", "barely", "hardly", "scarcely"}
ADJECTIVE_TAGS = {"JJ", "JJR", "JJS"}
ADVERB_TAGS = {"RB", "RBR", "RBS"}
MAXIMUM_WORDS = {"best", "worst"}

_NLP: Any = None


def _get_nlp() -> Any:
    global _NLP
    if _NLP is None:
        try:
            _NLP = spacy.load("en_core_web_sm")
        except OSError as exc:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' is required for CHR-SA-008. "
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


def _has_negation_scope(doc: Any, token: Any) -> bool:
    start = max(0, token.i - 3)
    return any(doc[index].text.lower() in NEGATION_SCOPE_WORDS for index in range(start, token.i))


def _already_modified(token: Any) -> bool:
    return any(child.dep_ == "advmod" for child in token.children)


def _is_extreme(token: Any, doc: Any) -> bool:
    if token.tag_ in {"JJR", "JJS", "RBR", "RBS"}:
        return True
    if token.text.lower() in MAXIMUM_WORDS:
        return True
    return token.i > 0 and doc[token.i - 1].text.lower() == "most"


def _valid_token(doc: Any, token: Any) -> bool:
    return not _already_modified(token) and not _has_negation_scope(doc, token) and not _is_extreme(token, doc)


def _select_target(doc: Any) -> Any | None:
    adjective_tokens = [token for token in doc if token.tag_ in ADJECTIVE_TAGS]
    adjectives = [token for token in adjective_tokens if _valid_token(doc, token)]
    for token in adjectives:
        if token.dep_ in {"acomp", "oprd"}:
            return token
    for token in adjectives:
        if token.dep_ == "amod":
            return token
    if adjectives:
        return adjectives[0]
    if adjective_tokens:
        return None

    adverbs = [token for token in doc if token.tag_ in ADVERB_TAGS and _valid_token(doc, token)]
    return adverbs[0] if adverbs else None


def _insert_before_token(doc: Any, token_index: int, inserted_text: str) -> str:
    parts: list[str] = []
    for token in doc:
        if token.i == token_index:
            parts.append(f"{inserted_text} {token.text}{token.whitespace_}")
        else:
            parts.append(f"{token.text}{token.whitespace_}")
    return "".join(parts)


def _find_inserted_intensifier(source_text: str, followup_text: str) -> str | None:
    source_words = WORD_TOKEN_PATTERN.findall(source_text)
    followup_words = WORD_TOKEN_PATTERN.findall(followup_text)
    if len(followup_words) != len(source_words) + 1:
        return None

    i = 0
    j = 0
    inserted: str | None = None
    while i < len(source_words) and j < len(followup_words):
        if source_words[i].lower() == followup_words[j].lower():
            i += 1
            j += 1
            continue

        if (
            inserted is None
            and followup_words[j].lower() in INTENSIFIER_POOL
            and j + 1 < len(followup_words)
            and source_words[i].lower() == followup_words[j + 1].lower()
        ):
            inserted = followup_words[j].lower()
            j += 1
            continue
        return None

    if inserted is None and j < len(followup_words):
        if followup_words[j].lower() in INTENSIFIER_POOL:
            inserted = followup_words[j].lower()
            j += 1

    if inserted is None or i != len(source_words) or j != len(followup_words):
        return None
    return inserted


class CHRSA008(BaseMR):
    @property
    def mr_id(self) -> str:
        return "CHR-SA-008"

    @property
    def subtasks(self) -> list[str]:
        return ["SA"]

    def transform(self, source_input: str | dict, seed: int = 42) -> str | None:
        text = _get_text(source_input)
        source_label = _get_label(source_input)
        if not text.strip():
            return None
        if _is_neutral_label(source_label):
            return None

        doc = _get_nlp()(text)
        target = _select_target(doc)
        if target is None:
            return None

        rng = random.Random(seed)
        intensifier = rng.choice(INTENSIFIER_POOL)
        return _insert_before_token(doc, target.i, intensifier)

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
        if _word_count(followup_text) != _word_count(source_text) + 1:
            return False
        return _find_inserted_intensifier(source_text, followup_text) in INTENSIFIER_POOL
