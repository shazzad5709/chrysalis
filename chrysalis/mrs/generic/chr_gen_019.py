from __future__ import annotations

import logging
import random
import re
from collections import deque

from nltk.corpus import words

from chrysalis.config import SCORE_TOLERANCE_NEARFORMAL
from chrysalis.mrs.base import BaseMR

logger = logging.getLogger(__name__)

KEYBOARD_ADJACENCY = {
    "q": ["w", "a", "s"],
    "w": ["q", "e", "a", "s", "d"],
    "e": ["w", "r", "s", "d", "f"],
    "r": ["e", "t", "d", "f", "g"],
    "t": ["r", "y", "f", "g", "h"],
    "y": ["t", "u", "g", "h", "j"],
    "u": ["y", "i", "h", "j", "k"],
    "i": ["u", "o", "j", "k", "l"],
    "o": ["i", "p", "k", "l"],
    "p": ["o", "l"],
    "a": ["q", "w", "s", "z", "x"],
    "s": ["a", "w", "e", "d", "z", "x", "c"],
    "d": ["s", "e", "r", "f", "x", "c", "v"],
    "f": ["d", "r", "t", "g", "c", "v", "b"],
    "g": ["f", "t", "y", "h", "v", "b", "n"],
    "h": ["g", "y", "u", "j", "b", "n", "m"],
    "j": ["h", "u", "i", "k", "n", "m"],
    "k": ["j", "i", "o", "l", "m"],
    "l": ["k", "o", "p"],
    "z": ["a", "s", "x"],
    "x": ["z", "s", "d", "c"],
    "c": ["x", "d", "f", "v"],
    "v": ["c", "f", "g", "b"],
    "b": ["v", "g", "h", "n"],
    "n": ["b", "h", "j", "m"],
    "m": ["n", "j", "k"],
}

TRAILING_PUNCTUATION = ".,!?;:\"'()"
URL_EMAIL_PATTERN = re.compile(r"(@|https?://|www\.)", re.IGNORECASE)
word_set = set(w.lower() for w in words.words())
_skip_window: deque[bool] = deque(maxlen=100)


def _track_skip(skipped: bool) -> None:
    _skip_window.append(skipped)
    if len(_skip_window) == _skip_window.maxlen and sum(_skip_window) / len(_skip_window) > 0.15:
        logger.warning("CHR-GEN-019 skip rate exceeded 15%%; consider a frequency-filtered word list.")


def _clean_token(token: str) -> tuple[str, str]:
    stripped = token.rstrip(TRAILING_PUNCTUATION)
    suffix = token[len(stripped) :]
    return stripped, suffix


def _eligible_token(token: str) -> bool:
    clean, _ = _clean_token(token)
    return len(clean) >= 4 and any(char.isalpha() for char in clean) and not clean.isdigit() and not URL_EMAIL_PATTERN.search(clean)


def _candidate_positions(clean: str) -> list[int]:
    positions = []
    for index, char in enumerate(clean[1:], start=1):
        if char.isalpha() and char.lower() in KEYBOARD_ADJACENCY:
            positions.append(index)
    return positions


def _collision(clean_word: str) -> bool:
    normalized = clean_word.lower().strip(TRAILING_PUNCTUATION)
    return normalized in word_set


def _replace_char(word: str, index: int, replacement: str) -> str:
    char = word[index]
    rendered = replacement.upper() if char.isupper() else replacement
    return f"{word[:index]}{rendered}{word[index + 1:]}"


def _mutate_token(token: str, rng: random.Random) -> tuple[str | None, tuple[str, str, int, str] | None]:
    clean, suffix = _clean_token(token)
    positions = _candidate_positions(clean)[:3]

    for position in positions:
        choices = list(KEYBOARD_ADJACENCY[clean[position].lower()])
        rng.shuffle(choices)
        for replacement in choices[:3]:
            candidate_clean = _replace_char(clean, position, replacement)
            if _collision(candidate_clean):
                continue
            return f"{candidate_clean}{suffix}", (token, f"{candidate_clean}{suffix}", position, replacement)
    return None, None


def _transform_text(text: str, seed: int) -> tuple[str | None, dict | None, str | None]:
    rng = random.Random(seed)
    tokens = text.split()
    if not tokens:
        _track_skip(True)
        return None, None, "empty_input"
    sentence_ranges: list[list[int]] = []
    current: list[int] = []
    for index, token in enumerate(tokens):
        current.append(index)
        if token.endswith((".", "!", "?")):
            sentence_ranges.append(current)
            current = []
    if current:
        sentence_ranges.append(current)

    transformed = list(tokens)
    metadata = []
    for sentence in sentence_ranges:
        candidate_words = [index for index in sentence if _eligible_token(tokens[index])]
        if not candidate_words:
            _track_skip(True)
            return None, None, "no_eligible_word_in_sentence"

        rng.shuffle(candidate_words)
        changed = False
        for word_index in candidate_words[:2]:
            mutated, change = _mutate_token(tokens[word_index], rng)
            if mutated is not None and change is not None:
                transformed[word_index] = mutated
                metadata.append((word_index, *change))
                changed = True
                break

        if not changed:
            _track_skip(True)
            return None, None, "real_word_collision_or_no_valid_keyboard_swap"

    _track_skip(False)
    return " ".join(transformed), {"changes": metadata}, None


class CHRGEN019(BaseMR):
    @property
    def mr_id(self) -> str:
        return "CHR-GEN-019"

    @property
    def subtasks(self) -> list[str]:
        return ["SA", "NLI", "TOPIC"]

    def transform(self, source_input: str | dict, seed: int = 42) -> str | dict | None:
        self.clear_skip_reason()
        if isinstance(source_input, dict):
            premise, premise_meta, premise_reason = _transform_text(source_input.get("premise", ""), seed)
            hypothesis, hypothesis_meta, hypothesis_reason = _transform_text(source_input.get("hypothesis", ""), seed)
            if premise is None:
                self.set_skip_reason(f"premise_{premise_reason or 'transform_skipped'}")
                return None
            if hypothesis is None:
                self.set_skip_reason(f"hypothesis_{hypothesis_reason or 'transform_skipped'}")
                return None
            return {
                "premise": premise,
                "hypothesis": hypothesis,
                "metadata": {"premise": premise_meta, "hypothesis": hypothesis_meta},
            }

        transformed, _, reason = _transform_text(source_input, seed)
        if transformed is None:
            self.set_skip_reason(reason or "transform_skipped")
            return None
        return transformed

    def check_pass(self, source_output: dict, followup_output: dict) -> bool:
        return (
            abs(followup_output["score"] - source_output["score"]) < SCORE_TOLERANCE_NEARFORMAL
            and followup_output["label"] == source_output["label"]
        )

    def automated_checks(self, source_input: str | dict, followup_input: str | dict) -> bool:
        if isinstance(source_input, dict) and isinstance(followup_input, dict):
            return self._validate_text(source_input.get("premise", ""), followup_input.get("premise", "")) and self._validate_text(
                source_input.get("hypothesis", ""), followup_input.get("hypothesis", "")
            )
        return self._validate_text(str(source_input), str(followup_input))

    def _validate_text(self, source_text: str, followup_text: str) -> bool:
        source_tokens = source_text.split()
        followup_tokens = followup_text.split()
        if len(source_tokens) != len(followup_tokens):
            return False

        differences = []
        for source_token, followup_token in zip(source_tokens, followup_tokens):
            if source_token == followup_token:
                continue
            differences.append((source_token, followup_token))

        if not differences:
            return False

        for source_token, followup_token in differences:
            source_clean, _ = _clean_token(source_token)
            followup_clean, _ = _clean_token(followup_token)
            if len(source_clean) != len(followup_clean):
                return False

            changed_positions = [index for index, (a, b) in enumerate(zip(source_clean, followup_clean)) if a != b]
            if len(changed_positions) != 1:
                return False

            position = changed_positions[0]
            if source_clean[position].lower() not in KEYBOARD_ADJACENCY:
                return False
            if followup_clean[position].lower() not in KEYBOARD_ADJACENCY[source_clean[position].lower()]:
                return False
            if _collision(followup_clean):
                return False

        return True
