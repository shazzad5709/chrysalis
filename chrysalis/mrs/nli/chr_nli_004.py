from __future__ import annotations

import random
import re
from typing import Any

from chrysalis.mrs.base import BaseMR

GENDERED_LEXICON = {
    "male_nouns": [
        "man",
        "men",
        "boy",
        "boys",
        "gentleman",
        "gentlemen",
        "guy",
        "guys",
        "lad",
        "lads",
        "male",
        "males",
        "father",
        "dad",
        "son",
        "brother",
        "husband",
        "uncle",
        "grandfather",
        "nephew",
        "king",
        "prince",
        "actor",
        "waiter",
        "host",
    ],
    "female_nouns": [
        "woman",
        "women",
        "girl",
        "girls",
        "lady",
        "ladies",
        "female",
        "females",
        "mother",
        "mom",
        "daughter",
        "sister",
        "wife",
        "aunt",
        "grandmother",
        "niece",
        "queen",
        "princess",
        "actress",
        "waitress",
        "hostess",
    ],
    "male_pronouns": ["he", "him", "his", "himself"],
    "female_pronouns": ["she", "her", "hers", "herself"],
    "neutral_pronouns": ["they", "them", "their", "themselves"],
}

SAME_GENDER_MAP = {
    "man": ["gentleman", "male"],
    "men": ["gentlemen"],
    "boy": ["lad"],
    "boys": ["lads"],
    "father": ["dad"],
    "dad": ["father"],
    "actor": ["performer"],
    "waiter": ["server"],
    "host": ["presenter"],
    "woman": ["lady", "female"],
    "women": ["ladies"],
    "girl": ["lass"],
    "girls": ["lasses"],
    "mother": ["mom"],
    "mom": ["mother"],
    "actress": ["performer"],
    "waitress": ["server"],
    "hostess": ["presenter"],
}

CROSS_GENDER_MAP = {
    "man": "woman",
    "men": "women",
    "boy": "girl",
    "boys": "girls",
    "gentleman": "lady",
    "gentlemen": "ladies",
    "guy": "woman",
    "lad": "lass",
    "male": "female",
    "males": "females",
    "father": "mother",
    "dad": "mom",
    "son": "daughter",
    "brother": "sister",
    "husband": "wife",
    "uncle": "aunt",
    "grandfather": "grandmother",
    "nephew": "niece",
    "king": "queen",
    "prince": "princess",
    "actor": "actress",
    "waiter": "waitress",
    "host": "hostess",
    "he": "she",
    "him": "her",
    "his": "her",
    "himself": "herself",
    "woman": "man",
    "women": "men",
    "girl": "boy",
    "girls": "boys",
    "lady": "gentleman",
    "ladies": "gentlemen",
    "female": "male",
    "females": "males",
    "mother": "father",
    "mom": "dad",
    "daughter": "son",
    "sister": "brother",
    "wife": "husband",
    "aunt": "uncle",
    "grandmother": "grandfather",
    "niece": "nephew",
    "queen": "king",
    "princess": "prince",
    "actress": "actor",
    "waitress": "waiter",
    "hostess": "host",
    "she": "he",
    "her": "him",
    "hers": "his",
    "herself": "himself",
}

TOKEN_PATTERN = re.compile(r"\s+|[A-Za-z]+|[^A-Za-z\s]")
WORD_PATTERN = re.compile(r"[A-Za-z]+")
GENDER_RESTRICTIVE_PATTERNS = (
    "only men",
    "only women",
    "must be male",
    "must be female",
    "men only",
    "women only",
)
BIOLOGICAL_SEX_PATTERNS = (
    "pregnant",
    "pregnancy",
    "breastfeeding",
    "breast feeding",
    "breast-feed",
    "breast fed",
)
MALE_WORDS = set(GENDERED_LEXICON["male_nouns"]) | set(GENDERED_LEXICON["male_pronouns"])
FEMALE_WORDS = set(GENDERED_LEXICON["female_nouns"]) | set(GENDERED_LEXICON["female_pronouns"])
POSSESSIVE_OR_REFLEXIVE = {"his", "her", "hers", "himself", "herself"}


def _tokenize_with_whitespace(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text)


def _word_positions(tokens: list[str]) -> list[int]:
    return [index for index, token in enumerate(tokens) if WORD_PATTERN.fullmatch(token)]


def _word_tokens(text: str) -> list[str]:
    return [token for token in _tokenize_with_whitespace(text) if WORD_PATTERN.fullmatch(token)]


def _next_word_token(tokens: list[str], start_index: int) -> str | None:
    for index in range(start_index + 1, len(tokens)):
        if WORD_PATTERN.fullmatch(tokens[index]):
            return tokens[index]
    return None


def _gender_category(token: str) -> str | None:
    lowered = token.lower()
    if lowered in MALE_WORDS:
        return "male"
    if lowered in FEMALE_WORDS:
        return "female"
    return None


def _preserve_case(original: str, replacement: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original.islower():
        return replacement.lower()
    if original.istitle():
        return replacement.title()

    chars: list[str] = []
    for index, char in enumerate(replacement):
        if index < len(original) and original[index].isupper():
            chars.append(char.upper())
        else:
            chars.append(char.lower())
    return "".join(chars)


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _has_gender_restrictive_language(premise: str, hypothesis: str) -> bool:
    combined = _normalize_space(f"{premise} {hypothesis}")
    return any(pattern in combined for pattern in GENDER_RESTRICTIVE_PATTERNS)


def _has_biological_sex_content(premise: str, hypothesis: str) -> bool:
    combined = _normalize_space(f"{premise} {hypothesis}")
    return any(pattern in combined for pattern in BIOLOGICAL_SEX_PATTERNS)


def _analyze_component(text: str) -> dict[str, Any]:
    tokens = _tokenize_with_whitespace(text)
    positions = _word_positions(tokens)
    gendered: list[dict[str, Any]] = []
    for word_index, token_index in enumerate(positions):
        token = tokens[token_index]
        gender = _gender_category(token)
        if gender is None:
            continue
        gendered.append(
            {
                "token_index": token_index,
                "word_index": word_index,
                "token": token,
                "normalized": token.lower(),
                "gender": gender,
            }
        )
    return {"text": text, "tokens": tokens, "word_positions": positions, "gendered": gendered}


def _pronoun_agreement_ok(text: str) -> bool:
    words = _word_tokens(text)
    lowered = [word.lower() for word in words]

    for index, word in enumerate(lowered):
        if word not in POSSESSIVE_OR_REFLEXIVE:
            continue

        pronoun_gender = _gender_category(word)
        if pronoun_gender is None:
            continue

        next_word_is_noun = index + 1 < len(lowered) and WORD_PATTERN.fullmatch(words[index + 1]) is not None
        if word == "her" and not next_word_is_noun:
            continue

        for lookback in range(max(0, index - 2), index):
            prior_gender = _gender_category(lowered[lookback])
            if prior_gender is None:
                continue
            if prior_gender != pronoun_gender:
                return False
            break

    return True


def _apply_component_substitutions(
    component: dict[str, Any],
    mode: str,
    rng: random.Random,
) -> tuple[str, list[tuple[str, str]], bool]:
    tokens = list(component["tokens"])
    changed_pairs: list[tuple[str, str]] = []
    unmappable = False

    for item in component["gendered"]:
        token = item["token"]
        normalized = item["normalized"]

        if mode == "same_gender":
            replacements = SAME_GENDER_MAP.get(normalized)
            if not replacements:
                continue
            replacement = rng.choice(replacements)
        else:
            replacement = _cross_gender_replacement(component["tokens"], item["token_index"], normalized)
            if replacement is None:
                unmappable = True
                continue

        rendered = _preserve_case(token, replacement)
        tokens[item["token_index"]] = rendered
        if rendered != token:
            changed_pairs.append((normalized, rendered.lower()))

    return "".join(tokens), changed_pairs, unmappable


def _cross_gender_replacement(tokens: list[str], token_index: int, normalized: str) -> str | None:
    if normalized != "her":
        return CROSS_GENDER_MAP.get(normalized)

    next_word = _next_word_token(tokens, token_index)
    if next_word is not None:
        return "his"
    return "him"


def _validate_same_gender_component(source_text: str, followup_text: str) -> bool:
    if source_text == followup_text:
        return False

    source_words = _word_tokens(source_text)
    followup_words = _word_tokens(followup_text)
    if len(source_words) != len(followup_words):
        return False

    changed = False
    for source_word, followup_word in zip(source_words, followup_words):
        source_lower = source_word.lower()
        followup_lower = followup_word.lower()
        if source_lower == followup_lower:
            continue
        expected = SAME_GENDER_MAP.get(source_lower)
        if not expected or followup_lower not in expected:
            return False
        changed = True

    return changed


def _validate_cross_gender_component(source_text: str, followup_text: str) -> bool:
    source_tokens = _tokenize_with_whitespace(source_text)
    followup_tokens = _tokenize_with_whitespace(followup_text)
    source_words = [token for token in source_tokens if WORD_PATTERN.fullmatch(token)]
    followup_words = [token for token in followup_tokens if WORD_PATTERN.fullmatch(token)]
    if len(source_words) != len(followup_words):
        return False

    changed = False
    source_word_indices = _word_positions(source_tokens)
    for word_index, (source_word, followup_word) in enumerate(zip(source_words, followup_words)):
        source_lower = source_word.lower()
        followup_lower = followup_word.lower()
        if source_lower in MALE_WORDS or source_lower in FEMALE_WORDS:
            token_index = source_word_indices[word_index]
            expected = _cross_gender_replacement(source_tokens, token_index, source_lower)
            if expected is None or followup_lower != expected:
                return False
            changed = True
        elif source_lower != followup_lower:
            return False

    return changed


def gender_swap(
    premise: str,
    hypothesis: str,
    mode: str,
    seed: int = 42,
) -> dict | None:
    transformed, _ = gender_swap_with_reason(premise, hypothesis, mode, seed)
    return transformed


def gender_swap_with_reason(
    premise: str,
    hypothesis: str,
    mode: str,
    seed: int = 42,
) -> tuple[dict | None, str | None]:
    if mode not in {"same_gender", "cross_gender"}:
        raise ValueError(f"Unsupported gender swap mode: {mode}")

    if not premise.strip() or not hypothesis.strip():
        return None, "missing_premise_or_hypothesis"

    if mode == "cross_gender" and _has_gender_restrictive_language(premise, hypothesis):
        return None, "gender_restrictive_language"
    if mode == "cross_gender" and _has_biological_sex_content(premise, hypothesis):
        return None, "biological_sex_content"

    premise_component = _analyze_component(premise)
    hypothesis_component = _analyze_component(hypothesis)
    all_gendered = premise_component["gendered"] + hypothesis_component["gendered"]

    male_count = sum(1 for item in all_gendered if item["gender"] == "male")
    female_count = sum(1 for item in all_gendered if item["gender"] == "female")
    if male_count > 0 and female_count > 0:
        return None, "mixed_gender_input"
    if male_count == 0 and female_count == 0:
        return None, "no_gendered_words"

    rng = random.Random(seed)
    transformed_premise, premise_changes, premise_unmappable = _apply_component_substitutions(
        premise_component,
        mode,
        rng,
    )
    transformed_hypothesis, hypothesis_changes, hypothesis_unmappable = _apply_component_substitutions(
        hypothesis_component,
        mode,
        rng,
    )

    if mode == "same_gender":
        if not premise_changes or not hypothesis_changes:
            return None, "partial_substitution"
    else:
        if premise_unmappable or hypothesis_unmappable:
            return None, "unmappable_gendered_token"
        if not premise_component["gendered"] or not hypothesis_component["gendered"]:
            return None, "partial_substitution"
        if len(premise_changes) != len(premise_component["gendered"]):
            return None, "partial_substitution"
        if len(hypothesis_changes) != len(hypothesis_component["gendered"]):
            return None, "partial_substitution"

    if transformed_premise == premise and transformed_hypothesis == hypothesis:
        return None, "no_effective_change"

    if not _pronoun_agreement_ok(transformed_premise) or not _pronoun_agreement_ok(transformed_hypothesis):
        return None, "pronoun_agreement_violation"

    if mode == "same_gender":
        if not _validate_same_gender_component(premise, transformed_premise):
            return None, "same_gender_validation_failed"
        if not _validate_same_gender_component(hypothesis, transformed_hypothesis):
            return None, "same_gender_validation_failed"
    else:
        if not _validate_cross_gender_component(premise, transformed_premise):
            return None, "cross_gender_validation_failed"
        if not _validate_cross_gender_component(hypothesis, transformed_hypothesis):
            return None, "cross_gender_validation_failed"

    return {"premise": transformed_premise, "hypothesis": transformed_hypothesis}, None


class CHRNLI004(BaseMR):
    @property
    def mr_id(self) -> str:
        return "CHR-NLI-004"

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
            mode="same_gender",
            seed=seed,
        )
        if transformed is None and reason is not None:
            self.set_skip_reason(reason)
        return transformed

    def check_pass(self, source_output: dict, followup_output: dict) -> bool:
        return source_output["label"] == followup_output["label"]

    def automated_checks(self, source_input: str | dict, followup_input: str | dict) -> bool:
        if not isinstance(source_input, dict) or not isinstance(followup_input, dict):
            return False

        source_premise = source_input.get("premise", "")
        source_hypothesis = source_input.get("hypothesis", "")
        followup_premise = followup_input.get("premise", "")
        followup_hypothesis = followup_input.get("hypothesis", "")

        if source_premise == followup_premise and source_hypothesis == followup_hypothesis:
            return False
        if not _validate_same_gender_component(source_premise, followup_premise):
            return False
        if not _validate_same_gender_component(source_hypothesis, followup_hypothesis):
            return False
        return _pronoun_agreement_ok(followup_premise) and _pronoun_agreement_ok(followup_hypothesis)
