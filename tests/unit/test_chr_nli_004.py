import pytest

from chrysalis.mrs.nli.chr_nli_004 import CHRNLI004, gender_swap


MR = CHRNLI004()


def _source(premise: str, hypothesis: str) -> dict:
    return {"premise": premise, "hypothesis": hypothesis}


def test_basic_transformation():
    source_input = _source("The man is smiling.", "The man is happy.")
    transformed = MR.transform(source_input, seed=42)

    assert transformed is not None
    assert transformed != source_input
    assert transformed["premise"] == "The gentleman is smiling."
    assert transformed["hypothesis"] == "The gentleman is happy."


def test_skip_conditions():
    assert MR.transform(_source("The man smiled.", "The woman laughed.")) is None
    assert MR.transform(_source("A person smiled.", "Someone laughed.")) is None
    assert MR.transform(_source("He smiled.", "The man laughed.")) is None


def test_automated_checks_pass():
    inputs = [
        _source("The man is kind.", "The man is calm."),
        _source("The woman is brave.", "The woman is ready."),
        _source("The father is cooking.", "The father is serving dinner."),
        _source("The mother is singing.", "The mother is smiling."),
        _source("The boy is running.", "The boy is laughing."),
        _source("The girl is dancing.", "The girl is grinning."),
        _source("The actor is waving.", "The actor is bowing."),
        _source("The actress is speaking.", "The actress is performing."),
        _source("The waiter is walking.", "The waiter is serving food."),
        _source("The hostess is greeting guests.", "The hostess is smiling."),
    ]

    for item in inputs:
        transformed = MR.transform(item, seed=42)
        assert transformed is not None
        assert MR.automated_checks(item, transformed)


def test_airtight_guarantee():
    pytest.skip("Not applicable to CHR-NLI-004.")


def test_real_word_collision_prevention():
    pytest.skip("Not applicable to CHR-NLI-004.")


def test_edge_cases():
    assert MR.transform(_source("", "The man laughed.")) is None
    single_word = MR.transform(_source("Man", "Man"))
    assert single_word is not None
    already_transformed = _source("The gentleman smiled.", "The gentleman laughed.")
    assert MR.transform(already_transformed) is None
    punctuation_case = _source("The mother, oddly, smiled!", "The mother, indeed, laughed.")
    transformed = MR.transform(punctuation_case)
    assert transformed is not None
    assert MR.automated_checks(punctuation_case, transformed)

    long_sentence = " ".join(["The man is calm."] * 30)
    long_input = _source(long_sentence, long_sentence)
    transformed_long = MR.transform(long_input)
    assert transformed_long is not None
    assert MR.automated_checks(long_input, transformed_long)


def test_seeded_reproducibility():
    source_input = _source("The woman is smiling.", "The woman is waving.")
    first = gender_swap(source_input["premise"], source_input["hypothesis"], "same_gender", seed=42)
    second = gender_swap(source_input["premise"], source_input["hypothesis"], "same_gender", seed=42)

    assert first == second
