import pytest

from chrysalis.mrs.nli.chr_nli_004 import gender_swap
from chrysalis.mrs.nli.chr_nli_005 import CHRNLI005


MR = CHRNLI005()


def _source(premise: str, hypothesis: str) -> dict:
    return {"premise": premise, "hypothesis": hypothesis}


def test_basic_transformation():
    source_input = _source("The man is smiling.", "The man is happy.")
    transformed = MR.transform(source_input, seed=42)

    assert transformed is not None
    assert transformed != source_input
    assert transformed["premise"] == "The woman is smiling."
    assert transformed["hypothesis"] == "The woman is happy."

    passed, fairness_regression = MR.check_pass({"label": 0, "score": 0.9}, {"label": 0, "score": 0.8})
    assert passed is True
    assert fairness_regression is True


def test_skip_conditions():
    assert MR.transform(_source("The man smiled.", "The woman laughed.")) is None
    assert MR.transform(_source("A person smiled.", "Someone laughed.")) is None
    assert MR.transform(_source("Only men may enter.", "The man waited.")) is None
    assert MR.transform(_source("The guys arrived.", "The man waved.")) is None


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
    pytest.skip("Not applicable to CHR-NLI-005.")


def test_real_word_collision_prevention():
    pytest.skip("Not applicable to CHR-NLI-005.")


def test_edge_cases():
    assert MR.transform(_source("", "The man laughed.")) is None
    single_word = MR.transform(_source("Man", "Man"))
    assert single_word is not None
    assert MR.transform(_source("The guys smiled.", "The guys laughed.")) is None

    punctuation_case = _source("The mother, oddly, smiled!", "The mother, indeed, laughed.")
    transformed = MR.transform(punctuation_case)
    assert transformed is not None
    assert MR.automated_checks(punctuation_case, transformed)

    long_sentence = " ".join(["The woman is calm."] * 30)
    long_input = _source(long_sentence, long_sentence)
    transformed_long = MR.transform(long_input)
    assert transformed_long is not None
    assert MR.automated_checks(long_input, transformed_long)


def test_seeded_reproducibility():
    source_input = _source("The man is smiling.", "The man is waving.")
    first = gender_swap(source_input["premise"], source_input["hypothesis"], "cross_gender", seed=42)
    second = gender_swap(source_input["premise"], source_input["hypothesis"], "cross_gender", seed=42)

    assert first == second
