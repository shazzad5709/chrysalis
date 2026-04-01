import pytest

from chrysalis.mrs.sa.chr_sa_001 import CHRSA001


MR = CHRSA001()


def _source(text: str, source_label: int = 1) -> dict:
    return {"text": text, "source_label": source_label}


def test_basic_transformation():
    assert MR.transform(_source("He is smiling.", 1)) == "He is not smiling."
    assert MR.transform(_source("She plays guitar.", 1)) == "She does not play guitar."
    assert MR.transform(_source("They play outside.", 1)) == "They do not play outside."
    assert MR.transform(_source("He played well.", 1)) == "He did not play well."


def test_skip_conditions_existing_negation():
    assert MR.transform(_source("He is not smiling.", 1)) is None


def test_skip_conditions_neutral_label():
    assert MR.transform(_source("He is smiling.", 2), seed=42) is None


def test_automated_checks_pass():
    inputs = [
        _source("He is smiling.", 1),
        _source("She is cooking dinner.", 1),
        _source("They play outside.", 1),
        _source("He played well.", 1),
        _source("The child is cheerful.", 1),
        _source("Dogs run outside.", 0),
        _source("Birds fly south.", 0),
        _source("The team won easily.", 1),
        _source("The actor is ready.", 1),
        _source("Cars move quickly.", 0),
    ]
    for item in inputs:
        transformed = MR.transform(item)
        assert transformed is not None
        assert MR.automated_checks(item, transformed)


def test_airtight_guarantee():
    pytest.skip("Not applicable to CHR-SA-001.")


def test_real_word_collision_prevention():
    pytest.skip("Not applicable to CHR-SA-001.")


def test_edge_cases():
    assert MR.transform(_source("", 1)) is None
    assert MR.transform(_source("Hello", 1)) is None
    assert MR.transform(_source("He is not happy.", 1)) is None
    unusual = MR.transform(_source("Really?!", 1))
    assert unusual is None
    long_text = " ".join(["They play outside."] * 30)
    assert MR.transform(_source(long_text, 1)) is not None


def test_seeded_reproducibility():
    source = _source("She plays guitar.", 1)
    assert MR.transform(source, seed=42) == MR.transform(source, seed=42)
