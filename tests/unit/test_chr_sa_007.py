import pytest

from chrysalis.mrs.sa.chr_sa_007 import CHRSA007


MR = CHRSA007()


def _source(text: str, source_label: int = 1) -> dict:
    return {"text": text, "source_label": source_label}


def test_basic_transformation():
    transformed = MR.transform(_source("This movie is great. It feels fresh."))
    assert transformed == "This movie is great! It feels fresh!"


def test_skip_conditions():
    assert MR.transform(_source("This line is neutral.", 2)) is None
    assert MR.transform(_source("This is great!", 1)) is None
    assert MR.transform(_source("Wait...", 1)) is None


def test_automated_checks_pass():
    inputs = [
        _source("This movie is great.", 1),
        _source("This film is awful.", 0),
        _source("The acting works.", 1),
        _source("The pacing drags.", 0),
        _source("A simple ending lands.", 1),
        _source("The jokes fail badly.", 0),
        _source("The finale surprises viewers.", 1),
        _source("The music annoys me.", 0),
        _source("The cast impresses everyone.", 1),
        _source("The dialogue frustrates fans.", 0),
    ]
    for item in inputs:
        transformed = MR.transform(item)
        assert transformed is not None
        assert MR.automated_checks(item, transformed)


def test_airtight_guarantee():
    pytest.skip("Not applicable to CHR-SA-007.")


def test_real_word_collision_prevention():
    pytest.skip("Not applicable to CHR-SA-007.")


def test_edge_cases():
    assert MR.transform(_source("", 1)) is None
    assert MR.transform(_source("Word", 1)) is None
    assert MR.transform(_source("Already excited!", 1)) is None
    assert MR.transform(_source("What now?!", 1)) is None
    long_text = " ".join(["This movie works."] * 30)
    assert MR.transform(_source(long_text, 1)) is not None


def test_seeded_reproducibility():
    source = _source("This movie is great.", 1)
    assert MR.transform(source, seed=42) == MR.transform(source, seed=42)
