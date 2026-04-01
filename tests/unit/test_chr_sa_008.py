import pytest

from chrysalis.config import INTENSIFIER_POOL
from chrysalis.mrs.sa.chr_sa_008 import CHRSA008


MR = CHRSA008()


def _source(text: str, source_label: int = 1) -> dict:
    return {"text": text, "source_label": source_label}


def test_basic_transformation():
    transformed = MR.transform(_source("The movie was great.", 1), seed=42)
    assert transformed is not None
    assert transformed != "The movie was great."
    assert any(word in transformed.lower().split() for word in INTENSIFIER_POOL)


def test_skip_conditions():
    assert MR.transform(_source("The movie was very good.", 1)) is None
    assert MR.transform(_source("The best ending arrived.", 1)) is None
    assert MR.transform(_source("Not good at all.", 0)) is None


def test_automated_checks_pass():
    inputs = [
        _source("The movie was great.", 1),
        _source("The ending felt awful.", 0),
        _source("A charming and bright story.", 1),
        _source("The pacing was rough.", 0),
        _source("The villain seemed cruel.", 0),
        _source("The cast looked joyful.", 1),
        _source("The dialogue felt stiff.", 0),
        _source("The visuals were lovely.", 1),
        _source("The script was clumsy.", 0),
        _source("The soundtrack sounded beautiful.", 1),
    ]
    for item in inputs:
        transformed = MR.transform(item, seed=42)
        assert transformed is not None
        assert MR.automated_checks(item, transformed)


def test_airtight_guarantee():
    pytest.skip("Not applicable to CHR-SA-008.")


def test_real_word_collision_prevention():
    pytest.skip("Not applicable to CHR-SA-008.")


def test_edge_cases():
    assert MR.transform(_source("", 1)) is None
    assert MR.transform(_source("Great", 1)) is not None
    assert MR.transform(_source("The movie was very good.", 1)) is None
    assert MR.transform(_source("Well-designed, sharp, and bright.", 1), seed=42) is not None
    long_text = " ".join(["The movie was great."] * 30)
    assert MR.transform(_source(long_text, 1), seed=42) is not None


def test_seeded_reproducibility():
    source = _source("The movie was great.", 1)
    assert MR.transform(source, seed=42) == MR.transform(source, seed=42)
