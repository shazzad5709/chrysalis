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
    article_case = MR.transform(_source("It is a charming story.", 1), seed=42)
    assert article_case is not None
    assert "an extremely charming story" in article_case.lower()


def test_skip_conditions():
    assert MR.transform(_source("The movie was very good.", 1)) is None
    assert MR.transform(_source("The best ending arrived.", 1)) is None
    assert MR.transform(_source("Not good at all.", 0)) is None
    assert MR.transform(_source("What was once original has been co-opted so frequently.", 0)) is None
    assert MR.transform(_source("Or doing last year's taxes with your ex-wife.", 0)) is None
    assert MR.transform(_source("Seldom has a movie felt this slow.", 0)) is None
    assert MR.transform(_source("This just felt like it lasted forever.", 0)) is None
    assert MR.transform(_source("The cold turkey approach failed.", 0)) is None


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
    hyphenated = MR.transform(_source("The co-opted premise felt awkward.", 0), seed=42)
    assert hyphenated is not None
    assert "co-extremely" not in hyphenated.lower()
    assert "-extremely" not in hyphenated.lower()
    long_text = " ".join(["The movie was great."] * 30)
    assert MR.transform(_source(long_text, 1), seed=42) is not None


def test_seeded_reproducibility():
    source = _source("The movie was great.", 1)
    assert MR.transform(source, seed=42) == MR.transform(source, seed=42)
