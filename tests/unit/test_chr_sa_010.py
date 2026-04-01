import pytest

from chrysalis.mrs.sa.chr_sa_010 import CHRSA010


MR = CHRSA010()


def _source(text: str, source_label: int = 1) -> dict:
    return {"text": text, "source_label": source_label}


def test_basic_transformation():
    transformed = MR.transform(_source("The well-designed movie surprised New York audiences.", 1))
    assert transformed is not None
    assert "WELL-DESIGNED" in transformed
    assert "MOVIE" in transformed
    assert "NEW YORK" in transformed


def test_skip_conditions():
    assert MR.transform(_source("RUN FAST", 1)) is None
    assert MR.transform(_source("", 1)) is None


def test_automated_checks_pass():
    inputs = [
        _source("The great movie won fans.", 1),
        _source("A bad script hurt momentum.", 0),
        _source("New York crowds cheered loudly.", 1),
        _source("The rough pacing annoyed viewers.", 0),
        _source("A lovely soundtrack carried scenes.", 1),
        _source("The clumsy ending ruined tension.", 0),
        _source("Bright visuals thrilled audiences.", 1),
        _source("The noisy dialogue felt awkward.", 0),
        _source("A sharp cast elevated material.", 1),
        _source("The weak villain bored everyone.", 0),
    ]
    for item in inputs:
        transformed = MR.transform(item)
        assert transformed is not None
        assert MR.automated_checks(item, transformed)


def test_airtight_guarantee():
    pytest.skip("Not applicable to CHR-SA-010.")


def test_real_word_collision_prevention():
    pytest.skip("Not applicable to CHR-SA-010.")


def test_edge_cases():
    assert MR.transform(_source("", 1)) is None
    assert MR.transform(_source("Movie", 1)) is not None
    assert MR.transform(_source("USA wins", 1)) is None
    assert MR.transform(_source("Odd punctuation, bright colors!", 1)) is not None
    long_text = " ".join(["The great movie won fans."] * 30)
    assert MR.transform(_source(long_text, 1)) is not None


def test_seeded_reproducibility():
    source = _source("The great movie won fans.", 1)
    assert MR.transform(source, seed=42) == MR.transform(source, seed=42)
