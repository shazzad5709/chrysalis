import pytest

from chrysalis.mrs.generic.chr_gen_005 import CHRGEN005


MR = CHRGEN005()


def test_basic_transformation():
    transformed = MR.transform("keyboard typing errors", seed=42)
    assert transformed is not None
    assert transformed != "keyboard typing errors"
    assert transformed.replace(" ", "") == "keyboard typing errors".replace(" ", "")


def test_skip_conditions():
    assert MR.transform("cat dog", seed=42) is None


def test_automated_checks_pass():
    inputs = [
        "keyboard typing errors",
        "another ordinary sentence",
        "spacing works nicely here",
        "review text stays stable",
        "people enjoyed the performance",
        "this example contains punctuation.",
        "careful validation matters",
        "simple tests improve reliability",
        "movies sometimes surprise audiences",
        "critics praised the soundtrack",
    ]
    for text in inputs:
        transformed = MR.transform(text, seed=42)
        assert transformed is not None
        assert MR.automated_checks(text, transformed)


def test_airtight_guarantee():
    for idx in range(100):
        text = f"example token number {idx} remains stable"
        transformed = MR.transform(text, seed=42)
        assert transformed is not None
        assert transformed.replace(" ", "") == text.replace(" ", "")
        assert MR.verify_airtight(text, transformed)


def test_real_word_collision_prevention():
    pytest.skip("Not applicable to CHR-GEN-005.")


def test_edge_cases():
    assert MR.transform("", seed=42) is None
    assert MR.transform("word", seed=42) is not None
    assert MR.transform("wo rd", seed=42) is None
    assert MR.transform("hello!!!", seed=42) is not None
    long_text = " ".join(["sentence"] * 120)
    assert MR.transform(long_text, seed=42) is not None


def test_seeded_reproducibility():
    text = "keyboard typing errors"
    assert MR.transform(text, seed=42) == MR.transform(text, seed=42)
