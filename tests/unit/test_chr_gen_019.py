import pytest

from chrysalis.mrs.generic.chr_gen_019 import CHRGEN019, _clean_token, word_set


MR = CHRGEN019()


def test_basic_transformation():
    transformed = MR.transform("keyboard typing sentence", seed=42)
    assert transformed is not None
    assert transformed != "keyboard typing sentence"


def test_skip_conditions():
    assert MR.transform("cat dog", seed=42) is None
    assert MR.transform("1234 5678", seed=42) is None


def test_automated_checks_pass():
    inputs = [
        "keyboard typing sentence",
        "another careful example",
        "reviewers disliked pacing",
        "audiences praised visuals",
        "performances felt natural",
        "dialogue sounded awkward",
        "soundtrack elevated scenes",
        "laughter filled theaters",
        "plotting became predictable",
        "direction stayed focused",
    ]
    for text in inputs:
        transformed = MR.transform(text, seed=42)
        assert transformed is not None
        assert MR.automated_checks(text, transformed)


def test_airtight_guarantee():
    pytest.skip("Not applicable to CHR-GEN-019.")


def test_real_word_collision_prevention():
    for idx in range(200):
        text = f"keyboard typing sentence number {idx}"
        transformed = MR.transform(text, seed=42 + idx)
        assert transformed is not None
        for token in transformed.split():
            clean, _ = _clean_token(token)
            if clean.lower() in word_set:
                original_tokens = text.split()
                if token not in original_tokens:
                    pytest.fail(f"Real-word collision found: {clean}")


def test_edge_cases():
    assert MR.transform("", seed=42) is None
    assert MR.transform("word", seed=42) is not None
    assert MR.transform("okay", seed=42) is not None
    assert MR.transform("weird?! punctuation", seed=42) is not None
    long_text = " ".join(["keyboard"] * 120)
    assert MR.transform(long_text, seed=42) is not None


def test_seeded_reproducibility():
    text = "keyboard typing sentence"
    assert MR.transform(text, seed=42) == MR.transform(text, seed=42)
