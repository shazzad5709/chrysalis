import pytest

from chrysalis.mrs.generic.chr_gen_018 import CHRGEN018


class FakeTokenizer:
    def __init__(self, uncased: bool = False) -> None:
        self.uncased = uncased

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        basis = text.lower() if self.uncased else text
        return [ord(char) for char in basis]


MR = CHRGEN018()


def test_basic_transformation():
    assert MR.transform("Hello World!", variant="uppercase") == "HELLO WORLD!"
    assert MR.transform("Hello World!", variant="lowercase") == "hello world!"


def test_skip_conditions():
    assert MR.transform("HELLO", variant="uppercase") is None
    assert MR.transform("hello", variant="lowercase") is None


def test_automated_checks_pass():
    inputs = [
        "Hello World!",
        "Mixed Case Text",
        "Some Review Sentence.",
        "Another Example",
        "Punctuation, Stays!",
        "Movie Review 123",
        "Caps and lower",
        "UP and down",
        "A longer sentence appears here.",
        "Final Example.",
    ]
    for text in inputs:
        for variant in ("uppercase", "lowercase"):
            transformed = MR.transform(text, variant=variant)
            if transformed is None:
                continue
            assert MR.automated_checks(text, transformed)


def test_airtight_guarantee():
    for idx in range(100):
        text = f"Case Example {idx}!"
        transformed = MR.transform(text, variant="uppercase")
        if transformed is None:
            continue
        assert len(transformed) == len(text)
        for a, b in zip(text, transformed):
            if not a.isalpha():
                assert a == b


def test_real_word_collision_prevention():
    pytest.skip("Not applicable to CHR-GEN-018.")


def test_edge_cases():
    assert MR.transform("", variant="uppercase") is None
    assert MR.transform("Word", variant="uppercase") == "WORD"
    assert MR.transform("WORD", variant="uppercase") is None
    assert MR.transform("Odd?!", variant="lowercase") == "odd?!"
    long_text = " ".join(["Sentence"] * 120)
    assert MR.transform(long_text, variant="uppercase") is not None


def test_seeded_reproducibility():
    text = "Hello World!"
    assert MR.transform(text, seed=42, variant="uppercase") == MR.transform(text, seed=42, variant="uppercase")


def test_tokenizer_check_function():
    assert MR.check_tokenizer_casing(FakeTokenizer(uncased=False)) is True
    assert MR.check_tokenizer_casing(FakeTokenizer(uncased=True)) is False
