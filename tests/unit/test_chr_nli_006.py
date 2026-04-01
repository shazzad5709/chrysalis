from __future__ import annotations

from dataclasses import dataclass

import pytest

import chrysalis.mrs.nli.chr_nli_006 as mr_module
from chrysalis.mrs.nli.chr_nli_006 import CHRNLI006


@dataclass
class FakeToken:
    i: int
    text: str
    whitespace_: str
    dep_: str
    tag_: str
    lemma_: str
    head_i: int
    _doc: "FakeDoc" | None = None

    @property
    def children(self) -> list["FakeToken"]:
        assert self._doc is not None
        return [token for token in self._doc.tokens if token.head_i == self.i and token.i != self.i]


class FakeDoc:
    def __init__(self, specs: list[tuple[str, str, str, str, str, int]]) -> None:
        self.tokens = [
            FakeToken(
                i=index,
                text=text,
                whitespace_=whitespace,
                dep_=dep,
                tag_=tag,
                lemma_=lemma,
                head_i=head_i,
            )
            for index, (text, whitespace, dep, tag, lemma, head_i) in enumerate(specs)
        ]
        for token in self.tokens:
            token._doc = self

    def __iter__(self):
        return iter(self.tokens)


class FakeNLP:
    def __init__(self, docs: dict[str, FakeDoc]) -> None:
        self.docs = docs

    def __call__(self, text: str) -> FakeDoc:
        if text not in self.docs:
            raise KeyError(f"No fake parse registered for: {text}")
        return self.docs[text]


def _make_doc(specs: list[tuple[str, str, str, str, str, int]]) -> FakeDoc:
    return FakeDoc(specs)


def _hypothesis_docs() -> dict[str, FakeDoc]:
    return {
        "A man is playing guitar.": _make_doc(
            [
                ("A", " ", "det", "DT", "a", 1),
                ("man", " ", "nsubj", "NN", "man", 3),
                ("is", " ", "aux", "VBZ", "be", 3),
                ("playing", " ", "ROOT", "VBG", "play", 3),
                ("guitar", "", "dobj", "NN", "guitar", 3),
                (".", "", "punct", ".", ".", 3),
            ]
        ),
        "A man plays guitar.": _make_doc(
            [
                ("A", " ", "det", "DT", "a", 1),
                ("man", " ", "nsubj", "NN", "man", 2),
                ("plays", " ", "ROOT", "VBZ", "play", 2),
                ("guitar", "", "dobj", "NN", "guitar", 2),
                (".", "", "punct", ".", ".", 2),
            ]
        ),
        "A man played guitar.": _make_doc(
            [
                ("A", " ", "det", "DT", "a", 1),
                ("man", " ", "nsubj", "NN", "man", 2),
                ("played", " ", "ROOT", "VBD", "play", 2),
                ("guitar", "", "dobj", "NN", "guitar", 2),
                (".", "", "punct", ".", ".", 2),
            ]
        ),
        "The man is tall.": _make_doc(
            [
                ("The", " ", "det", "DT", "the", 1),
                ("man", " ", "nsubj", "NN", "man", 2),
                ("is", " ", "ROOT", "VBZ", "be", 2),
                ("tall", "", "acomp", "JJ", "tall", 2),
                (".", "", "punct", ".", ".", 2),
            ]
        ),
        "A woman can sing.": _make_doc(
            [
                ("A", " ", "det", "DT", "a", 1),
                ("woman", " ", "nsubj", "NN", "woman", 3),
                ("can", " ", "aux", "MD", "can", 3),
                ("sing", "", "ROOT", "VB", "sing", 3),
                (".", "", "punct", ".", ".", 3),
            ]
        ),
        "A dog barks loudly.": _make_doc(
            [
                ("A", " ", "det", "DT", "a", 1),
                ("dog", " ", "nsubj", "NN", "dog", 2),
                ("barks", " ", "ROOT", "VBZ", "bark", 2),
                ("loudly", "", "advmod", "RB", "loudly", 2),
                (".", "", "punct", ".", ".", 2),
            ]
        ),
        "The team won yesterday.": _make_doc(
            [
                ("The", " ", "det", "DT", "the", 1),
                ("team", " ", "nsubj", "NN", "team", 2),
                ("won", " ", "ROOT", "VBD", "win", 2),
                ("yesterday", "", "npadvmod", "NN", "yesterday", 2),
                (".", "", "punct", ".", ".", 2),
            ]
        ),
        "The child was cheerful.": _make_doc(
            [
                ("The", " ", "det", "DT", "the", 1),
                ("child", " ", "nsubj", "NN", "child", 2),
                ("was", " ", "ROOT", "VBD", "be", 2),
                ("cheerful", "", "acomp", "JJ", "cheerful", 2),
                (".", "", "punct", ".", ".", 2),
            ]
        ),
        "Birds fly south.": _make_doc(
            [
                ("Birds", " ", "nsubj", "NNS", "bird", 1),
                ("fly", " ", "ROOT", "VBP", "fly", 1),
                ("south", "", "advmod", "RB", "south", 1),
                (".", "", "punct", ".", ".", 1),
            ]
        ),
        "The women are laughing loudly.": _make_doc(
            [
                ("The", " ", "det", "DT", "the", 1),
                ("women", " ", "nsubj", "NNS", "woman", 3),
                ("are", " ", "aux", "VBP", "be", 3),
                ("laughing", " ", "ROOT", "VBG", "laugh", 3),
                ("loudly", "", "advmod", "RB", "loudly", 3),
                (".", "", "punct", ".", ".", 3),
            ]
        ),
        "The musician performed tonight.": _make_doc(
            [
                ("The", " ", "det", "DT", "the", 1),
                ("musician", " ", "nsubj", "NN", "musician", 2),
                ("performed", " ", "ROOT", "VBD", "perform", 2),
                ("tonight", "", "npadvmod", "NN", "tonight", 2),
                (".", "", "punct", ".", ".", 2),
            ]
        ),
        "The artist is ready.": _make_doc(
            [
                ("The", " ", "det", "DT", "the", 1),
                ("artist", " ", "nsubj", "NN", "artist", 2),
                ("is", " ", "ROOT", "VBZ", "be", 2),
                ("ready", "", "acomp", "JJ", "ready", 2),
                (".", "", "punct", ".", ".", 2),
            ]
        ),
        "Cars move quickly.": _make_doc(
            [
                ("Cars", " ", "nsubj", "NNS", "car", 1),
                ("move", " ", "ROOT", "VBP", "move", 1),
                ("quickly", "", "advmod", "RB", "quickly", 1),
                (".", "", "punct", ".", ".", 1),
            ]
        ),
        "The athlete has finished training.": _make_doc(
            [
                ("The", " ", "det", "DT", "the", 1),
                ("athlete", " ", "nsubj", "NN", "athlete", 3),
                ("has", " ", "aux", "VBZ", "have", 3),
                ("finished", " ", "ROOT", "VBN", "finish", 3),
                ("training", "", "dobj", "NN", "training", 3),
                (".", "", "punct", ".", ".", 3),
            ]
        ),
        "Hello": _make_doc(
            [
                ("Hello", "", "ROOT", "UH", "hello", 0),
            ]
        ),
        "Really?!": _make_doc(
            [
                ("Really", "", "ROOT", "RB", "really", 0),
                ("?", "", "punct", ".", "?", 0),
                ("!", "", "punct", ".", "!", 0),
            ]
        ),
        " ".join(["A man plays guitar."] * 30): _make_doc(
            [
                ("A", " ", "det", "DT", "a", 1),
                ("man", " ", "nsubj", "NN", "man", 2),
                ("plays", " ", "ROOT", "VBZ", "play", 2),
                ("guitar", "", "dobj", "NN", "guitar", 2),
                (".", " ", "punct", ".", ".", 2),
            ]
        ),
    }


@pytest.fixture()
def mr(monkeypatch: pytest.MonkeyPatch) -> CHRNLI006:
    monkeypatch.setattr(mr_module, "_get_nlp", lambda: FakeNLP(_hypothesis_docs()))
    return CHRNLI006()


def _source(hypothesis: str, source_label: int = 0, premise: str = "A premise stays fixed.") -> dict:
    return {"premise": premise, "hypothesis": hypothesis, "source_label": source_label}


def test_basic_transformation(mr: CHRNLI006):
    case_a = mr.transform(_source("A man is playing guitar."))
    case_b = mr.transform(_source("A man plays guitar."))
    case_b_plural = mr.transform(_source("Birds fly south."))
    case_c = mr.transform(_source("A man played guitar.", source_label=2))
    case_d = mr.transform(_source("The man is tall."))

    assert case_a is not None and case_a["hypothesis"] == "A man is not playing guitar."
    assert case_b is not None and case_b["hypothesis"] == "A man does not play guitar."
    assert case_b_plural is not None and case_b_plural["hypothesis"] == "Birds do not fly south."
    assert case_c is not None and case_c["hypothesis"] == "A man did not play guitar."
    assert case_d is not None and case_d["hypothesis"] == "The man is not tall."


def test_skip_conditions(mr: CHRNLI006):
    assert mr.transform(_source("A man plays guitar.", source_label=1)) is None
    assert mr.transform(_source("A man is not playing guitar.")) is None


def test_automated_checks_pass(mr: CHRNLI006):
    hypotheses = [
        "A man is playing guitar.",
        "A man plays guitar.",
        "A man played guitar.",
        "The man is tall.",
        "A woman can sing.",
        "A dog barks loudly.",
        "The team won yesterday.",
        "The child was cheerful.",
        "Birds fly south.",
        "The women are laughing loudly.",
    ]

    for hypothesis in hypotheses:
        source = _source(hypothesis, source_label=0 if "played" not in hypothesis and "won" not in hypothesis else 2)
        transformed = mr.transform(source)
        assert transformed is not None
        assert mr.automated_checks(source, transformed)


def test_airtight_guarantee():
    pytest.skip("Not applicable to CHR-NLI-006.")


def test_real_word_collision_prevention():
    pytest.skip("Not applicable to CHR-NLI-006.")


def test_edge_cases(mr: CHRNLI006):
    assert mr.transform(_source("", source_label=0)) is None
    assert mr.transform(_source("Hello", source_label=0)) is None
    assert mr.transform(_source("A man does not play guitar.", source_label=0)) is None

    unusual = _source("Really?!", source_label=0)
    assert mr.transform(unusual) is None

    long_text = " ".join(["A man plays guitar."] * 30)
    transformed = mr.transform(_source(long_text, source_label=0))
    assert transformed is not None


def test_seeded_reproducibility(mr: CHRNLI006):
    source = _source("A man plays guitar.", source_label=0)
    first = mr.transform(source, seed=42)
    second = mr.transform(source, seed=42)
    assert first == second
