from __future__ import annotations

import csv
import json
from pathlib import Path

from chrysalis.corpus.generator import CorpusGenerator
from chrysalis.corpus.validator import CorpusValidator
from chrysalis.snapshot.engine import SnapshotEngine


ALL_MR_IDS = [
    "CHR-SA-001",
    "CHR-SA-007",
    "CHR-SA-008",
    "CHR-SA-010",
    "CHR-NLI-004",
    "CHR-NLI-005",
    "CHR-NLI-006",
    "CHR-GEN-005",
    "CHR-GEN-018",
    "CHR-GEN-019",
]


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(char) for char in text]


class FakeModel:
    POSITIVE_WORDS = {"good", "great", "excellent", "amazing", "love", "wins", "enjoyable"}
    NEGATIVE_WORDS = {"bad", "awful", "terrible", "boring", "dull", "hate"}
    INTENSIFIERS = {"very", "extremely", "incredibly", "remarkably", "exceptionally", "particularly", "truly", "genuinely"}

    def predict(self, payload, tokenizer=None, subtask: str | None = None) -> dict[str, float | int]:
        del tokenizer
        if subtask == "NLI":
            return self._predict_nli(payload)
        return self._predict_sa(str(payload))

    def _predict_sa(self, text: str) -> dict[str, float | int]:
        collapsed = "".join(char.lower() for char in text if char.isalpha() or char.isspace())
        score = 0.5
        if any(word in collapsed for word in self.POSITIVE_WORDS):
            score = 0.8
        if any(word in collapsed for word in self.NEGATIVE_WORDS):
            score = 0.2
        if "not" in collapsed:
            score = min(score, 0.2)

        emphasis = collapsed.count("!") * 0.03 + sum(collapsed.count(word) for word in self.INTENSIFIERS) * 0.04
        if score >= 0.5:
            score = min(0.99, score + emphasis)
        else:
            score = max(0.01, score - emphasis)
        return {"label": 1 if score >= 0.5 else 0, "score": score}

    @staticmethod
    def _normalize_nli_text(text: str) -> str:
        return " ".join(text.lower().split())

    def _predict_nli(self, payload: dict[str, str]) -> dict[str, float | int]:
        premise = self._normalize_nli_text(payload["premise"])
        hypothesis = self._normalize_nli_text(payload["hypothesis"])

        if premise == hypothesis:
            return {"label": 0, "score": 0.9}

        premise_without_not = premise.replace(" not ", " ").replace("n't", "")
        hypothesis_without_not = hypothesis.replace(" not ", " ").replace("n't", "")
        if premise_without_not == hypothesis_without_not and premise != hypothesis:
            return {"label": 2, "score": 0.1}

        return {"label": 1, "score": 0.5}


def _parse_nli_serialized(text: str) -> dict[str, str]:
    prefix = "premise: "
    delimiter = " || hypothesis: "
    premise, hypothesis = text[len(prefix) :].split(delimiter, 1)
    return {"premise": premise, "hypothesis": hypothesis}


def _validator_payload(mr_id: str, subtask: str, text: str, label: int):
    if subtask == "SA":
        if mr_id.startswith("CHR-SA-"):
            return {"text": text, "source_label": label}
        return text
    payload = _parse_nli_serialized(text)
    payload["source_label"] = label
    return payload


def test_corpus_generation_end_to_end(tmp_path):
    sa_source = [
        {"input_id": "sa-0", "text": "The Movie is good.", "label": 1},
        {"input_id": "sa-1", "text": "The Film was excellent.", "label": 1},
        {"input_id": "sa-2", "text": "The Plot was bad.", "label": 0},
        {"input_id": "sa-3", "text": "Actors deliver amazing performances.", "label": 1},
        {"input_id": "sa-4", "text": "The script feels awful.", "label": 0},
        {"input_id": "sa-5", "text": "A well-designed story.", "label": 1},
        {"input_id": "sa-6", "text": "Dr. Adams arrived.", "label": 1},
        {"input_id": "sa-7", "text": "The ending is great.", "label": 1},
        {"input_id": "sa-8", "text": "This is terrible.", "label": 0},
        {"input_id": "sa-9", "text": "USA wins easily.", "label": 1},
    ]
    nli_source = [
        {"input_id": "nli-0", "premise": "The man is singing.", "hypothesis": "The man is singing.", "label": 0},
        {"input_id": "nli-1", "premise": "The woman is tall.", "hypothesis": "The woman is tall.", "label": 0},
        {"input_id": "nli-2", "premise": "The actor played music.", "hypothesis": "The actor played music.", "label": 0},
        {"input_id": "nli-3", "premise": "The father cooks dinner.", "hypothesis": "The father cooks dinner.", "label": 0},
        {"input_id": "nli-4", "premise": "A runner wins races.", "hypothesis": "A runner wins races.", "label": 0},
        {"input_id": "nli-5", "premise": "A chef cooks meals.", "hypothesis": "A chef cooks meals.", "label": 0},
        {"input_id": "nli-6", "premise": "A prince rides horses.", "hypothesis": "A prince rides horses.", "label": 0},
        {"input_id": "nli-7", "premise": "Two men walk outside.", "hypothesis": "Two men walk outside.", "label": 0},
        {"input_id": "nli-8", "premise": "The hostess greets guests.", "hypothesis": "The hostess greets guests.", "label": 0},
        {"input_id": "nli-9", "premise": "The man thanked the woman.", "hypothesis": "The man thanked the woman.", "label": 0},
    ]

    corpus_dir = tmp_path / "corpus"
    manual_dir = tmp_path / "manual_validation"
    snapshot_dir = tmp_path / "snapshots"

    generator = CorpusGenerator(tokenizer=FakeTokenizer(), manual_validation_dir=manual_dir)
    generator.generate(ALL_MR_IDS, sa_source, nli_source, str(corpus_dir), seed=42)

    rejection_log = corpus_dir / "rejection_log.csv"
    assert rejection_log.exists()
    with rejection_log.open("r", newline="", encoding="utf-8") as handle:
        rejection_rows = list(csv.DictReader(handle))
    assert rejection_rows

    validator = CorpusValidator()
    corpus_files = sorted(corpus_dir.glob("*_corpus.csv"))
    assert len(corpus_files) == len(ALL_MR_IDS)
    for corpus_file in corpus_files:
        with corpus_file.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            source_payload = _validator_payload(row["mr_id"], row["subtask"], row["source_text"], int(row["source_label"]))
            followup_payload = _validator_payload(row["mr_id"], row["subtask"], row["followup_text"], int(row["source_label"]))
            valid, reason = validator.validate_pair(row["mr_id"], source_payload, followup_payload)
            assert valid, reason

    expected_manual_files = [
        manual_dir / "manual_validation_artifacts_CHR-SA-001.csv",
        manual_dir / "manual_validation_artifacts_CHR-SA-007.csv",
        manual_dir / "manual_validation_artifacts_CHR-SA-008.csv",
        manual_dir / "manual_validation_artifacts_CHR-SA-010.csv",
        manual_dir / "manual_validation_artifacts_CHR-NLI-004.csv",
        manual_dir / "manual_validation_artifacts_CHR-NLI-005.csv",
        manual_dir / "manual_validation_artifacts_CHR-NLI-006.csv",
        manual_dir / "manual_validation_artifacts_CHR-GEN-005.csv",
        manual_dir / "manual_validation_artifacts_CHR-GEN-018-A.csv",
        manual_dir / "manual_validation_artifacts_CHR-GEN-018-B.csv",
        manual_dir / "manual_validation_artifacts_CHR-GEN-019.csv",
    ]
    for path in expected_manual_files:
        assert path.exists()

    manifest_path = corpus_dir / "corpus_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest
    assert all(len(digest) == 64 for digest in manifest.values())

    engine = SnapshotEngine()
    assert engine.verify_corpus_hashes(corpus_dir) is True
    engine.run(FakeModel(), FakeTokenizer(), "vtest", str(corpus_dir), str(snapshot_dir))

    snapshot_files = sorted((snapshot_dir / "vtest").glob("*_snapshot.csv"))
    assert len(snapshot_files) == len(ALL_MR_IDS)
