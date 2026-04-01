from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import importlib
import inspect
import json
from pathlib import Path

from chrysalis.corpus.schemas import CorpusRecord, SnapshotRecord
from chrysalis.mrs.base import BaseMR
from chrysalis.registry.registry import RegistryLoader

SNAPSHOT_FIELDNAMES = [
    "model_version",
    "mr_id",
    "input_id",
    "variant",
    "source_pred_label",
    "source_pred_score",
    "followup_pred_label",
    "followup_pred_score",
    "mr_pass",
    "fairness_regression",
    "timestamp",
]


def _sha256_for_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _deserialize_nli_text(text: str) -> dict[str, str]:
    prefix = "premise: "
    delimiter = " || hypothesis: "
    if text.startswith(prefix) and delimiter in text:
        premise, hypothesis = text[len(prefix) :].split(delimiter, 1)
        return {"premise": premise, "hypothesis": hypothesis}
    return {"premise": "", "hypothesis": text}


class SnapshotEngine:
    def __init__(self, registry_loader: RegistryLoader | None = None) -> None:
        self.registry_loader = registry_loader or RegistryLoader()
        self._mr_cache: dict[str, BaseMR] = {}

    def run(self, model, tokenizer, model_version: str, corpus_dir: str, output_dir: str) -> None:
        corpus_path = Path(corpus_dir)
        output_path = Path(output_dir) / model_version
        output_path.mkdir(parents=True, exist_ok=True)
        self.verify_corpus_hashes(corpus_path)

        manifest_path = corpus_path / "corpus_manifest.json"
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        for corpus_name in sorted(manifest):
            if not corpus_name.endswith("_corpus.csv"):
                continue
            corpus_file = corpus_path / corpus_name
            records = self._read_corpus_records(corpus_file)
            if not records:
                self._write_snapshot(output_path / corpus_name.replace("_corpus.csv", "_snapshot.csv"), [])
                continue

            mr_id = records[0].mr_id
            mr = self._get_mr_instance(mr_id)
            snapshots: list[SnapshotRecord] = []
            for record in records:
                source_payload = self._record_payload(record.source_text, record.subtask)
                followup_payload = self._record_payload(record.followup_text, record.subtask)
                source_pred = self._predict(model, tokenizer, record.subtask, source_payload)
                followup_pred = self._predict(model, tokenizer, record.subtask, followup_payload)

                check_result = mr.check_pass(
                    {"label": source_pred["label"], "score": source_pred["score"]},
                    {"label": followup_pred["label"], "score": followup_pred["score"]},
                )
                if isinstance(check_result, tuple):
                    mr_pass, fairness_flag = check_result
                else:
                    mr_pass = bool(check_result)
                    fairness_flag = False
                if mr_id == "CHR-NLI-005" and not mr_pass:
                    fairness_flag = True

                snapshots.append(
                    SnapshotRecord(
                        model_version=model_version,
                        mr_id=record.mr_id,
                        input_id=record.input_id,
                        variant=record.variant,
                        source_pred_label=int(source_pred["label"]),
                        source_pred_score=float(source_pred["score"]),
                        followup_pred_label=int(followup_pred["label"]),
                        followup_pred_score=float(followup_pred["score"]),
                        mr_pass=mr_pass,
                        fairness_regression=fairness_flag,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )
                )

            snapshot_path = output_path / corpus_name.replace("_corpus.csv", "_snapshot.csv")
            self._write_snapshot(snapshot_path, snapshots)

    def verify_corpus_hashes(self, corpus_dir: str | Path) -> bool:
        corpus_path = Path(corpus_dir)
        manifest_path = corpus_path / "corpus_manifest.json"
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        for filename, expected_hash in manifest.items():
            actual_hash = _sha256_for_file(corpus_path / filename)
            if actual_hash != expected_hash:
                msg = f"Corpus hash mismatch for {filename}"
                raise ValueError(msg)
        return True

    @staticmethod
    def _record_payload(text: str, subtask: str):
        if subtask == "NLI":
            return _deserialize_nli_text(text)
        return text

    @staticmethod
    def _predict(model, tokenizer, subtask: str, payload) -> dict[str, float | int]:
        if hasattr(model, "predict"):
            try:
                result = model.predict(payload, tokenizer=tokenizer, subtask=subtask)
            except TypeError:
                try:
                    result = model.predict(payload, subtask=subtask)
                except TypeError:
                    result = model.predict(payload)
        elif callable(model):
            try:
                result = model(payload, tokenizer=tokenizer, subtask=subtask)
            except TypeError:
                result = model(payload)
        else:
            raise TypeError("Model must provide a callable interface or a predict() method.")

        if isinstance(result, dict):
            return {"label": int(result["label"]), "score": float(result["score"])}
        if isinstance(result, tuple) and len(result) == 2:
            return {"label": int(result[0]), "score": float(result[1])}
        raise TypeError("Model prediction must return {'label': int, 'score': float} or a (label, score) tuple.")

    @staticmethod
    def _read_corpus_records(path: Path) -> list[CorpusRecord]:
        with path.open("r", newline="", encoding="utf-8") as handle:
            return [CorpusRecord.from_csv_row(row) for row in csv.DictReader(handle)]

    @staticmethod
    def _write_snapshot(path: Path, records: list[SnapshotRecord]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=SNAPSHOT_FIELDNAMES)
            writer.writeheader()
            for record in records:
                writer.writerow(record.to_csv_row())

    def _get_mr_instance(self, mr_id: str) -> BaseMR:
        cached = self._mr_cache.get(mr_id)
        if cached is not None:
            return cached

        record = self.registry_loader.get_mr(mr_id)
        module = importlib.import_module(record["implementation_module"])
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BaseMR) and obj is not BaseMR and obj.__module__ == module.__name__:
                instance = obj()
                self._mr_cache[mr_id] = instance
                return instance

        msg = f"No BaseMR implementation found for {mr_id}"
        raise ValueError(msg)
