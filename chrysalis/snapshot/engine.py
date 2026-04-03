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
        self._source_prediction_cache: dict[tuple[str, str], dict[str, float | int]] = {}

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
            snapshot_path = output_path / corpus_name.replace("_corpus.csv", "_snapshot.csv")
            if snapshot_path.exists():
                continue
            corpus_file = corpus_path / corpus_name
            records = self._read_corpus_records(corpus_file)
            if not records:
                self._write_snapshot(snapshot_path, [])
                continue

            mr_id = records[0].mr_id
            mr = self._get_mr_instance(mr_id)
            snapshots: list[SnapshotRecord] = []
            source_predictions, followup_predictions = self._predict_record_sets(model, tokenizer, records)
            for record, source_pred, followup_pred in zip(records, source_predictions, followup_predictions, strict=False):

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

    def _predict_record_sets(self, model, tokenizer, records: list[CorpusRecord]) -> tuple[list[dict[str, float | int]], list[dict[str, float | int]]]:
        source_predictions: list[dict[str, float | int] | None] = [None] * len(records)
        followup_predictions: list[dict[str, float | int] | None] = [None] * len(records)

        for subtask in sorted({record.subtask for record in records}):
            indices = [index for index, record in enumerate(records) if record.subtask == subtask]
            if not indices:
                continue
            source_payloads = [self._record_payload(records[index].source_text, subtask) for index in indices]
            followup_payloads = [SnapshotEngine._record_payload(records[index].followup_text, subtask) for index in indices]
            uncached_source_payloads = []
            uncached_source_indices = []
            for index, payload in zip(indices, source_payloads, strict=False):
                cache_key = (subtask, records[index].source_text)
                cached = self._source_prediction_cache.get(cache_key)
                if cached is not None:
                    source_predictions[index] = cached
                    continue
                uncached_source_indices.append(index)
                uncached_source_payloads.append(payload)

            if uncached_source_payloads:
                source_batch = self._predict_many(model, tokenizer, subtask, uncached_source_payloads)
                for index, prediction in zip(uncached_source_indices, source_batch, strict=False):
                    source_predictions[index] = prediction
                    cache_key = (subtask, records[index].source_text)
                    self._source_prediction_cache[cache_key] = prediction

            followup_batch = self._predict_many(model, tokenizer, subtask, followup_payloads)
            for index, prediction in zip(indices, followup_batch, strict=False):
                followup_predictions[index] = prediction

        return source_predictions, followup_predictions

    @staticmethod
    def _predict_many(model, tokenizer, subtask: str, payloads) -> list[dict[str, float | int]]:
        if hasattr(model, "predict_many"):
            try:
                results = model.predict_many(payloads, tokenizer=tokenizer, subtask=subtask)
            except TypeError:
                try:
                    results = model.predict_many(payloads, subtask=subtask)
                except TypeError:
                    results = model.predict_many(payloads)
            return [
                {"label": int(result["label"]), "score": float(result["score"])}
                if isinstance(result, dict)
                else {"label": int(result[0]), "score": float(result[1])}
                for result in results
            ]
        return [SnapshotEngine._predict(model, tokenizer, subtask, payload) for payload in payloads]

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
