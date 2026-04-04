from __future__ import annotations

import csv
import hashlib
import importlib
import inspect
import json
import logging
from pathlib import Path
import random
from typing import Iterable

from chrysalis.corpus.schemas import CorpusRecord
from chrysalis.corpus.validator import CorpusValidator
from chrysalis.mrs.base import BaseMR
from chrysalis.registry.registry import RegistryLoader

logger = logging.getLogger(__name__)
GENERATION_PROGRESS_EVERY = 1000

CORPUS_FIELDNAMES = [
    "mr_id",
    "input_id",
    "subtask",
    "source_text",
    "source_label",
    "followup_text",
    "expected_output_relation",
    "variant",
    "skip_reason",
]
MANUAL_VALIDATION_FIELDNAMES = [
    "input_id",
    "source_text",
    "followup_text",
    "source_label",
    "expected_output_relation",
    "automated_checks_passed",
    "notes",
]
REJECTION_FIELDNAMES = ["mr_id", "input_id", "subtask", "variant", "reason", "source_text"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _serialize_nli_input(premise: str, hypothesis: str) -> str:
    return f"premise: {premise} || hypothesis: {hypothesis}"


def _normalize_label(value) -> int:
    if value is None:
        return -1
    if isinstance(value, bool):
        return int(value)
    return int(value)


def _sha256_for_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_sa_examples(source: Iterable[dict]) -> list[dict]:
    normalized: list[dict] = []
    for index, raw in enumerate(source):
        text = raw.get("text", raw.get("sentence", ""))
        if not isinstance(text, str):
            continue
        normalized.append(
            {
                "input_id": str(raw.get("input_id", raw.get("id", raw.get("idx", f"sa-{index}")))),
                "text": text,
                "source_label": _normalize_label(raw.get("source_label", raw.get("label"))),
            }
        )
    return normalized


def _normalize_topic_examples(source: Iterable[dict]) -> list[dict]:
    normalized: list[dict] = []
    for index, raw in enumerate(source):
        text = raw.get("text", raw.get("sentence", ""))
        if not isinstance(text, str):
            continue
        source_label = _normalize_label(raw.get("source_label", raw.get("label")))
        normalized.append(
            {
                "input_id": str(raw.get("input_id", raw.get("id", raw.get("idx", f"topic-{index}")))),
                "text": text,
                "source_label": source_label,
            }
        )
    return normalized


def _normalize_nli_examples(source: Iterable[dict]) -> list[dict]:
    normalized: list[dict] = []
    for index, raw in enumerate(source):
        premise = raw.get("premise", "")
        hypothesis = raw.get("hypothesis", "")
        if not isinstance(premise, str) or not isinstance(hypothesis, str):
            continue
        source_label = _normalize_label(raw.get("source_label", raw.get("label", raw.get("gold_label", -1))))
        if source_label not in {0, 1, 2}:
            continue
        normalized.append(
            {
                "input_id": str(raw.get("input_id", raw.get("id", raw.get("idx", f"nli-{index}")))),
                "premise": premise,
                "hypothesis": hypothesis,
                "source_label": source_label,
            }
        )
    return normalized


class CorpusGenerator:
    def __init__(
        self,
        registry_loader: RegistryLoader | None = None,
        validator: CorpusValidator | None = None,
        tokenizer=None,
        manual_validation_dir: str | Path | None = None,
    ) -> None:
        self.registry_loader = registry_loader or RegistryLoader()
        self.validator = validator or CorpusValidator(self.registry_loader)
        self.tokenizer = tokenizer
        self.manual_validation_dir = (
            Path(manual_validation_dir)
            if manual_validation_dir is not None
            else _repo_root() / "pilot" / "artifacts" / "manual_validation"
        )
        self._mr_cache: dict[str, BaseMR] = {}

    def generate(
        self,
        mr_ids: list[str],
        sa_source,
        nli_source,
        output_dir: str,
        seed: int = 42,
        sa_source_overrides: dict[str, object] | None = None,
        nli_source_overrides: dict[str, object] | None = None,
        topic_source=None,
        topic_source_overrides: dict[str, object] | None = None,
    ) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.manual_validation_dir.mkdir(parents=True, exist_ok=True)

        sa_examples = _normalize_sa_examples(list(sa_source))
        nli_examples = _normalize_nli_examples(list(nli_source))
        topic_examples = _normalize_topic_examples(list(topic_source or []))
        normalized_sa_overrides = {
            mr_id: _normalize_sa_examples(list(source))
            for mr_id, source in (sa_source_overrides or {}).items()
        }
        normalized_nli_overrides = {
            mr_id: _normalize_nli_examples(list(source))
            for mr_id, source in (nli_source_overrides or {}).items()
        }
        normalized_topic_overrides = {
            mr_id: _normalize_topic_examples(list(source))
            for mr_id, source in (topic_source_overrides or {}).items()
        }
        rejection_rows: list[dict[str, str]] = []
        manifest: dict[str, str] = {}
        rng = random.Random(seed)

        logger.info(
            "Corpus generation starting: mr_count=%s sa_examples=%s nli_examples=%s topic_examples=%s output_dir=%s",
            len(mr_ids),
            len(sa_examples),
            len(nli_examples),
            len(topic_examples),
            output_path,
        )

        for mr_id in mr_ids:
            mr = self._get_mr_instance(mr_id)
            logger.info("Generating corpus for %s with subtasks=%s", mr_id, ",".join(mr.subtasks))
            if mr_id == "CHR-GEN-018" and self.tokenizer is not None:
                tokenizer_ok = getattr(mr, "check_tokenizer_casing")(self.tokenizer)
                if not tokenizer_ok:
                    logger.warning("Skipping %s because tokenizer casing check failed.", mr_id)
                    self._write_corpus_csv(output_path / f"{mr_id}_corpus.csv", [])
                    self._write_gen_018_manual_artifacts([], rng)
                    manifest[f"{mr_id}_corpus.csv"] = _sha256_for_file(output_path / f"{mr_id}_corpus.csv")
                    continue
            elif mr_id == "CHR-GEN-018" and self.tokenizer is None:
                logger.warning("No tokenizer provided for CHR-GEN-018; assuming the tokenizer is cased.")

            records: list[CorpusRecord] = []
            attempts = 0
            skips = 0

            if "SA" in mr.subtasks:
                mr_sa_examples = normalized_sa_overrides.get(mr_id, sa_examples)
                generated_records, generated_rejections, sa_attempts, sa_skips = self._generate_for_subtask(
                    mr_id=mr_id,
                    mr=mr,
                    subtask="SA",
                    examples=mr_sa_examples,
                    seed=seed,
                )
                records.extend(generated_records)
                rejection_rows.extend(generated_rejections)
                attempts += sa_attempts
                skips += sa_skips

            if "NLI" in mr.subtasks:
                mr_nli_examples = normalized_nli_overrides.get(mr_id, nli_examples)
                generated_records, generated_rejections, nli_attempts, nli_skips = self._generate_for_subtask(
                    mr_id=mr_id,
                    mr=mr,
                    subtask="NLI",
                    examples=mr_nli_examples,
                    seed=seed,
                )
                records.extend(generated_records)
                rejection_rows.extend(generated_rejections)
                attempts += nli_attempts
                skips += nli_skips

            if "TOPIC" in mr.subtasks:
                mr_topic_examples = normalized_topic_overrides.get(mr_id, topic_examples)
                generated_records, generated_rejections, topic_attempts, topic_skips = self._generate_for_subtask(
                    mr_id=mr_id,
                    mr=mr,
                    subtask="TOPIC",
                    examples=mr_topic_examples,
                    seed=seed,
                )
                records.extend(generated_records)
                rejection_rows.extend(generated_rejections)
                attempts += topic_attempts
                skips += topic_skips

            if mr_id == "CHR-GEN-019" and attempts > 0 and skips / attempts > 0.15:
                logger.warning("CHR-GEN-019 skip rate exceeded 15%% during corpus generation.")

            corpus_file = output_path / f"{mr_id}_corpus.csv"
            self._write_corpus_csv(corpus_file, records)
            manifest[corpus_file.name] = _sha256_for_file(corpus_file)
            self._write_manual_validation_artifacts(mr_id, records, rng)
            logger.info(
                "Finished %s: attempts=%s skips=%s kept=%s corpus=%s",
                mr_id,
                attempts,
                skips,
                len(records),
                corpus_file,
            )

        self._write_rejection_log(output_path / "rejection_log.csv", rejection_rows)
        manifest_path = output_path / "corpus_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        logger.info("Corpus generation complete: manifest=%s rejections=%s", manifest_path, len(rejection_rows))

    def _generate_for_subtask(
        self,
        mr_id: str,
        mr: BaseMR,
        subtask: str,
        examples: list[dict],
        seed: int,
    ) -> tuple[list[CorpusRecord], list[dict[str, str]], int, int]:
        records: list[CorpusRecord] = []
        rejections: list[dict[str, str]] = []
        attempts = 0
        skips = 0

        logger.info("  %s/%s start: source_examples=%s", mr_id, subtask, len(examples))

        for example in examples:
            attempts += 1
            source_input = self._build_source_input(mr_id, subtask, example)

            if mr_id == "CHR-GEN-018":
                for variant_name in ["uppercase", "lowercase"]:
                    followup = mr.transform(source_input, seed=seed, variant=variant_name)
                    if followup is None:
                        skips += 1
                        reason = mr.last_skip_reason or "transform_skipped"
                        rejections.append(self._rejection_row(mr_id, example["input_id"], subtask, variant_name, reason, example))
                        continue
                    record_or_reason = self._build_corpus_record(
                        mr_id=mr_id,
                        subtask=subtask,
                        example=example,
                        source_input=source_input,
                        followup_input=followup,
                        variant=variant_name,
                    )
                    if isinstance(record_or_reason, CorpusRecord):
                        records.append(record_or_reason)
                    else:
                        rejections.append(
                            self._rejection_row(mr_id, example["input_id"], subtask, variant_name, record_or_reason, example)
                        )
            else:
                followup = mr.transform(source_input, seed=seed)
                if followup is None:
                    skips += 1
                    reason = mr.last_skip_reason or "transform_skipped"
                    rejections.append(self._rejection_row(mr_id, example["input_id"], subtask, "", reason, example))
                    continue

                record_or_reason = self._build_corpus_record(
                    mr_id=mr_id,
                    subtask=subtask,
                    example=example,
                    source_input=source_input,
                    followup_input=followup,
                    variant=None,
                )
                if isinstance(record_or_reason, CorpusRecord):
                    records.append(record_or_reason)
                else:
                    rejections.append(self._rejection_row(mr_id, example["input_id"], subtask, "", record_or_reason, example))

            if attempts % GENERATION_PROGRESS_EVERY == 0:
                logger.info(
                    "  %s/%s progress: processed=%s kept=%s skipped=%s rejected=%s",
                    mr_id,
                    subtask,
                    attempts,
                    len(records),
                    skips,
                    len(rejections),
                )

        logger.info(
            "  %s/%s done: processed=%s kept=%s skipped=%s rejected=%s",
            mr_id,
            subtask,
            attempts,
            len(records),
            skips,
            len(rejections),
        )
        return records, rejections, attempts, skips

    def _build_corpus_record(
        self,
        mr_id: str,
        subtask: str,
        example: dict,
        source_input,
        followup_input,
        variant: str | None,
    ) -> CorpusRecord | str:
        valid, reason = self.validator.validate_pair(mr_id, source_input, followup_input)
        if not valid:
            return reason

        if subtask in {"SA", "TOPIC"}:
            source_text = example["text"]
            followup_text = str(followup_input)
        else:
            source_text = _serialize_nli_input(example["premise"], example["hypothesis"])
            followup_text = _serialize_nli_input(followup_input["premise"], followup_input["hypothesis"])

        return CorpusRecord(
            mr_id=mr_id,
            input_id=example["input_id"],
            subtask=subtask,
            source_text=source_text,
            source_label=example["source_label"],
            followup_text=followup_text,
            expected_output_relation=self._expected_output_relation(mr_id, example["source_label"]),
            variant=variant,
            skip_reason=None,
        )

    @staticmethod
    def _build_source_input(mr_id: str, subtask: str, example: dict):
        if subtask in {"SA", "TOPIC"}:
            if mr_id.startswith("CHR-SA-"):
                return {"text": example["text"], "source_label": example["source_label"]}
            return example["text"]
        return {
            "premise": example["premise"],
            "hypothesis": example["hypothesis"],
            "source_label": example["source_label"],
        }

    @staticmethod
    def _expected_output_relation(mr_id: str, source_label: int) -> str:
        if mr_id == "CHR-SA-001":
            return "label_flip"
        if mr_id in {"CHR-SA-007", "CHR-SA-008", "CHR-SA-010"}:
            return "score_increase" if source_label == 1 else "score_decrease"
        if mr_id == "CHR-NLI-006":
            return "entailment_contradiction_flip"
        return "label_unchanged"

    @staticmethod
    def _rejection_row(mr_id: str, input_id: str, subtask: str, variant: str | None, reason: str, example: dict) -> dict[str, str]:
        source_text = example["text"] if subtask in {"SA", "TOPIC"} else _serialize_nli_input(example["premise"], example["hypothesis"])
        return {
            "mr_id": mr_id,
            "input_id": input_id,
            "subtask": subtask,
            "variant": variant or "",
            "reason": reason,
            "source_text": source_text,
        }

    def _write_manual_validation_artifacts(self, mr_id: str, records: list[CorpusRecord], rng: random.Random) -> None:
        if mr_id == "CHR-GEN-018":
            self._write_gen_018_manual_artifacts(records, rng)
            return

        selected = records if len(records) <= 50 else rng.sample(records, 50)
        path = self.manual_validation_dir / f"manual_validation_artifacts_{mr_id}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANUAL_VALIDATION_FIELDNAMES)
            writer.writeheader()
            for record in selected:
                writer.writerow(
                    {
                        "input_id": record.input_id,
                        "source_text": record.source_text,
                        "followup_text": record.followup_text,
                        "source_label": record.source_label,
                        "expected_output_relation": record.expected_output_relation,
                        "automated_checks_passed": True,
                        "notes": "",
                    }
                )

    def _write_gen_018_manual_artifacts(self, records: list[CorpusRecord], rng: random.Random) -> None:
        variant_map = {"uppercase": "A", "lowercase": "B"}
        for variant, suffix in variant_map.items():
            variant_records = [record for record in records if record.variant == variant]
            selected = variant_records if len(variant_records) <= 50 else rng.sample(variant_records, 50)
            path = self.manual_validation_dir / f"manual_validation_artifacts_CHR-GEN-018-{suffix}.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=MANUAL_VALIDATION_FIELDNAMES)
                writer.writeheader()
                for record in selected:
                    writer.writerow(
                        {
                            "input_id": record.input_id,
                            "source_text": record.source_text,
                            "followup_text": record.followup_text,
                            "source_label": record.source_label,
                            "expected_output_relation": record.expected_output_relation,
                            "automated_checks_passed": True,
                            "notes": "",
                        }
                    )

    @staticmethod
    def _write_corpus_csv(path: Path, records: list[CorpusRecord]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CORPUS_FIELDNAMES)
            writer.writeheader()
            for record in records:
                writer.writerow(record.to_csv_row())

    @staticmethod
    def _write_rejection_log(path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=REJECTION_FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)

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
