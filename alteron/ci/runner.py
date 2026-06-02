from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import importlib
import importlib.resources
import importlib.util
import json
import logging
from pathlib import Path
import random
import shutil
import sys
from typing import Any

import yaml

from alteron.config import REGRESSION_THRESHOLD, SEED
from alteron.corpus.schemas import CorpusRecord
from alteron.corpus.generator import CORPUS_FIELDNAMES
from alteron.regression.differ import RegressionDiffer, RegressionReport
from alteron.registry.registry import RegistryLoader
from alteron.snapshot.engine import SnapshotEngine

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CIProfile:
    name: str
    mr_ids: list[str] | str
    max_records_per_mr: int | None
    fail_on_severity: list[str]
    regression_threshold: float | None


@dataclass(slots=True)
class CIRunConfig:
    seed: int
    regression_threshold: float
    profiles: dict[str, CIProfile]
    run: dict[str, Any]


@dataclass(slots=True)
class CISummary:
    profile: str
    baseline_version: str
    candidate_version: str
    mr_ids: list[str]
    corpus_dir: str
    working_corpus_dir: str
    baseline_snapshot_dir: str
    candidate_snapshot_dir: str
    output_dir: str
    regression_threshold: float
    reports_written: list[str]
    blocking_regressions: int
    fairness_alerts: int
    exit_code: int

    def to_json(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "baseline_version": self.baseline_version,
            "candidate_version": self.candidate_version,
            "mr_ids": self.mr_ids,
            "corpus_dir": self.corpus_dir,
            "working_corpus_dir": self.working_corpus_dir,
            "baseline_snapshot_dir": self.baseline_snapshot_dir,
            "candidate_snapshot_dir": self.candidate_snapshot_dir,
            "output_dir": self.output_dir,
            "regression_threshold": self.regression_threshold,
            "reports_written": self.reports_written,
            "blocking_regressions": self.blocking_regressions,
            "fairness_alerts": self.fairness_alerts,
            "exit_code": self.exit_code,
        }


def default_config_path() -> Path:
    return Path(str(importlib.resources.files("alteron.ci") / "profiles.yml"))


def load_ci_config(path: str | Path | None = None) -> CIRunConfig:
    config_path = Path(path) if path is not None else default_config_path()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    profiles: dict[str, CIProfile] = {}
    for name, profile_data in (raw.get("profiles") or {}).items():
        profiles[name] = CIProfile(
            name=name,
            mr_ids=profile_data.get("mr_ids", "all"),
            max_records_per_mr=profile_data.get("max_records_per_mr"),
            fail_on_severity=list(profile_data.get("fail_on_severity", ["hard-fail"])),
            regression_threshold=(
                float(profile_data["regression_threshold"])
                if profile_data.get("regression_threshold") is not None
                else None
            ),
        )
    if not profiles:
        raise ValueError(f"No CI profiles configured in {config_path}")

    return CIRunConfig(
        seed=int(raw.get("seed", SEED)),
        regression_threshold=float(raw.get("regression_threshold", REGRESSION_THRESHOLD)),
        profiles=profiles,
        run=dict(raw.get("run") or {}),
    )


def run_ci_check(args: argparse.Namespace) -> CISummary:
    run_config = load_ci_config(args.config)
    if args.profile not in run_config.profiles:
        available = ", ".join(sorted(run_config.profiles))
        raise ValueError(f"Unknown CI profile {args.profile!r}. Available profiles: {available}")

    profile = run_config.profiles[args.profile]
    regression_threshold = (
        float(args.regression_threshold)
        if getattr(args, "regression_threshold", None) is not None
        else profile.regression_threshold
        if profile.regression_threshold is not None
        else run_config.regression_threshold
    )
    candidate_model_dir_value = _resolve_runtime_value(args, run_config.run, "candidate_model_dir")
    candidate_version = _resolve_runtime_value(args, run_config.run, "candidate_version")
    baseline_snapshot_dir_value = _resolve_runtime_value(args, run_config.run, "baseline_snapshot_dir")
    baseline_version = _resolve_runtime_value(args, run_config.run, "baseline_version")
    corpus_dir_value = _resolve_runtime_value(args, run_config.run, "corpus_dir")
    output_dir_value = _resolve_runtime_value(args, run_config.run, "output_dir")
    model_loader = _resolve_runtime_value(args, run_config.run, "model_loader")

    output_dir = Path(output_dir_value)
    snapshot_root = output_dir / "snapshots"
    report_dir = output_dir / "regression_reports"
    working_corpus_dir = output_dir / "working_corpus"
    summary_path = output_dir / "ci_summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    corpus_dir = Path(corpus_dir_value)
    baseline_snapshot_dir = Path(baseline_snapshot_dir_value)
    candidate_model_dir = Path(candidate_model_dir_value)
    if not candidate_model_dir.exists():
        raise FileNotFoundError(f"Candidate model directory does not exist: {candidate_model_dir}")
    if not baseline_snapshot_dir.exists():
        raise FileNotFoundError(f"Baseline snapshot directory does not exist: {baseline_snapshot_dir}")

    registry_loader = RegistryLoader()
    snapshot_engine = SnapshotEngine(registry_loader=registry_loader)
    snapshot_engine.verify_corpus_hashes(corpus_dir)
    selected_mr_ids = resolve_mr_ids(profile.mr_ids, corpus_dir, registry_loader)
    prepare_working_corpus(
        source_corpus_dir=corpus_dir,
        output_corpus_dir=working_corpus_dir,
        mr_ids=selected_mr_ids,
        max_records_per_mr=profile.max_records_per_mr,
        seed=run_config.seed,
    )

    model, tokenizer = load_model_bundle(
        loader_spec=model_loader,
        candidate_version=candidate_version,
        candidate_model_dir=candidate_model_dir,
    )

    candidate_snapshot_dir = snapshot_root / candidate_version
    if args.force and candidate_snapshot_dir.exists():
        shutil.rmtree(candidate_snapshot_dir)

    snapshot_engine.run(
        model=model,
        tokenizer=tokenizer,
        model_version=candidate_version,
        corpus_dir=str(working_corpus_dir),
        output_dir=str(snapshot_root),
    )

    reports = diff_against_baseline(
        baseline_version=baseline_version,
        candidate_version=candidate_version,
        baseline_snapshot_dir=baseline_snapshot_dir,
        candidate_snapshot_dir=candidate_snapshot_dir,
        corpus_dir=working_corpus_dir,
        mr_ids=selected_mr_ids,
        fail_on_severity=profile.fail_on_severity,
        regression_threshold=regression_threshold,
        registry_loader=registry_loader,
    )

    transition = f"{baseline_version}→{candidate_version}"
    suffix = _sanitize_transition(transition)
    standard_report_path = report_dir / f"regression_report_{suffix}.csv"
    fairness_report_path = report_dir / f"fairness_regression_report_{suffix}.csv"

    differ = RegressionDiffer(registry_loader=registry_loader, regression_threshold=regression_threshold)
    differ.write_report(reports, str(standard_report_path))
    reports_written = [str(standard_report_path)]
    fairness_alerts = sum(
        1 for report in reports if report.mr_id == "CHR-NLI-005" and report.behavioral_regression_flag
    )
    if fairness_alerts:
        differ.write_fairness_report(reports, str(fairness_report_path))
        reports_written.append(str(fairness_report_path))

    blocking_regressions = sum(
        1
        for report in reports
        if report.behavioral_regression_flag and report.pipeline_severity in profile.fail_on_severity
    )
    exit_code = 1 if blocking_regressions else 0
    summary = CISummary(
        profile=profile.name,
        baseline_version=baseline_version,
        candidate_version=candidate_version,
        mr_ids=selected_mr_ids,
        corpus_dir=str(corpus_dir),
        working_corpus_dir=str(working_corpus_dir),
        baseline_snapshot_dir=str(baseline_snapshot_dir),
        candidate_snapshot_dir=str(candidate_snapshot_dir),
        output_dir=str(output_dir),
        regression_threshold=regression_threshold,
        reports_written=reports_written,
        blocking_regressions=blocking_regressions,
        fairness_alerts=fairness_alerts,
        exit_code=exit_code,
    )
    summary_path.write_text(json.dumps(summary.to_json(), indent=2) + "\n", encoding="utf-8")
    logger.info("CI summary written to %s", summary_path)
    return summary


def resolve_mr_ids(mr_selection: list[str] | str, corpus_dir: Path, registry_loader: RegistryLoader) -> list[str]:
    manifest = _load_manifest(corpus_dir)
    corpus_mr_ids = {
        filename.removesuffix("_corpus.csv")
        for filename in manifest
        if filename.endswith("_corpus.csv")
    }
    if mr_selection == "all":
        registry_order = [record["mr_id"] for record in registry_loader.load()]
        selected = [mr_id for mr_id in registry_order if mr_id in corpus_mr_ids]
    else:
        selected = list(mr_selection)
        missing = [mr_id for mr_id in selected if mr_id not in corpus_mr_ids]
        if missing:
            raise FileNotFoundError(f"Selected MRs are missing from corpus manifest: {', '.join(missing)}")
    if not selected:
        raise ValueError(f"No MR corpus files selected from {corpus_dir}")
    return selected


def prepare_working_corpus(
    source_corpus_dir: Path,
    output_corpus_dir: Path,
    mr_ids: list[str],
    max_records_per_mr: int | None,
    seed: int,
) -> None:
    if output_corpus_dir.exists():
        shutil.rmtree(output_corpus_dir)
    output_corpus_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, str] = {}
    rng = random.Random(seed)
    for mr_id in mr_ids:
        source_file = source_corpus_dir / f"{mr_id}_corpus.csv"
        output_file = output_corpus_dir / source_file.name
        if max_records_per_mr is None:
            shutil.copy2(source_file, output_file)
        else:
            records = _read_corpus_records(source_file)
            sorted_records = sorted(records, key=lambda record: (record.input_id, record.variant or ""))
            sample_size = min(max_records_per_mr, len(sorted_records))
            if sample_size < len(sorted_records):
                sampled_records = sorted(
                    rng.sample(sorted_records, sample_size),
                    key=lambda record: (record.input_id, record.variant or ""),
                )
            else:
                sampled_records = sorted_records
            _write_corpus_records(output_file, sampled_records)
        manifest[output_file.name] = _sha256_for_file(output_file)

    (output_corpus_dir / "corpus_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def diff_against_baseline(
    baseline_version: str,
    candidate_version: str,
    baseline_snapshot_dir: Path,
    candidate_snapshot_dir: Path,
    corpus_dir: Path,
    mr_ids: list[str],
    fail_on_severity: list[str],
    regression_threshold: float,
    registry_loader: RegistryLoader,
) -> list[RegressionReport]:
    differ = RegressionDiffer(registry_loader=registry_loader, regression_threshold=regression_threshold)
    transition = f"{baseline_version}→{candidate_version}"
    reports: list[RegressionReport] = []
    for mr_id in mr_ids:
        report = differ.diff(
            mr_id=mr_id,
            old_snapshot_path=str(baseline_snapshot_dir / f"{mr_id}_snapshot.csv"),
            new_snapshot_path=str(candidate_snapshot_dir / f"{mr_id}_snapshot.csv"),
            ground_truth_path=str(corpus_dir / f"{mr_id}_corpus.csv"),
        )
        report.transition = transition
        report.release_blocked = report.behavioral_regression_flag and report.pipeline_severity in fail_on_severity
        reports.append(report)
    return reports


def load_model_bundle(loader_spec: str, candidate_version: str, candidate_model_dir: Path):
    loader = _resolve_import_spec(loader_spec)
    try:
        bundle = loader(model_version=candidate_version, model_dir=candidate_model_dir)
    except TypeError:
        try:
            bundle = loader(candidate_model_dir)
        except TypeError:
            try:
                bundle = loader(candidate_version)
            except TypeError:
                bundle = loader()

    if isinstance(bundle, dict):
        return bundle["model"], bundle["tokenizer"]
    if isinstance(bundle, tuple) and len(bundle) == 2:
        return bundle
    raise TypeError("Model loader must return (model, tokenizer) or {'model': model, 'tokenizer': tokenizer}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="alteron-ci", description="Run an Alteron CI behavioral regression check.")
    parser.add_argument("--config", default=str(default_config_path()))
    parser.add_argument("--profile", required=True)
    parser.add_argument("--candidate-model-dir")
    parser.add_argument("--candidate-version")
    parser.add_argument("--baseline-snapshot-dir")
    parser.add_argument("--baseline-version")
    parser.add_argument("--corpus-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--model-loader")
    parser.add_argument(
        "--regression-threshold",
        type=float,
        help=f"Override the configured behavioral regression threshold. Default: {REGRESSION_THRESHOLD}.",
    )
    parser.add_argument("--force", action="store_true", help="Remove an existing candidate snapshot directory before running.")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    summary = run_ci_check(args)
    print(json.dumps(summary.to_json(), indent=2))
    raise SystemExit(summary.exit_code)


def _resolve_import_spec(spec: str):
    module_name, _, attr_name = spec.rpartition(":")
    if not module_name or not attr_name:
        raise ValueError(f"Invalid import spec: {spec}")

    if module_name.endswith(".py") or module_name.startswith("/"):
        module_path = Path(module_name).resolve()
        module_spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"Unable to load module from path: {module_path}")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)

    return getattr(module, attr_name)


def _load_manifest(corpus_dir: Path) -> dict[str, str]:
    with (corpus_dir / "corpus_manifest.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_corpus_records(path: Path) -> list[CorpusRecord]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [CorpusRecord.from_csv_row(row) for row in csv.DictReader(handle)]


def _write_corpus_records(path: Path, records: list[CorpusRecord]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CORPUS_FIELDNAMES)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_csv_row())


def _sha256_for_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sanitize_transition(transition: str) -> str:
    return transition.replace("→", "_to_").replace("->", "_to_").replace("/", "_")


def _resolve_runtime_value(args: argparse.Namespace, config_run: dict[str, Any], name: str) -> Any:
    cli_value = getattr(args, name, None)
    if cli_value is not None:
        return cli_value
    if name in config_run and config_run[name] is not None:
        return config_run[name]
    raise ValueError(
        f"Missing required runtime setting {name!r}. Provide it on the command line or under the 'run' section of the CI config."
    )


if __name__ == "__main__":
    main(sys.argv[1:])
