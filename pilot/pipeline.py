from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import subprocess
import sys
from typing import Any

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover
    load_dataset = None

import pyarrow.ipc as ipc

from chrysalis.config import SEED
from chrysalis.corpus.generator import CorpusGenerator
from chrysalis.regression.differ import RegressionDiffer, RegressionReport
from chrysalis.snapshot.engine import SnapshotEngine

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
PILOT_ROOT = ROOT / "pilot"
ARTIFACT_ROOT = PILOT_ROOT / "artifacts"
DEFAULT_CORPUS_DIR = ARTIFACT_ROOT / "corpus"
DEFAULT_SNAPSHOT_DIR = ARTIFACT_ROOT / "snapshots"
DEFAULT_REPORT_DIR = ARTIFACT_ROOT / "regression_reports"
DEFAULT_MANUAL_VALIDATION_DIR = ARTIFACT_ROOT / "manual_validation"
DEFAULT_SST2_CACHE_DIR = PILOT_ROOT / "data" / "sst2"
DEFAULT_SNLI_CACHE_DIR = PILOT_ROOT / "data" / "snli"


class SimpleCasedTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(char) for char in text]


def _resolve_import_spec(spec: str):
    module_name, _, attr_name = spec.partition(":")
    if not module_name or not attr_name:
        raise ValueError(f"Invalid import spec: {spec}")

    if module_name.endswith(".py") or module_name.startswith("/"):
        import importlib.util

        module_path = Path(module_name).resolve()
        module_spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"Unable to load module from path: {module_path}")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        import importlib

        module = importlib.import_module(module_name)

    return getattr(module, attr_name)


def _load_model_bundle(loader_spec: str, model_version: str):
    loader = _resolve_import_spec(loader_spec)
    model_dir = PILOT_ROOT / "models" / model_version
    try:
        bundle = loader(model_version=model_version, model_dir=model_dir)
    except TypeError:
        try:
            bundle = loader(model_version)
        except TypeError:
            bundle = loader()

    if isinstance(bundle, dict):
        return bundle["model"], bundle["tokenizer"]
    if isinstance(bundle, tuple) and len(bundle) == 2:
        return bundle
    raise TypeError("Model loader must return (model, tokenizer) or {'model': model, 'tokenizer': tokenizer}.")


def _load_loader_map(value: str | None) -> dict[str, str]:
    if value is None:
        return {}
    candidate = Path(value)
    if candidate.exists():
        return json.loads(candidate.read_text(encoding="utf-8"))
    return json.loads(value)


def _load_dataset_with_fallback(
    primary_name: str,
    split: str,
    cache_dir: Path,
    fallback_arrow_path: Path | None = None,
    config_name: str | None = None,
    limit: int | None = None,
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    if load_dataset is not None:
        try:
            if config_name is None:
                dataset = load_dataset(primary_name, split=split, cache_dir=str(cache_dir))
            else:
                dataset = load_dataset(primary_name, config_name, split=split, cache_dir=str(cache_dir))
            return list(dataset if limit is None else dataset.select(range(min(limit, len(dataset)))))
        except Exception as exc:  # pragma: no cover
            logger.warning("Falling back to cached Arrow data for %s: %s", primary_name, exc)

    if fallback_arrow_path is None or not fallback_arrow_path.exists():
        raise FileNotFoundError(f"Unable to load dataset {primary_name}; fallback cache not found.")

    with ipc.open_stream(fallback_arrow_path) as reader:
        rows = reader.read_all().to_pylist()
    return rows[:limit] if limit is not None else rows


def _load_sst2_validation(limit: int | None = None) -> list[dict]:
    fallback = Path(
        "/Users/shazzad/.cache/huggingface/datasets/stanfordnlp___sst2/default/0.0.0/"
        "8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb/sst2-validation.arrow"
    )
    try:
        return _load_dataset_with_fallback(
            primary_name="glue",
            config_name="sst2",
            split="validation",
            cache_dir=DEFAULT_SST2_CACHE_DIR,
            fallback_arrow_path=None,
            limit=limit,
        )
    except Exception:
        return _load_dataset_with_fallback(
            primary_name="stanfordnlp/sst2",
            split="validation",
            cache_dir=DEFAULT_SST2_CACHE_DIR,
            fallback_arrow_path=fallback,
            limit=limit,
        )


def _load_snli_validation(limit: int | None = None) -> list[dict]:
    fallback = Path(
        "/Users/shazzad/.cache/huggingface/datasets/snli/plain_text/0.0.0/"
        "cdb5c3d5eed6ead6e5a341c8e56e669bb666725b/snli-validation.arrow"
    )
    return _load_dataset_with_fallback(
        primary_name="snli",
        split="validation",
        cache_dir=DEFAULT_SNLI_CACHE_DIR,
        fallback_arrow_path=fallback,
        limit=limit,
    )


def _sanitize_transition(transition: str) -> str:
    return transition.replace("→", "_to_").replace("->", "_to_").replace("/", "_")


def _parse_transition(transition: str | None, old_version: str | None, new_version: str | None) -> tuple[str, str, str]:
    if transition:
        if "→" in transition:
            old, new = transition.split("→", 1)
        elif "->" in transition:
            old, new = transition.split("->", 1)
        else:
            raise ValueError("Transition must use '->' or '→'.")
        return transition.replace("->", "→"), old.strip(), new.strip()

    if not old_version or not new_version:
        raise ValueError("Provide either --transition or both --old-version and --new-version.")
    return f"{old_version}→{new_version}", old_version, new_version


def _default_report_paths(report_dir: Path, transition: str) -> tuple[Path, Path]:
    suffix = _sanitize_transition(transition)
    return (
        report_dir / f"regression_report_{suffix}.csv",
        report_dir / f"fairness_regression_report_{suffix}.csv",
    )


def _format_summary_table(reports: list[RegressionReport]) -> str:
    headers = ["MR ID", "transition", "n_matched", "pass_rate_old", "pass_rate_new", "delta", "flag", "severity"]
    rows = [
        [
            report.mr_id,
            report.transition,
            str(report.n_matched),
            f"{report.pass_rate_old:.3f}",
            f"{report.pass_rate_new:.3f}",
            f"{report.matched_pass_rate_delta:.3f}",
            str(report.behavioral_regression_flag),
            report.pipeline_severity,
        ]
        for report in reports
    ]
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def render(row: list[str]) -> str:
        return " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))

    separator = "-+-".join("-" * width for width in widths)
    return "\n".join([render(headers), separator, *(render(row) for row in rows)])


def run_training_stage(versions: list[str]) -> None:
    for version in versions:
        script = PILOT_ROOT / "training" / f"train_{version.split('_', 1)[0]}.py"
        if not script.exists():
            raise FileNotFoundError(f"Training script not found: {script}")
        subprocess.run([sys.executable, str(script)], check=True)


def run_corpus_stage(args) -> None:
    sa_source = _load_sst2_validation(limit=args.sa_limit)
    nli_source = _load_snli_validation(limit=args.nli_limit)
    tokenizer = SimpleCasedTokenizer()
    generator = CorpusGenerator(
        tokenizer=tokenizer,
        manual_validation_dir=Path(args.manual_validation_dir),
    )
    mr_ids = [record["mr_id"] for record in generator.registry_loader.load()]
    generator.generate(
        mr_ids=mr_ids,
        sa_source=sa_source,
        nli_source=nli_source,
        output_dir=args.corpus_dir,
        seed=args.seed,
    )


def run_snapshot_stage(args, model_version: str | None = None, model_loader: str | None = None) -> None:
    resolved_model_version = model_version or args.model_version
    resolved_loader = model_loader or args.model_loader
    if not resolved_model_version:
        raise ValueError("--model-version is required for snapshot stage.")
    if not resolved_loader:
        raise ValueError("--model-loader is required for snapshot stage.")

    model, tokenizer = _load_model_bundle(resolved_loader, resolved_model_version)
    engine = SnapshotEngine()
    engine.run(
        model=model,
        tokenizer=tokenizer,
        model_version=resolved_model_version,
        corpus_dir=args.corpus_dir,
        output_dir=args.snapshot_dir,
    )


def run_diff_stage(args, transition: str | None = None, old_version: str | None = None, new_version: str | None = None) -> list[RegressionReport]:
    resolved_transition, resolved_old, resolved_new = _parse_transition(transition or args.transition, old_version or args.old_version, new_version or args.new_version)
    differ = RegressionDiffer()
    reports = differ.diff_transition(
        transition=resolved_transition,
        old_version=resolved_old,
        new_version=resolved_new,
        snapshot_dir=args.snapshot_dir,
        corpus_dir=args.corpus_dir,
    )

    report_dir = Path(args.report_dir)
    standard_path = Path(args.standard_report_path) if args.standard_report_path else _default_report_paths(report_dir, resolved_transition)[0]
    fairness_path = Path(args.fairness_report_path) if args.fairness_report_path else _default_report_paths(report_dir, resolved_transition)[1]
    differ.write_report(reports, str(standard_path))
    fairness_reports = [report for report in reports if report.mr_id == "CHR-NLI-005" and report.behavioral_regression_flag]
    if fairness_reports:
        differ.write_fairness_report(reports, str(fairness_path))

    print(_format_summary_table(reports))
    return reports


def run_all_stage(args) -> None:
    loader_map = _load_loader_map(args.model_loader_map)
    required_versions = ["v1_base", "v2_retrain", "v3_distilled"]
    missing = [version for version in required_versions if version not in loader_map]
    if missing:
        raise ValueError(f"--model-loader-map must provide loaders for: {', '.join(missing)}")

    run_training_stage(["v1_base"])
    run_corpus_stage(args)
    run_training_stage(["v2_retrain", "v3_distilled"])
    for version in required_versions:
        run_snapshot_stage(args, model_version=version, model_loader=loader_map[version])

    all_reports: list[RegressionReport] = []
    all_reports.extend(run_diff_stage(args, transition="v1_base→v2_retrain", old_version="v1_base", new_version="v2_retrain"))
    all_reports.extend(run_diff_stage(args, transition="v2_retrain→v3_distilled", old_version="v2_retrain", new_version="v3_distilled"))

    print()
    print(_format_summary_table(all_reports))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Chrysalis pilot pipeline.")
    parser.add_argument("--stage", required=True, choices=["all", "train", "corpus", "snapshot", "diff"])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--sa-limit", type=int, default=None)
    parser.add_argument("--nli-limit", type=int, default=None)
    parser.add_argument("--model-version")
    parser.add_argument("--model-loader")
    parser.add_argument("--model-loader-map")
    parser.add_argument("--transition")
    parser.add_argument("--old-version")
    parser.add_argument("--new-version")
    parser.add_argument("--corpus-dir", default=str(DEFAULT_CORPUS_DIR))
    parser.add_argument("--snapshot-dir", default=str(DEFAULT_SNAPSHOT_DIR))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--manual-validation-dir", default=str(DEFAULT_MANUAL_VALIDATION_DIR))
    parser.add_argument("--standard-report-path")
    parser.add_argument("--fairness-report-path")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args()

    if args.stage == "train":
        run_training_stage(["v1_base", "v2_retrain", "v3_distilled"])
        return
    if args.stage == "corpus":
        run_corpus_stage(args)
        return
    if args.stage == "snapshot":
        run_snapshot_stage(args)
        return
    if args.stage == "diff":
        run_diff_stage(args)
        return
    run_all_stage(args)


if __name__ == "__main__":
    main()
