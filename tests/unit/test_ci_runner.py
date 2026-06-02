from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from alteron.ci import runner
from alteron.corpus.generator import CORPUS_FIELDNAMES
from alteron.corpus.schemas import CorpusRecord, SnapshotRecord
from alteron.snapshot.engine import SNAPSHOT_FIELDNAMES


def _sha256_for_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_corpus(path: Path, records: list[CorpusRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CORPUS_FIELDNAMES)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_csv_row())


def _write_snapshot(path: Path, records: list[SnapshotRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SNAPSHOT_FIELDNAMES)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_csv_row())


def _corpus_record(mr_id: str, input_id: str, label: int = 1) -> CorpusRecord:
    return CorpusRecord(
        mr_id=mr_id,
        input_id=input_id,
        subtask="SA",
        source_text=f"source {input_id}",
        source_label=label,
        followup_text=f"followup {input_id}",
        expected_output_relation="label_unchanged",
        variant=None,
        skip_reason=None,
    )


def _snapshot_record(model_version: str, mr_id: str, input_id: str, label: int, mr_pass: bool) -> SnapshotRecord:
    return SnapshotRecord(
        model_version=model_version,
        mr_id=mr_id,
        input_id=input_id,
        variant=None,
        source_pred_label=label,
        source_pred_score=0.9,
        followup_pred_label=label,
        followup_pred_score=0.9,
        mr_pass=mr_pass,
        fairness_regression=False,
        timestamp="2026-04-14T00:00:00+00:00",
    )


def _write_manifest(corpus_dir: Path) -> None:
    manifest = {
        path.name: _sha256_for_file(path)
        for path in sorted(corpus_dir.glob("*_corpus.csv"))
    }
    (corpus_dir / "corpus_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def test_prepare_working_corpus_samples_selected_mrs_deterministically(tmp_path):
    corpus_dir = tmp_path / "corpus"
    gen_005_records = [_corpus_record("CHR-GEN-005", f"gen005-{index:03d}") for index in range(25)]
    gen_018_records = [_corpus_record("CHR-GEN-018", f"gen018-{index:03d}") for index in range(8)]
    _write_corpus(corpus_dir / "CHR-GEN-005_corpus.csv", gen_005_records)
    _write_corpus(corpus_dir / "CHR-GEN-018_corpus.csv", gen_018_records)
    _write_manifest(corpus_dir)

    output_a = tmp_path / "work-a"
    output_b = tmp_path / "work-b"
    runner.prepare_working_corpus(corpus_dir, output_a, ["CHR-GEN-005", "CHR-GEN-018"], max_records_per_mr=10, seed=42)
    runner.prepare_working_corpus(corpus_dir, output_b, ["CHR-GEN-005", "CHR-GEN-018"], max_records_per_mr=10, seed=42)

    with (output_a / "CHR-GEN-005_corpus.csv").open("r", newline="", encoding="utf-8") as handle:
        sampled_gen_005 = list(csv.DictReader(handle))
    with (output_a / "CHR-GEN-018_corpus.csv").open("r", newline="", encoding="utf-8") as handle:
        sampled_gen_018 = list(csv.DictReader(handle))

    assert len(sampled_gen_005) == 10
    assert len(sampled_gen_018) == 8
    assert (output_a / "corpus_manifest.json").read_text(encoding="utf-8") == (
        output_b / "corpus_manifest.json"
    ).read_text(encoding="utf-8")
    assert (output_a / "CHR-GEN-005_corpus.csv").read_text(encoding="utf-8") == (
        output_b / "CHR-GEN-005_corpus.csv"
    ).read_text(encoding="utf-8")


def test_run_ci_check_writes_summary_and_returns_blocking_exit_code(tmp_path, monkeypatch):
    config_path = tmp_path / "alteron_ci.yml"
    config_path.write_text(
        """
version: 1
seed: 42
regression_threshold: -0.05
profiles:
  pr-fast:
    mr_ids:
      - CHR-GEN-005
    max_records_per_mr: null
    fail_on_severity:
      - hard-fail
""",
        encoding="utf-8",
    )

    corpus_dir = tmp_path / "corpus"
    _write_corpus(corpus_dir / "CHR-GEN-005_corpus.csv", [_corpus_record("CHR-GEN-005", "row-1", label=1)])
    _write_manifest(corpus_dir)

    baseline_snapshot_dir = tmp_path / "baseline_snapshots"
    _write_snapshot(
        baseline_snapshot_dir / "CHR-GEN-005_snapshot.csv",
        [_snapshot_record("stable", "CHR-GEN-005", "row-1", label=1, mr_pass=True)],
    )

    candidate_model_dir = tmp_path / "models" / "candidate"
    candidate_model_dir.mkdir(parents=True)

    class FakeSnapshotEngine:
        def __init__(self, registry_loader=None):
            del registry_loader

        def verify_corpus_hashes(self, corpus_dir: str | Path) -> bool:
            del corpus_dir
            return True

        def run(self, model, tokenizer, model_version: str, corpus_dir: str, output_dir: str) -> None:
            del model, tokenizer, corpus_dir
            _write_snapshot(
                Path(output_dir) / model_version / "CHR-GEN-005_snapshot.csv",
                [_snapshot_record(model_version, "CHR-GEN-005", "row-1", label=1, mr_pass=False)],
            )

    monkeypatch.setattr(runner, "SnapshotEngine", FakeSnapshotEngine)
    monkeypatch.setattr(runner, "load_model_bundle", lambda **kwargs: (object(), object()))

    args = argparse.Namespace(
        config=str(config_path),
        profile="pr-fast",
        candidate_model_dir=str(candidate_model_dir),
        candidate_version="candidate",
        baseline_snapshot_dir=str(baseline_snapshot_dir),
        baseline_version="stable",
        corpus_dir=str(corpus_dir),
        output_dir=str(tmp_path / "ci_output"),
        model_loader="unused:loader",
        regression_threshold=None,
        force=False,
    )

    summary = runner.run_ci_check(args)

    assert summary.exit_code == 1
    assert summary.blocking_regressions == 1
    assert summary.fairness_alerts == 0
    assert summary.mr_ids == ["CHR-GEN-005"]
    summary_json = json.loads((Path(args.output_dir) / "ci_summary.json").read_text(encoding="utf-8"))
    assert summary_json["exit_code"] == 1
    report_path = Path(args.output_dir) / "regression_reports" / "regression_report_stable_to_candidate.csv"
    with report_path.open("r", newline="", encoding="utf-8") as handle:
        report_rows = list(csv.DictReader(handle))
    assert report_rows[0]["behavioral_regression_flag"] == "True"
    assert report_rows[0]["release_blocked"] == "True"
    assert summary_json["regression_threshold"] == -0.05


def test_run_ci_check_allows_cli_threshold_override(tmp_path, monkeypatch):
    config_path = tmp_path / "alteron_ci.yml"
    config_path.write_text(
        """
version: 1
seed: 42
regression_threshold: -0.05
profiles:
  pr-fast:
    mr_ids:
      - CHR-GEN-005
    max_records_per_mr: null
    fail_on_severity:
      - hard-fail
""",
        encoding="utf-8",
    )

    corpus_dir = tmp_path / "corpus"
    records = [_corpus_record("CHR-GEN-005", f"row-{index}", label=1) for index in range(10)]
    _write_corpus(corpus_dir / "CHR-GEN-005_corpus.csv", records)
    _write_manifest(corpus_dir)

    baseline_snapshot_dir = tmp_path / "baseline_snapshots"
    _write_snapshot(
        baseline_snapshot_dir / "CHR-GEN-005_snapshot.csv",
        [_snapshot_record("stable", "CHR-GEN-005", f"row-{index}", label=1, mr_pass=True) for index in range(10)],
    )

    candidate_model_dir = tmp_path / "models" / "candidate"
    candidate_model_dir.mkdir(parents=True)

    class FakeSnapshotEngine:
        def __init__(self, registry_loader=None):
            del registry_loader

        def verify_corpus_hashes(self, corpus_dir: str | Path) -> bool:
            del corpus_dir
            return True

        def run(self, model, tokenizer, model_version: str, corpus_dir: str, output_dir: str) -> None:
            del model, tokenizer, corpus_dir
            candidate_rows = [
                _snapshot_record(model_version, "CHR-GEN-005", f"row-{index}", label=1, mr_pass=index != 0)
                for index in range(10)
            ]
            _write_snapshot(Path(output_dir) / model_version / "CHR-GEN-005_snapshot.csv", candidate_rows)

    monkeypatch.setattr(runner, "SnapshotEngine", FakeSnapshotEngine)
    monkeypatch.setattr(runner, "load_model_bundle", lambda **kwargs: (object(), object()))

    args = argparse.Namespace(
        config=str(config_path),
        profile="pr-fast",
        candidate_model_dir=str(candidate_model_dir),
        candidate_version="candidate",
        baseline_snapshot_dir=str(baseline_snapshot_dir),
        baseline_version="stable",
        corpus_dir=str(corpus_dir),
        output_dir=str(tmp_path / "ci_output"),
        model_loader="unused:loader",
        regression_threshold=-0.2,
        force=False,
    )

    summary = runner.run_ci_check(args)

    assert summary.exit_code == 0
    assert summary.blocking_regressions == 0
    summary_json = json.loads((Path(args.output_dir) / "ci_summary.json").read_text(encoding="utf-8"))
    assert summary_json["regression_threshold"] == -0.2
    report_path = Path(args.output_dir) / "regression_reports" / "regression_report_stable_to_candidate.csv"
    with report_path.open("r", newline="", encoding="utf-8") as handle:
        report_rows = list(csv.DictReader(handle))
    assert report_rows[0]["matched_pass_rate_delta"] == "-0.09999999999999998"
    assert report_rows[0]["behavioral_regression_flag"] == "False"


def test_load_ci_config_allows_profile_threshold_override(tmp_path):
    config_path = tmp_path / "alteron_ci.yml"
    config_path.write_text(
        """
version: 1
seed: 42
regression_threshold: -0.05
profiles:
  pr-fast:
    mr_ids: all
    max_records_per_mr: 100
    regression_threshold: -0.10
    fail_on_severity:
      - hard-fail
  release-full:
    mr_ids: all
    max_records_per_mr: null
    fail_on_severity:
      - hard-fail
""",
        encoding="utf-8",
    )

    config = runner.load_ci_config(config_path)

    assert config.regression_threshold == -0.05
    assert config.profiles["pr-fast"].regression_threshold == -0.10
    assert config.profiles["release-full"].regression_threshold is None
    assert config.run == {}


def test_load_ci_config_reads_runtime_defaults(tmp_path):
    config_path = tmp_path / "alteron_ci.yml"
    config_path.write_text(
        """
version: 1
seed: 42
regression_threshold: -0.05
run:
  candidate_model_dir: /tmp/candidate
  candidate_version: v2_retrain
  baseline_snapshot_dir: /tmp/baseline
  baseline_version: v1_base
  corpus_dir: /tmp/corpus
  output_dir: /tmp/output
  model_loader: ./demo_loader.py:load_model_bundle
profiles:
  pr-fast:
    mr_ids: all
    max_records_per_mr: 100
    fail_on_severity:
      - hard-fail
""",
        encoding="utf-8",
    )

    config = runner.load_ci_config(config_path)

    assert config.run["candidate_model_dir"] == "/tmp/candidate"
    assert config.run["candidate_version"] == "v2_retrain"
    assert config.run["baseline_snapshot_dir"] == "/tmp/baseline"
    assert config.run["baseline_version"] == "v1_base"
    assert config.run["corpus_dir"] == "/tmp/corpus"
    assert config.run["output_dir"] == "/tmp/output"
    assert config.run["model_loader"] == "./demo_loader.py:load_model_bundle"


def test_run_ci_check_uses_runtime_defaults_from_config(tmp_path, monkeypatch):
    config_path = tmp_path / "alteron_ci.yml"
    candidate_model_dir = tmp_path / "models" / "candidate"
    candidate_model_dir.mkdir(parents=True)
    baseline_snapshot_dir = tmp_path / "baseline_snapshots"
    corpus_dir = tmp_path / "corpus"
    output_dir = tmp_path / "ci_output"
    config_path.write_text(
        f"""
version: 1
seed: 42
regression_threshold: -0.05
run:
  candidate_model_dir: {candidate_model_dir}
  candidate_version: candidate
  baseline_snapshot_dir: {baseline_snapshot_dir}
  baseline_version: stable
  corpus_dir: {corpus_dir}
  output_dir: {output_dir}
  model_loader: unused:loader
profiles:
  pr-fast:
    mr_ids:
      - CHR-GEN-005
    max_records_per_mr: null
    fail_on_severity:
      - hard-fail
""",
        encoding="utf-8",
    )

    _write_corpus(corpus_dir / "CHR-GEN-005_corpus.csv", [_corpus_record("CHR-GEN-005", "row-1", label=1)])
    _write_manifest(corpus_dir)
    _write_snapshot(
        baseline_snapshot_dir / "CHR-GEN-005_snapshot.csv",
        [_snapshot_record("stable", "CHR-GEN-005", "row-1", label=1, mr_pass=True)],
    )

    class FakeSnapshotEngine:
        def __init__(self, registry_loader=None):
            del registry_loader

        def verify_corpus_hashes(self, corpus_dir: str | Path) -> bool:
            del corpus_dir
            return True

        def run(self, model, tokenizer, model_version: str, corpus_dir: str, output_dir: str) -> None:
            del model, tokenizer, corpus_dir
            _write_snapshot(
                Path(output_dir) / model_version / "CHR-GEN-005_snapshot.csv",
                [_snapshot_record(model_version, "CHR-GEN-005", "row-1", label=1, mr_pass=False)],
            )

    monkeypatch.setattr(runner, "SnapshotEngine", FakeSnapshotEngine)
    monkeypatch.setattr(runner, "load_model_bundle", lambda **kwargs: (object(), object()))

    args = argparse.Namespace(
        config=str(config_path),
        profile="pr-fast",
        candidate_model_dir=None,
        candidate_version=None,
        baseline_snapshot_dir=None,
        baseline_version=None,
        corpus_dir=None,
        output_dir=None,
        model_loader=None,
        regression_threshold=None,
        force=False,
    )

    summary = runner.run_ci_check(args)

    assert summary.exit_code == 1
    assert summary.baseline_version == "stable"
    assert summary.candidate_version == "candidate"
    assert Path(summary.output_dir) == output_dir
