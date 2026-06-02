from __future__ import annotations

import csv
from pathlib import Path

import pytest

from alteron.corpus.schemas import CorpusRecord, SnapshotRecord
from alteron.regression.differ import RegressionDiffer, RegressionReport


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _corpus_row(input_id: str, label: int) -> dict[str, str]:
    return CorpusRecord(
        mr_id="CHR-SA-001",
        input_id=input_id,
        subtask="SA",
        source_text=f"source {input_id}",
        source_label=label,
        followup_text=f"followup {input_id}",
        expected_output_relation="label_flip",
        variant=None,
        skip_reason=None,
    ).to_csv_row()


def _snapshot_row(model_version: str, input_id: str, source_label: int, mr_pass: bool) -> dict[str, str]:
    return SnapshotRecord(
        model_version=model_version,
        mr_id="CHR-SA-001",
        input_id=input_id,
        variant=None,
        source_pred_label=source_label,
        source_pred_score=0.9,
        followup_pred_label=1 - source_label,
        followup_pred_score=0.9,
        mr_pass=mr_pass,
        fairness_regression=False,
        timestamp="2026-04-13T00:00:00",
    ).to_csv_row()


def test_diff_reports_source_accuracy_delta_separately_from_mr_delta(tmp_path):
    corpus_path = tmp_path / "corpus" / "CHR-SA-001_corpus.csv"
    old_snapshot_path = tmp_path / "snapshots" / "v1_base" / "CHR-SA-001_snapshot.csv"
    new_snapshot_path = tmp_path / "snapshots" / "v2_retrain" / "CHR-SA-001_snapshot.csv"

    labels = {"i0": 1, "i1": 1, "i2": 0, "i3": 0, "i4": 1}
    _write_rows(corpus_path, [_corpus_row(input_id, label) for input_id, label in labels.items()])

    _write_rows(
        old_snapshot_path,
        [
            _snapshot_row("v1_base", "i0", 1, True),
            _snapshot_row("v1_base", "i1", 1, True),
            _snapshot_row("v1_base", "i2", 1, True),
            _snapshot_row("v1_base", "i3", 0, True),
            _snapshot_row("v1_base", "i4", 1, True),
        ],
    )
    _write_rows(
        new_snapshot_path,
        [
            _snapshot_row("v2_retrain", "i0", 1, False),
            _snapshot_row("v2_retrain", "i1", 1, True),
            _snapshot_row("v2_retrain", "i2", 0, True),
            _snapshot_row("v2_retrain", "i3", 0, True),
            _snapshot_row("v2_retrain", "i4", 1, True),
        ],
    )

    report = RegressionDiffer().diff(
        mr_id="CHR-SA-001",
        old_snapshot_path=str(old_snapshot_path),
        new_snapshot_path=str(new_snapshot_path),
        ground_truth_path=str(corpus_path),
    )

    assert report.n_total == 5
    assert report.source_accuracy_old == pytest.approx(0.8)
    assert report.source_accuracy_new == pytest.approx(1.0)
    assert report.source_accuracy_delta == pytest.approx(0.2)
    assert report.n_matched == 4
    assert report.pass_rate_old == pytest.approx(1.0)
    assert report.pass_rate_new == pytest.approx(0.75)
    assert report.matched_pass_rate_delta == pytest.approx(-0.25)
    assert report.behavioral_regression_flag is True
    assert report.release_blocked is True


def test_regression_report_from_csv_row_accepts_legacy_rows_without_accuracy_fields():
    report = RegressionReport.from_csv_row(
        {
            "transition": "v1_base→v2_retrain",
            "mr_id": "CHR-SA-001",
            "n_matched": "10",
            "pass_rate_old": "1.0",
            "pass_rate_new": "0.9",
            "matched_pass_rate_delta": "-0.1",
            "behavioral_regression_flag": "True",
            "pipeline_severity": "hard-fail",
            "release_blocked": "True",
        }
    )

    assert report.n_total == 0
    assert report.source_accuracy_old == 0.0
    assert report.source_accuracy_new == 0.0
    assert report.source_accuracy_delta == 0.0


def test_diff_uses_configurable_regression_threshold(tmp_path):
    corpus_path = tmp_path / "corpus" / "CHR-SA-001_corpus.csv"
    old_snapshot_path = tmp_path / "snapshots" / "v1_base" / "CHR-SA-001_snapshot.csv"
    new_snapshot_path = tmp_path / "snapshots" / "v2_retrain" / "CHR-SA-001_snapshot.csv"

    labels = {f"i{index}": 1 for index in range(10)}
    _write_rows(corpus_path, [_corpus_row(input_id, label) for input_id, label in labels.items()])
    _write_rows(
        old_snapshot_path,
        [_snapshot_row("v1_base", input_id, 1, True) for input_id in labels],
    )
    _write_rows(
        new_snapshot_path,
        [
            _snapshot_row("v2_retrain", input_id, 1, mr_pass=input_id != "i0")
            for input_id in labels
        ],
    )

    default_report = RegressionDiffer().diff(
        mr_id="CHR-SA-001",
        old_snapshot_path=str(old_snapshot_path),
        new_snapshot_path=str(new_snapshot_path),
        ground_truth_path=str(corpus_path),
    )
    relaxed_report = RegressionDiffer(regression_threshold=-0.2).diff(
        mr_id="CHR-SA-001",
        old_snapshot_path=str(old_snapshot_path),
        new_snapshot_path=str(new_snapshot_path),
        ground_truth_path=str(corpus_path),
    )

    assert default_report.matched_pass_rate_delta == pytest.approx(-0.1)
    assert default_report.behavioral_regression_flag is True
    assert relaxed_report.matched_pass_rate_delta == pytest.approx(-0.1)
    assert relaxed_report.behavioral_regression_flag is False
