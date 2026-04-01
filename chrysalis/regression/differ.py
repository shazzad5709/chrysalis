from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from chrysalis.config import REGRESSION_THRESHOLD
from chrysalis.corpus.schemas import CorpusRecord, SnapshotRecord
from chrysalis.registry.registry import RegistryLoader


def _serialize_optional(value: object | None) -> str:
    return "" if value is None else str(value)


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


@dataclass(slots=True)
class RegressionReport:
    transition: str
    mr_id: str
    n_matched: int
    pass_rate_old: float
    pass_rate_new: float
    matched_pass_rate_delta: float
    behavioral_regression_flag: bool
    pipeline_severity: str
    release_blocked: bool

    def to_csv_row(self) -> dict[str, str]:
        return {
            "transition": self.transition,
            "mr_id": self.mr_id,
            "n_matched": str(self.n_matched),
            "pass_rate_old": str(self.pass_rate_old),
            "pass_rate_new": str(self.pass_rate_new),
            "matched_pass_rate_delta": str(self.matched_pass_rate_delta),
            "behavioral_regression_flag": str(self.behavioral_regression_flag),
            "pipeline_severity": self.pipeline_severity,
            "release_blocked": str(self.release_blocked),
        }

    @classmethod
    def from_csv_row(cls, row: dict[str, str]) -> "RegressionReport":
        return cls(
            transition=row["transition"],
            mr_id=row["mr_id"],
            n_matched=int(row["n_matched"]),
            pass_rate_old=float(row["pass_rate_old"]),
            pass_rate_new=float(row["pass_rate_new"]),
            matched_pass_rate_delta=float(row["matched_pass_rate_delta"]),
            behavioral_regression_flag=_parse_bool(row["behavioral_regression_flag"]),
            pipeline_severity=row["pipeline_severity"],
            release_blocked=_parse_bool(row["release_blocked"]),
        )


REPORT_FIELDNAMES = [
    "transition",
    "mr_id",
    "n_matched",
    "pass_rate_old",
    "pass_rate_new",
    "matched_pass_rate_delta",
    "behavioral_regression_flag",
    "pipeline_severity",
    "release_blocked",
]


class RegressionDiffer:
    def __init__(self, registry_loader: RegistryLoader | None = None) -> None:
        self.registry_loader = registry_loader or RegistryLoader()

    def diff(self, mr_id: str, old_snapshot_path: str, new_snapshot_path: str, ground_truth_path: str) -> RegressionReport:
        old_rows = self._load_snapshot(Path(old_snapshot_path))
        new_rows = self._load_snapshot(Path(new_snapshot_path))
        ground_truth_rows = self._load_ground_truth(Path(ground_truth_path))

        old_by_key = {self._record_key(row.input_id, row.variant): row for row in old_rows}
        new_by_key = {self._record_key(row.input_id, row.variant): row for row in new_rows}
        truth_by_key = {self._record_key(row.input_id, row.variant): row for row in ground_truth_rows}

        matched_keys = []
        for key in sorted(set(old_by_key) & set(new_by_key) & set(truth_by_key)):
            truth = truth_by_key[key]
            old_snapshot = old_by_key[key]
            new_snapshot = new_by_key[key]
            if (
                old_snapshot.source_pred_label == truth.source_label
                and new_snapshot.source_pred_label == truth.source_label
            ):
                matched_keys.append(key)

        n_matched = len(matched_keys)
        old_pass_count = sum(1 for key in matched_keys if old_by_key[key].mr_pass)
        new_pass_count = sum(1 for key in matched_keys if new_by_key[key].mr_pass)
        pass_rate_old = old_pass_count / n_matched if n_matched else 0.0
        pass_rate_new = new_pass_count / n_matched if n_matched else 0.0
        delta = pass_rate_new - pass_rate_old
        behavioral_regression_flag = delta < REGRESSION_THRESHOLD

        registry_record = self.registry_loader.get_mr(mr_id)
        severity = registry_record["pipeline_severity"]
        return RegressionReport(
            transition=f"{Path(old_snapshot_path).parent.name}→{Path(new_snapshot_path).parent.name}",
            mr_id=mr_id,
            n_matched=n_matched,
            pass_rate_old=pass_rate_old,
            pass_rate_new=pass_rate_new,
            matched_pass_rate_delta=delta,
            behavioral_regression_flag=behavioral_regression_flag,
            pipeline_severity=severity,
            release_blocked=severity == "hard-fail" and behavioral_regression_flag,
        )

    def diff_transition(
        self,
        transition: str,
        old_version: str,
        new_version: str,
        snapshot_dir: str,
        corpus_dir: str,
    ) -> list[RegressionReport]:
        snapshot_root = Path(snapshot_dir)
        corpus_root = Path(corpus_dir)
        reports: list[RegressionReport] = []
        for record in self.registry_loader.load():
            mr_id = record["mr_id"]
            report = self.diff(
                mr_id=mr_id,
                old_snapshot_path=str(snapshot_root / old_version / f"{mr_id}_snapshot.csv"),
                new_snapshot_path=str(snapshot_root / new_version / f"{mr_id}_snapshot.csv"),
                ground_truth_path=str(corpus_root / f"{mr_id}_corpus.csv"),
            )
            report.transition = transition
            reports.append(report)
        return reports

    def write_report(self, reports: list[RegressionReport], output_path: str) -> None:
        filtered = [report for report in reports if report.mr_id != "CHR-NLI-005"]
        self._write_reports(filtered, Path(output_path))

    def write_fairness_report(self, reports: list[RegressionReport], output_path: str) -> None:
        filtered = [
            report
            for report in reports
            if report.mr_id == "CHR-NLI-005" and report.behavioral_regression_flag
        ]
        self._write_reports(filtered, Path(output_path))

    @staticmethod
    def _write_reports(reports: list[RegressionReport], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDNAMES)
            writer.writeheader()
            for report in reports:
                writer.writerow(report.to_csv_row())

    @staticmethod
    def _load_snapshot(path: Path) -> list[SnapshotRecord]:
        with path.open("r", newline="", encoding="utf-8") as handle:
            return [SnapshotRecord.from_csv_row(row) for row in csv.DictReader(handle)]

    @staticmethod
    def _load_ground_truth(path: Path) -> list[CorpusRecord]:
        with path.open("r", newline="", encoding="utf-8") as handle:
            return [CorpusRecord.from_csv_row(row) for row in csv.DictReader(handle)]

    @staticmethod
    def _record_key(input_id: str, variant: str | None) -> tuple[str, str]:
        return input_id, _serialize_optional(variant)
