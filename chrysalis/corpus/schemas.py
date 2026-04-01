from __future__ import annotations

from dataclasses import dataclass


def _serialize_optional(value: object | None) -> str:
    return "" if value is None else str(value)


def _parse_optional_str(value: str) -> str | None:
    return value if value != "" else None


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


@dataclass(slots=True)
class CorpusRecord:
    mr_id: str
    input_id: str
    subtask: str
    source_text: str
    source_label: int
    followup_text: str
    expected_output_relation: str
    variant: str | None
    skip_reason: str | None

    def to_csv_row(self) -> dict[str, str]:
        return {
            "mr_id": self.mr_id,
            "input_id": self.input_id,
            "subtask": self.subtask,
            "source_text": self.source_text,
            "source_label": str(self.source_label),
            "followup_text": self.followup_text,
            "expected_output_relation": self.expected_output_relation,
            "variant": _serialize_optional(self.variant),
            "skip_reason": _serialize_optional(self.skip_reason),
        }

    @classmethod
    def from_csv_row(cls, row: dict[str, str]) -> "CorpusRecord":
        return cls(
            mr_id=row["mr_id"],
            input_id=row["input_id"],
            subtask=row["subtask"],
            source_text=row["source_text"],
            source_label=int(row["source_label"]),
            followup_text=row["followup_text"],
            expected_output_relation=row["expected_output_relation"],
            variant=_parse_optional_str(row.get("variant", "")),
            skip_reason=_parse_optional_str(row.get("skip_reason", "")),
        )


@dataclass(slots=True)
class SnapshotRecord:
    model_version: str
    mr_id: str
    input_id: str
    variant: str | None
    source_pred_label: int
    source_pred_score: float
    followup_pred_label: int
    followup_pred_score: float
    mr_pass: bool
    fairness_regression: bool
    timestamp: str

    def to_csv_row(self) -> dict[str, str]:
        return {
            "model_version": self.model_version,
            "mr_id": self.mr_id,
            "input_id": self.input_id,
            "variant": _serialize_optional(self.variant),
            "source_pred_label": str(self.source_pred_label),
            "source_pred_score": str(self.source_pred_score),
            "followup_pred_label": str(self.followup_pred_label),
            "followup_pred_score": str(self.followup_pred_score),
            "mr_pass": str(self.mr_pass),
            "fairness_regression": str(self.fairness_regression),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_csv_row(cls, row: dict[str, str]) -> "SnapshotRecord":
        return cls(
            model_version=row["model_version"],
            mr_id=row["mr_id"],
            input_id=row["input_id"],
            variant=_parse_optional_str(row.get("variant", "")),
            source_pred_label=int(row["source_pred_label"]),
            source_pred_score=float(row["source_pred_score"]),
            followup_pred_label=int(row["followup_pred_label"]),
            followup_pred_score=float(row["followup_pred_score"]),
            mr_pass=_parse_bool(row["mr_pass"]),
            fairness_regression=_parse_bool(row["fairness_regression"]),
            timestamp=row["timestamp"],
        )
