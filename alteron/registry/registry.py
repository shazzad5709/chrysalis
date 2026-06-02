from __future__ import annotations

from pathlib import Path

import yaml


class RegistryLoader:
    def __init__(self, registry_path: str | Path | None = None) -> None:
        default_path = Path(__file__).with_name("mr_registry.yaml")
        self.registry_path = Path(registry_path) if registry_path is not None else default_path

    def load(self) -> list[dict]:
        with self.registry_path.open("r", encoding="utf-8") as registry_file:
            records = yaml.safe_load(registry_file) or []

        if not isinstance(records, list):
            msg = f"Registry file must contain a list of records: {self.registry_path}"
            raise ValueError(msg)

        return records

    def get_mr(self, mr_id: str) -> dict:
        for record in self.load():
            if record.get("mr_id") == mr_id:
                return record

        msg = f"MR not found in registry: {mr_id}"
        raise KeyError(msg)

    def get_by_subtask(self, subtask: str) -> list[dict]:
        return [
            record
            for record in self.load()
            if subtask in record.get("applicable_subtasks", [])
        ]

    def get_by_severity(self, severity: str) -> list[dict]:
        return [
            record
            for record in self.load()
            if record.get("pipeline_severity") == severity
        ]
