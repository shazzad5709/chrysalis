from __future__ import annotations

import json
from pathlib import Path


class StudyModel:
    def __init__(self, model_dir: Path) -> None:
        payload = json.loads((model_dir / "candidate_behavior.json").read_text(encoding="utf-8"))
        self.source_predictions = payload["source_predictions"]
        self.followup_predictions = payload["followup_predictions"]

    def predict(self, payload, tokenizer=None, subtask=None):
        del tokenizer, subtask
        if not isinstance(payload, str):
            raise TypeError("StudyModel only supports single-text payloads.")

        if payload in self.source_predictions:
            return self.source_predictions[payload]
        if payload in self.followup_predictions:
            return self.followup_predictions[payload]

        raise KeyError(f"No prediction configured for payload: {payload!r}")

    def predict_many(self, payloads, tokenizer=None, subtask=None):
        return [self.predict(payload, tokenizer=tokenizer, subtask=subtask) for payload in payloads]


def load_model(model_version=None, model_dir=None):
    del model_version
    if model_dir is None:
        raise ValueError("model_dir is required for the study loader")
    return StudyModel(Path(model_dir)), None
