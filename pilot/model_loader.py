from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class PilotModelBundle:
    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = Path(model_dir)
        metadata_path = self.model_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata.json in {self.model_dir}")

        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.sa_model = AutoModelForSequenceClassification.from_pretrained(self.model_dir / "sa").to(self.device)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(self.model_dir / "nli").to(self.device)
        self.sa_tokenizer = AutoTokenizer.from_pretrained(self.model_dir / "sa", use_fast=True)
        self.nli_tokenizer = AutoTokenizer.from_pretrained(self.model_dir / "nli", use_fast=True)
        self.max_length = int(self.metadata.get("max_length", 128))

    def predict(self, payload, tokenizer=None, subtask: str | None = None) -> dict[str, float | int]:
        del tokenizer
        if subtask == "NLI":
            return self._predict_nli(payload)
        return self._predict_sa(str(payload))

    def _predict_sa(self, text: str) -> dict[str, float | int]:
        encoded = self.sa_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        self.sa_model.eval()
        with torch.no_grad():
            logits = self.sa_model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1)[0]
        label = int(torch.argmax(probabilities).item())
        score = float(probabilities[label].item())
        return {"label": label, "score": score}

    def _predict_nli(self, payload: dict[str, str]) -> dict[str, float | int]:
        encoded = self.nli_tokenizer(
            payload.get("premise", ""),
            payload.get("hypothesis", ""),
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        self.nli_model.eval()
        with torch.no_grad():
            logits = self.nli_model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1)[0]
        label = int(torch.argmax(probabilities).item())
        score = float(probabilities[label].item())
        return {"label": label, "score": score}


def load_model_bundle(model_version: str | None = None, model_dir: str | Path | None = None) -> tuple[Any, Any]:
    if model_dir is None:
        if model_version is None:
            raise ValueError("Provide either model_dir or model_version.")
        model_dir = Path(__file__).resolve().parents[1] / "pilot" / "models" / model_version
    bundle = PilotModelBundle(model_dir)
    return bundle, None
