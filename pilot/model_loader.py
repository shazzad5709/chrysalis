from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


def _resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class PilotModelBundle:
    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = Path(model_dir)
        metadata_path = self.model_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata.json in {self.model_dir}")

        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.device = _resolve_device()

        self.sa_model = AutoModelForSequenceClassification.from_pretrained(self.model_dir / "sa").to(self.device)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(self.model_dir / "nli").to(self.device)
        self.sa_tokenizer = AutoTokenizer.from_pretrained(self.model_dir / "sa", use_fast=True)
        self.nli_tokenizer = AutoTokenizer.from_pretrained(self.model_dir / "nli", use_fast=True)
        self.sa_model.eval()
        self.nli_model.eval()
        topic_dir = self.model_dir / "topic"
        if topic_dir.exists():
            self.topic_model = AutoModelForSequenceClassification.from_pretrained(topic_dir).to(self.device)
            self.topic_tokenizer = AutoTokenizer.from_pretrained(topic_dir, use_fast=True)
            self.topic_model.eval()
        else:
            self.topic_model = None
            self.topic_tokenizer = None
        self.max_length = int(self.metadata.get("max_length", 128))
        if self.device == "mps":
            self.infer_batch_size = 128
        elif self.device == "cuda":
            self.infer_batch_size = 32
        else:
            self.infer_batch_size = 8

    def predict(self, payload, tokenizer=None, subtask: str | None = None) -> dict[str, float | int]:
        del tokenizer
        if subtask == "NLI":
            return self._predict_nli(payload)
        if subtask == "TOPIC":
            return self._predict_topic(str(payload))
        return self._predict_sa(str(payload))

    def predict_many(self, payloads, tokenizer=None, subtask: str | None = None) -> list[dict[str, float | int]]:
        del tokenizer
        if subtask == "NLI":
            return self._predict_many_nli(payloads)
        if subtask == "TOPIC":
            return self._predict_many_topic([str(payload) for payload in payloads])
        return self._predict_many_sa([str(payload) for payload in payloads])

    def _predict_sa(self, text: str) -> dict[str, float | int]:
        return self._predict_many_sa([text])[0]

    def _predict_nli(self, payload: dict[str, str]) -> dict[str, float | int]:
        return self._predict_many_nli([payload])[0]

    def _predict_topic(self, text: str) -> dict[str, float | int]:
        return self._predict_many_topic([text])[0]

    def _predict_many_sa(self, texts: list[str]) -> list[dict[str, float | int]]:
        return self._batched_predict(
            model=self.sa_model,
            tokenizer=self.sa_tokenizer,
            tokenizer_args={"text": texts},
        )

    def _predict_many_nli(self, payloads: list[dict[str, str]]) -> list[dict[str, float | int]]:
        premises = [payload.get("premise", "") for payload in payloads]
        hypotheses = [payload.get("hypothesis", "") for payload in payloads]
        return self._batched_predict(
            model=self.nli_model,
            tokenizer=self.nli_tokenizer,
            tokenizer_args={"text": premises, "text_pair": hypotheses},
        )

    def _predict_many_topic(self, texts: list[str]) -> list[dict[str, float | int]]:
        if self.topic_model is None or self.topic_tokenizer is None:
            raise FileNotFoundError(f"Missing topic model artifacts in {self.model_dir / 'topic'}")
        return self._batched_predict(
            model=self.topic_model,
            tokenizer=self.topic_tokenizer,
            tokenizer_args={"text": texts},
        )

    def _batched_predict(self, *, model, tokenizer, tokenizer_args: dict[str, list[str]]) -> list[dict[str, float | int]]:
        results: list[dict[str, float | int]] = []
        total = len(next(iter(tokenizer_args.values()), []))
        batch_total = max(1, (total + self.infer_batch_size - 1) // self.infer_batch_size)
        for start in range(0, total, self.infer_batch_size):
            end = min(total, start + self.infer_batch_size)
            batch_number = start // self.infer_batch_size + 1
            logger.info(
                "    Inference batch %s/%s size=%s device=%s",
                batch_number,
                batch_total,
                end - start,
                self.device,
            )
            batch_kwargs = {key: value[start:end] for key, value in tokenizer_args.items()}
            encoded = tokenizer(
                batch_kwargs["text"],
                batch_kwargs.get("text_pair"),
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_length,
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                logits = model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1)
            labels = torch.argmax(probabilities, dim=-1)
            for index in range(probabilities.shape[0]):
                label = int(labels[index].item())
                score = float(probabilities[index, label].item())
                results.append({"label": label, "score": score})
        return results


def load_model_bundle(model_version: str | None = None, model_dir: str | Path | None = None) -> tuple[Any, Any]:
    if model_dir is None:
        if model_version is None:
            raise ValueError("Provide either model_dir or model_version.")
        model_dir = Path(__file__).resolve().parents[1] / "pilot" / "models" / model_version
    bundle = PilotModelBundle(model_dir)
    return bundle, None
