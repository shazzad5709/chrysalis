from __future__ import annotations

from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_model_bundle(model_version: str | None = None, model_dir: str | Path | None = None):
    if model_dir is None:
        raise ValueError("model_dir is required")

    model_path = Path(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def load_tokenizer():
    return AutoTokenizer.from_pretrained("bert-base-cased")
