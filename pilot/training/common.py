from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

try:
    from datasets import Dataset, concatenate_datasets, load_dataset
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("The 'datasets' package is required for pilot training.") from exc

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

import numpy as np
import pyarrow.ipc as ipc
import torch

ROOT = Path(__file__).resolve().parents[2]
PILOT_ROOT = ROOT / "pilot"
DEFAULT_SST2_CACHE_DIR = PILOT_ROOT / "data" / "sst2"
DEFAULT_SNLI_CACHE_DIR = PILOT_ROOT / "data" / "snli"
DEFAULT_IMDB_CACHE_DIR = PILOT_ROOT / "data" / "imdb"
DEFAULT_MULTINLI_CACHE_DIR = PILOT_ROOT / "data" / "multinli"
DEFAULT_AG_NEWS_CACHE_DIR = PILOT_ROOT / "data" / "ag_news"
FALLBACK_SST2_ARROW = Path(
    "/Users/shazzad/.cache/huggingface/datasets/stanfordnlp___sst2/default/0.0.0/"
    "8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb/sst2-validation.arrow"
)
FALLBACK_SNLI_VALIDATION_ARROW = Path(
    "/Users/shazzad/.cache/huggingface/datasets/snli/plain_text/0.0.0/"
    "cdb5c3d5eed6ead6e5a341c8e56e669bb666725b/snli-validation.arrow"
)

DEVELOPMENT_NOTE = (
    "Reduced development-size training run for local validation on constrained hardware. "
    "Use the full Section 11.1 sizes for the final submission."
)

TRAINING_PROFILES = {
    "sa_sst2": {"subtask": "SA", "model_subdir": "sa", "dataset_name": "SST-2"},
    "sa_imdb": {"subtask": "SA", "model_subdir": "sa", "dataset_name": "IMDb"},
    "nli_snli": {"subtask": "NLI", "model_subdir": "nli", "dataset_name": "SNLI"},
    "nli_multinli": {"subtask": "NLI", "model_subdir": "nli", "dataset_name": "MultiNLI"},
    "topic_agnews": {"subtask": "TOPIC", "model_subdir": "topic", "dataset_name": "AG News"},
}

VERSION_CONFIGS = {
    "v1_base": {
        "model_name": "bert-base-cased",
        "learning_rate": 2e-5,
        "full": {"sa_train_n": 1500, "sa_epochs": 1, "nli_train_n": 2500, "nli_epochs": 1, "topic_train_n": 2500, "topic_epochs": 1},
        "reduced": {"sa_train_n": 300, "sa_epochs": 1, "nli_train_n": 500, "nli_epochs": 1, "topic_train_n": 500, "topic_epochs": 1},
    },
    "v2_retrain": {
        "model_name": "bert-base-cased",
        "learning_rate": 2e-5,
        "full": {"sa_train_n": 8000, "sa_epochs": 3, "nli_train_n": 11500, "nli_epochs": 3, "topic_train_n": 11500, "topic_epochs": 3},
        "reduced": {"sa_train_n": 1000, "sa_epochs": 2, "nli_train_n": 1500, "nli_epochs": 2, "topic_train_n": 1500, "topic_epochs": 2},
    },
    "v3_distilled": {
        "model_name": "distilbert-base-cased",
        "learning_rate": 3e-5,
        "full": {"sa_train_n": 8000, "sa_epochs": 2, "nli_train_n": 11500, "nli_epochs": 2, "topic_train_n": 11500, "topic_epochs": 2},
        "reduced": {"sa_train_n": 1000, "sa_epochs": 1, "nli_train_n": 1500, "nli_epochs": 1, "topic_train_n": 1500, "topic_epochs": 1},
    },
}


def _log(message: str) -> None:
    print(message, flush=True)


class ProgressPrinterCallback(TrainerCallback):
    def __init__(self, *, run_name: str, train_size: int, eval_size: int) -> None:
        self.run_name = run_name
        self.train_size = train_size
        self.eval_size = eval_size

    def on_train_begin(self, args, state, control, **kwargs):
        _log(
            f"[train:{self.run_name}] starting: train_examples={self.train_size} "
            f"eval_examples={self.eval_size} epochs={args.num_train_epochs} max_steps={state.max_steps}"
        )
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return control
        rendered = []
        for key in ("loss", "grad_norm", "learning_rate", "epoch"):
            if key in logs:
                value = logs[key]
                rendered.append(f"{key}={value:.6f}" if isinstance(value, float) else f"{key}={value}")
        if rendered:
            _log(f"[train:{self.run_name}] step={state.global_step}/{state.max_steps} " + " ".join(rendered))
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            rendered = " ".join(
                f"{key}={value:.6f}" if isinstance(value, float) else f"{key}={value}"
                for key, value in sorted(metrics.items())
            )
            _log(f"[train:{self.run_name}] evaluation {rendered}")
        return control

    def on_save(self, args, state, control, **kwargs):
        _log(f"[train:{self.run_name}] checkpoint saved at step={state.global_step}")
        return control

    def on_train_end(self, args, state, control, **kwargs):
        _log(f"[train:{self.run_name}] finished at step={state.global_step}")
        return control


def _load_arrow_rows(path: Path, limit: int | None = None) -> list[dict]:
    with ipc.open_stream(path) as reader:
        rows = reader.read_all().to_pylist()
    return rows[:limit] if limit is not None else rows


def _load_glue_sst2(split: str, limit: int | None = None) -> list[dict]:
    DEFAULT_SST2_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        dataset = load_dataset("glue", "sst2", split=split, cache_dir=str(DEFAULT_SST2_CACHE_DIR))
        if limit is not None:
            dataset = dataset.select(range(min(limit, len(dataset))))
        return list(dataset)
    except Exception:
        if split != "validation" or not FALLBACK_SST2_ARROW.exists():
            raise
        return _load_arrow_rows(FALLBACK_SST2_ARROW, limit=limit)


def _load_imdb(split: str, limit: int | None = None) -> list[dict]:
    DEFAULT_IMDB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("imdb", split=split, cache_dir=str(DEFAULT_IMDB_CACHE_DIR))
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return list(dataset)


def _load_snli(split: str, limit: int | None = None) -> list[dict]:
    DEFAULT_SNLI_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        dataset = load_dataset("snli", split=split, cache_dir=str(DEFAULT_SNLI_CACHE_DIR))
        if limit is not None:
            dataset = dataset.select(range(min(limit, len(dataset))))
        return list(dataset)
    except Exception:
        if split != "validation" or not FALLBACK_SNLI_VALIDATION_ARROW.exists():
            raise
        return _load_arrow_rows(FALLBACK_SNLI_VALIDATION_ARROW, limit=limit)


def _load_multinli(split: str, limit: int | None = None) -> list[dict]:
    DEFAULT_MULTINLI_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("nyu-mll/multi_nli", split=split, cache_dir=str(DEFAULT_MULTINLI_CACHE_DIR))
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return list(dataset)


def _load_ag_news(split: str, limit: int | None = None) -> list[dict]:
    DEFAULT_AG_NEWS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("ag_news", split=split, cache_dir=str(DEFAULT_AG_NEWS_CACHE_DIR))
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return list(dataset)


def _prepare_sst2(train_n: int, seed: int) -> tuple[Dataset, Dataset]:
    train_rows = [row for row in _load_glue_sst2("train") if row.get("label") in {0, 1}]
    validation_rows = [row for row in _load_glue_sst2("validation") if row.get("label") in {0, 1}]
    train_dataset = Dataset.from_list(train_rows).shuffle(seed=seed).select(range(min(train_n, len(train_rows))))
    eval_dataset = Dataset.from_list(validation_rows).shuffle(seed=seed).select(range(min(200, len(validation_rows))))
    return train_dataset, eval_dataset


def _prepare_imdb(train_n: int, seed: int) -> tuple[Dataset, Dataset]:
    train_rows = [row for row in _load_imdb("train") if row.get("label") in {0, 1}]
    test_rows = [row for row in _load_imdb("test") if row.get("label") in {0, 1}]
    train_dataset = Dataset.from_list(train_rows).shuffle(seed=seed).select(range(min(train_n, len(train_rows))))
    eval_dataset = Dataset.from_list(test_rows).shuffle(seed=seed).select(range(min(300, len(test_rows))))
    return train_dataset, eval_dataset


def _prepare_snli(train_n: int, seed: int) -> tuple[Dataset, Dataset]:
    train_rows = [row for row in _load_snli("train") if row.get("label") in {0, 1, 2}]
    validation_rows = [row for row in _load_snli("validation") if row.get("label") in {0, 1, 2}]
    train_dataset = Dataset.from_list(train_rows).shuffle(seed=seed).select(range(min(train_n, len(train_rows))))
    eval_dataset = Dataset.from_list(validation_rows).shuffle(seed=seed).select(range(min(300, len(validation_rows))))
    return train_dataset, eval_dataset


def _prepare_multinli(train_n: int, seed: int) -> tuple[Dataset, Dataset]:
    train_rows = [row for row in _load_multinli("train") if row.get("label") in {0, 1, 2}]
    matched_rows = [row for row in _load_multinli("validation_matched") if row.get("label") in {0, 1, 2}]
    mismatched_rows = [row for row in _load_multinli("validation_mismatched") if row.get("label") in {0, 1, 2}]
    train_dataset = Dataset.from_list(train_rows).shuffle(seed=seed).select(range(min(train_n, len(train_rows))))
    eval_dataset = concatenate_datasets(
        [
            Dataset.from_list(matched_rows).shuffle(seed=seed).select(range(min(200, len(matched_rows)))),
            Dataset.from_list(mismatched_rows).shuffle(seed=seed).select(range(min(200, len(mismatched_rows)))),
        ]
    ).shuffle(seed=seed)
    return train_dataset, eval_dataset


def _prepare_ag_news(train_n: int, seed: int) -> tuple[Dataset, Dataset]:
    train_rows = _load_ag_news("train")
    test_rows = _load_ag_news("test")
    train_dataset = Dataset.from_list(train_rows).shuffle(seed=seed).select(range(min(train_n, len(train_rows))))
    eval_dataset = Dataset.from_list(test_rows).shuffle(seed=seed).select(range(min(400, len(test_rows))))
    return train_dataset, eval_dataset


def _profile_size_config(version: str, profile: str, full_spec: bool) -> tuple[int, int]:
    size_config = VERSION_CONFIGS[version]["full" if full_spec else "reduced"]
    if profile.startswith("sa_"):
        return int(size_config["sa_train_n"]), int(size_config["sa_epochs"])
    if profile.startswith("nli_"):
        return int(size_config["nli_train_n"]), int(size_config["nli_epochs"])
    if profile == "topic_agnews":
        return int(size_config["topic_train_n"]), int(size_config["topic_epochs"])
    raise ValueError(f"Unsupported training profile: {profile}")


def _prepare_profile_dataset(profile: str, train_n: int, seed: int) -> tuple[Dataset, Dataset]:
    if profile == "sa_sst2":
        return _prepare_sst2(train_n, seed)
    if profile == "sa_imdb":
        return _prepare_imdb(train_n, seed)
    if profile == "nli_snli":
        return _prepare_snli(train_n, seed)
    if profile == "nli_multinli":
        return _prepare_multinli(train_n, seed)
    if profile == "topic_agnews":
        return _prepare_ag_news(train_n, seed)
    raise ValueError(f"Unsupported training profile: {profile}")


def _accuracy_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": float((predictions == labels).mean())}


def _device_fp16(device: str) -> bool:
    return device == "cuda" and torch.cuda.is_available()


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _default_num_workers() -> int:
    cpu_count = os.cpu_count() or 2
    return max(2, min(4, cpu_count // 2))


def _tokenize_sa(batch: dict[str, list[str]], tokenizer, max_length: int) -> dict[str, Any]:
    return tokenizer(batch["sentence"], truncation=True, max_length=max_length)


def _tokenize_imdb(batch: dict[str, list[str]], tokenizer, max_length: int) -> dict[str, Any]:
    return tokenizer(batch["text"], truncation=True, max_length=max_length)


def _tokenize_nli(batch: dict[str, list[str]], tokenizer, max_length: int) -> dict[str, Any]:
    return tokenizer(batch["premise"], batch["hypothesis"], truncation=True, max_length=max_length)


def _tokenize_topic(batch: dict[str, list[str]], tokenizer, max_length: int) -> dict[str, Any]:
    return tokenizer(batch["text"], truncation=True, max_length=max_length)


def _tokenize_fn_for_profile(profile: str):
    if profile == "sa_sst2":
        return _tokenize_sa
    if profile == "sa_imdb":
        return _tokenize_imdb
    if profile in {"nli_snli", "nli_multinli"}:
        return _tokenize_nli
    if profile == "topic_agnews":
        return _tokenize_topic
    raise ValueError(f"Unsupported training profile: {profile}")


def _num_labels_for_profile(profile: str) -> int:
    if profile.startswith("sa_"):
        return 2
    if profile.startswith("nli_"):
        return 3
    if profile == "topic_agnews":
        return 4
    raise ValueError(f"Unsupported training profile: {profile}")


def _save_metadata(output_dir: Path, metadata: dict[str, Any]) -> None:
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def _train_task(
    *,
    run_name: str,
    model_name: str,
    num_labels: int,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    tokenize_fn,
    output_dir: Path,
    learning_rate: float,
    epochs: int,
    seed: int,
    batch_size: int,
    eval_batch_size: int,
    gradient_accumulation_steps: int,
    max_length: int,
    device: str,
    num_workers: int,
) -> dict[str, float]:
    _log(
        f"[train:{run_name}] preparing tokenizer/model={model_name} labels={num_labels} "
        f"device={device} batch_size={batch_size} eval_batch_size={eval_batch_size} "
        f"grad_accum={gradient_accumulation_steps} max_length={max_length} num_workers={num_workers}"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    _log(f"[train:{run_name}] tokenizing train_size={len(train_dataset)} eval_size={len(eval_dataset)}")
    tokenized_train = train_dataset.map(lambda batch: tokenize_fn(batch, tokenizer, max_length), batched=True)
    tokenized_eval = eval_dataset.map(lambda batch: tokenize_fn(batch, tokenizer, max_length), batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=25,
        report_to="none",
        load_best_model_at_end=False,
        save_total_limit=1,
        logging_first_step=True,
        seed=seed,
        dataloader_num_workers=num_workers,
        disable_tqdm=True,
        fp16=_device_fp16(device),
        use_cpu=device != "cuda",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
        compute_metrics=_accuracy_metrics,
    )
    trainer.add_callback(ProgressPrinterCallback(run_name=run_name, train_size=len(train_dataset), eval_size=len(eval_dataset)))
    _log(f"[train:{run_name}] entering trainer.train()")
    trainer.train()
    _log(f"[train:{run_name}] running final evaluation")
    metrics = trainer.evaluate()
    _log(f"[train:{run_name}] saving model to {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    _log(f"[train:{run_name}] saved model/tokenizer")
    return {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}


def train_version(
    *,
    version: str,
    profile: str,
    output_dir: str | Path | None = None,
    seed: int = 42,
    device: str = "auto",
    batch_size: int = 16,
    eval_batch_size: int = 32,
    gradient_accumulation_steps: int = 2,
    max_length: int = 128,
    num_workers: int | None = None,
    full_spec: bool = False,
) -> None:
    if version not in VERSION_CONFIGS:
        raise ValueError(f"Unsupported version: {version}")
    if profile not in TRAINING_PROFILES:
        raise ValueError(f"Unsupported training profile: {profile}")

    profile_spec = TRAINING_PROFILES[profile]
    resolved_device = _resolve_device(device)
    resolved_num_workers = _default_num_workers() if num_workers is None else max(0, num_workers)
    version_dir = Path(output_dir) if output_dir is not None else PILOT_ROOT / "models" / profile / version
    version_dir.mkdir(parents=True, exist_ok=True)

    train_n, epochs = _profile_size_config(version, profile, full_spec)
    train_dataset, eval_dataset = _prepare_profile_dataset(profile, train_n, seed)
    model_subdir = profile_spec["model_subdir"]

    _log(
        f"[train:{version}] starting profile={profile} dataset={profile_spec['dataset_name']} "
        f"subtask={profile_spec['subtask']} full_spec={full_spec} device={resolved_device} "
        f"output_dir={version_dir} batch_size={batch_size} eval_batch_size={eval_batch_size} "
        f"grad_accum={gradient_accumulation_steps} num_workers={resolved_num_workers}"
    )
    _log(f"[train:{version}] dataset sizes: train={len(train_dataset)} eval={len(eval_dataset)}")

    metrics = _train_task(
        run_name=f"{version}/{profile}",
        model_name=VERSION_CONFIGS[version]["model_name"],
        num_labels=_num_labels_for_profile(profile),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenize_fn=_tokenize_fn_for_profile(profile),
        output_dir=version_dir / model_subdir,
        learning_rate=VERSION_CONFIGS[version]["learning_rate"],
        epochs=epochs,
        seed=seed,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_length=max_length,
        device=resolved_device,
        num_workers=resolved_num_workers,
    )

    _save_metadata(
        version_dir,
        {
            "version": version,
            "profile": profile,
            "dataset_name": profile_spec["dataset_name"],
            "subtask": profile_spec["subtask"],
            "development_note": None if full_spec else DEVELOPMENT_NOTE,
            "full_spec": full_spec,
            "model_name": VERSION_CONFIGS[version]["model_name"],
            "learning_rate": VERSION_CONFIGS[version]["learning_rate"],
            "weight_decay": 0.01,
            "batch_size": batch_size,
            "eval_batch_size": eval_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "num_workers": resolved_num_workers,
            "max_length": max_length,
            "seed": seed,
            "device": resolved_device,
            "train_n": train_n,
            "epochs": epochs,
            "metrics": metrics,
            "model_dir": model_subdir,
        },
    )
    _log(f"[train:{version}] metadata saved to {version_dir / 'metadata.json'}")
    _log(f"[train:{version}] complete")


def build_training_parser(version: str):
    import argparse

    parser = argparse.ArgumentParser(description=f"Train {version} pilot models.")
    parser.add_argument("--profile", required=True, choices=sorted(TRAINING_PROFILES))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--full-spec", action="store_true", help="Use the full Section 11.1 training sizes.")
    return parser


def run_cli(version: str) -> None:
    parser = build_training_parser(version)
    args = parser.parse_args()
    train_version(
        version=version,
        profile=args.profile,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        num_workers=args.num_workers,
        full_spec=args.full_spec,
    )


if __name__ == "__main__":  # pragma: no cover
    if len(sys.argv) < 2:
        raise SystemExit("Use one of the version-specific training scripts instead.")
