from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
import random
import shutil
from typing import Iterable

import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from chrysalis.config import SEED

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
PILOT_ROOT = ROOT / "pilot"
MODELS_ROOT = PILOT_ROOT / "models"
DEFAULT_SST2_CACHE_DIR = PILOT_ROOT / "data" / "sst2"
DEFAULT_SNLI_CACHE_DIR = PILOT_ROOT / "data" / "snli"
DEFAULT_IMDB_CACHE_DIR = PILOT_ROOT / "data" / "imdb"
DEFAULT_MULTINLI_CACHE_DIR = PILOT_ROOT / "data" / "multinli"
DEFAULT_AG_NEWS_CACHE_DIR = PILOT_ROOT / "data" / "ag_news"
DEFAULT_LOADER_SPEC = "pilot/quantized_model_loader.py:load_model_bundle"

PROFILE_ALIASES = {
    "sa_sst2": "sa_sst2",
    "sa_imdb": "sa_imdb",
    "nli_snli": "nli_snli",
    "nli_multinli": "nli_multinli",
    "topic_agnews": "topic_agnews",
    "gen_sst2": "sa_sst2",
    "gen_imdb": "sa_imdb",
    "gen_snli": "nli_snli",
    "gen_multinli": "nli_multinli",
    "gen_agnews": "topic_agnews",
}

PROFILE_SPECS = {
    "sa_sst2": {
        "dataset_name": "SST-2",
        "subtask": "SA",
        "head": "sa",
        "num_labels": 2,
    },
    "sa_imdb": {
        "dataset_name": "IMDb",
        "subtask": "SA",
        "head": "sa",
        "num_labels": 2,
    },
    "nli_snli": {
        "dataset_name": "SNLI",
        "subtask": "NLI",
        "head": "nli",
        "num_labels": 3,
    },
    "nli_multinli": {
        "dataset_name": "MultiNLI",
        "subtask": "NLI",
        "head": "nli",
        "num_labels": 3,
    },
    "topic_agnews": {
        "dataset_name": "AG News",
        "subtask": "TOPIC",
        "head": "topic",
        "num_labels": 4,
    },
}


def _resolve_profile(profile: str) -> str:
    if profile not in PROFILE_ALIASES:
        raise ValueError(f"Unsupported profile: {profile}")
    return PROFILE_ALIASES[profile]


def _ensure_quant_engine() -> str:
    supported = list(torch.backends.quantized.supported_engines)
    if not supported:
        raise RuntimeError("No quantized backend is available in this PyTorch build.")
    preferred = "qnnpack" if "qnnpack" in supported else "fbgemm" if "fbgemm" in supported else supported[0]
    torch.backends.quantized.engine = preferred
    return preferred


def _load_sst2(split: str) -> list[dict]:
    DEFAULT_SST2_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("glue", "sst2", split=split, cache_dir=str(DEFAULT_SST2_CACHE_DIR))
    return list(dataset)


def _load_imdb(split: str) -> list[dict]:
    DEFAULT_IMDB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("imdb", split=split, cache_dir=str(DEFAULT_IMDB_CACHE_DIR))
    return list(dataset)


def _load_snli(split: str) -> list[dict]:
    DEFAULT_SNLI_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("snli", split=split, cache_dir=str(DEFAULT_SNLI_CACHE_DIR))
    return [row for row in dataset if row.get("label") in {0, 1, 2}]


def _load_multinli(split: str) -> list[dict]:
    DEFAULT_MULTINLI_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("nyu-mll/multi_nli", split=split, cache_dir=str(DEFAULT_MULTINLI_CACHE_DIR))
    return [row for row in dataset if row.get("label") in {0, 1, 2}]


def _load_ag_news(split: str) -> list[dict]:
    DEFAULT_AG_NEWS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("ag_news", split=split, cache_dir=str(DEFAULT_AG_NEWS_CACHE_DIR))
    return list(dataset)


def _load_profile_rows(profile: str, *, split: str) -> list[dict]:
    if profile == "sa_sst2":
        return [row for row in _load_sst2(split) if row.get("label") in {0, 1}]
    if profile == "sa_imdb":
        return [row for row in _load_imdb(split) if row.get("label") in {0, 1}]
    if profile == "nli_snli":
        return _load_snli(split)
    if profile == "nli_multinli":
        if split == "validation":
            return [*_load_multinli("validation_matched"), *_load_multinli("validation_mismatched")]
        return _load_multinli(split)
    if profile == "topic_agnews":
        return _load_ag_news(split)
    raise ValueError(f"Unsupported resolved profile: {profile}")


def _text_for_row(profile: str, row: dict) -> str | tuple[str, str]:
    if profile == "sa_sst2":
        return str(row["sentence"])
    if profile == "sa_imdb":
        return str(row["text"])
    if profile in {"nli_snli", "nli_multinli"}:
        return str(row["premise"]), str(row["hypothesis"])
    if profile == "topic_agnews":
        return str(row["text"])
    raise ValueError(f"Unsupported resolved profile: {profile}")


def _label_for_row(row: dict) -> int:
    return int(row["label"])


def _sample_rows(rows: list[dict], sample_size: int, seed: int) -> list[dict]:
    if sample_size <= 0:
        return []
    rng = random.Random(seed)
    count = min(sample_size, len(rows))
    indices = sorted(rng.sample(range(len(rows)), count))
    return [rows[index] for index in indices]


def _batch_encode(tokenizer, profile: str, rows: list[dict], max_length: int):
    payloads = [_text_for_row(profile, row) for row in rows]
    if profile in {"nli_snli", "nli_multinli"}:
        premises = [payload[0] for payload in payloads]
        hypotheses = [payload[1] for payload in payloads]
        return tokenizer(
            premises,
            hypotheses,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
    texts = [payload for payload in payloads]
    return tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )


def _forward_batches(model, tokenizer, profile: str, rows: list[dict], max_length: int, batch_size: int) -> list[torch.Tensor]:
    logits_batches: list[torch.Tensor] = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        encoded = _batch_encode(tokenizer, profile, batch_rows, max_length)
        with torch.no_grad():
            logits = model(**encoded).logits.detach().cpu()
        logits_batches.append(logits)
    return logits_batches


def _collect_activation_ranges(model, tokenizer, profile: str, rows: list[dict], max_length: int, batch_size: int) -> dict[str, dict[str, float | int]]:
    stats: dict[str, dict[str, float | int]] = {}
    hooks = []

    def _make_hook(name: str):
        def _hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(tensor):
                return
            tensor = tensor.detach().float()
            observed_min = float(tensor.min().item())
            observed_max = float(tensor.max().item())
            entry = stats.setdefault(name, {"min": observed_min, "max": observed_max, "observations": 0})
            entry["min"] = min(float(entry["min"]), observed_min)
            entry["max"] = max(float(entry["max"]), observed_max)
            entry["observations"] = int(entry["observations"]) + int(tensor.numel())

        return _hook

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(_make_hook(name)))
    try:
        _forward_batches(model, tokenizer, profile, rows, max_length, batch_size)
    finally:
        for hook in hooks:
            hook.remove()
    return stats


def _quantize_dynamic(model):
    _ensure_quant_engine()
    model.eval()
    return torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


def _predict_rows(model, tokenizer, profile: str, rows: list[dict], max_length: int, batch_size: int) -> list[dict]:
    outputs: list[dict] = []
    batch_logits = _forward_batches(model, tokenizer, profile, rows, max_length, batch_size)
    offset = 0
    for logits in batch_logits:
        probabilities = torch.softmax(logits, dim=-1)
        labels = torch.argmax(probabilities, dim=-1)
        for index in range(probabilities.shape[0]):
            label = int(labels[index].item())
            score = float(probabilities[index, label].item())
            if math.isnan(score):
                raise ValueError("Quantized model produced NaN probability output.")
            source_row = rows[offset + index]
            text_payload = _text_for_row(profile, source_row)
            outputs.append(
                {
                    "index": offset + index,
                    "gold_label": _label_for_row(source_row),
                    "pred_label": label,
                    "pred_score": score,
                    "text": text_payload if isinstance(text_payload, str) else {"premise": text_payload[0], "hypothesis": text_payload[1]},
                }
            )
        offset += probabilities.shape[0]
    return outputs


def _directory_size_mb(path: Path) -> float:
    total = sum(file.stat().st_size for file in path.rglob("*") if file.is_file())
    return total / (1024 * 1024)


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_readme(path: Path, *, resolved_profile: str, source_version: str, target_version: str, calibration_size: int, seed: int) -> None:
    spec = PROFILE_SPECS[resolved_profile]
    content = "\n".join(
        [
            f"# {target_version}",
            "",
            f"- Source model: {source_version}",
            "- Quantization method: post-training dynamic INT8",
            f"- Dataset: {spec['dataset_name']}",
            f"- Profile: {resolved_profile}",
            f"- Calibration sample size: {calibration_size}",
            f"- Seed: {seed}",
            f"- Loader spec: `{DEFAULT_LOADER_SPEC}`",
            "",
            "This model is intended to be used with the existing snapshot and regression pipeline via the quantized loader.",
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")


def _build_target_metadata(source_metadata: dict, *, resolved_profile: str, target_version: str, source_version: str, calibration_size: int, seed: int, verification_size: int) -> dict:
    updated = dict(source_metadata)
    updated["version"] = target_version
    updated["profile"] = resolved_profile
    updated["source_version"] = source_version
    updated["quantization"] = {
        "method": "post-training dynamic INT8",
        "calibration_sample_size": calibration_size,
        "verification_sample_size": verification_size,
        "seed": seed,
        "loader_spec": DEFAULT_LOADER_SPEC,
        "infer_batch_size": 16,
    }
    return updated


def quantize_profile(
    *,
    requested_profile: str,
    source_version: str,
    target_version: str,
    calibration_size: int,
    verification_size: int,
    seed: int,
    force: bool,
    source_model_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict:
    resolved_profile = _resolve_profile(requested_profile)
    profile_spec = PROFILE_SPECS[resolved_profile]
    source_version_dir = Path(source_model_dir) if source_model_dir is not None else MODELS_ROOT / resolved_profile / source_version
    target_version_dir = Path(output_dir) if output_dir is not None else MODELS_ROOT / resolved_profile / target_version
    source_head_dir = source_version_dir / profile_spec["head"]
    target_head_dir = target_version_dir / profile_spec["head"]

    if not source_version_dir.exists():
        raise FileNotFoundError(f"Source model directory not found: {source_version_dir}")
    if not source_head_dir.exists():
        raise FileNotFoundError(f"Source model head not found: {source_head_dir}")

    if target_version_dir.exists():
        if not force:
            raise FileExistsError(f"Target model directory already exists: {target_version_dir}")
        shutil.rmtree(target_version_dir)

    target_head_dir.mkdir(parents=True, exist_ok=True)
    source_metadata_path = source_version_dir / "metadata.json"
    source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8")) if source_metadata_path.exists() else {}
    max_length = int(source_metadata.get("max_length", 128))

    logger.info(
        "Quantizing profile requested=%s resolved=%s source=%s target=%s",
        requested_profile,
        resolved_profile,
        source_version_dir,
        target_version_dir,
    )

    calibration_rows = _sample_rows(_load_profile_rows(resolved_profile, split="train"), calibration_size, seed)
    verification_rows = _sample_rows(_load_profile_rows(resolved_profile, split="validation" if resolved_profile != "sa_imdb" and resolved_profile != "topic_agnews" else "test"), verification_size, seed + 1)

    tokenizer = AutoTokenizer.from_pretrained(source_head_dir, use_fast=True)
    source_model = AutoModelForSequenceClassification.from_pretrained(source_head_dir)
    source_model.eval()

    logger.info("Running calibration pass on %s examples", len(calibration_rows))
    activation_ranges = _collect_activation_ranges(
        source_model,
        tokenizer,
        resolved_profile,
        calibration_rows,
        max_length,
        batch_size=8,
    )

    logger.info("Applying dynamic INT8 quantization")
    quant_engine = _ensure_quant_engine()
    quantized_model = _quantize_dynamic(source_model)
    quantized_model.eval()

    logger.info("Running verification on %s held-out examples", len(verification_rows))
    verification_outputs = _predict_rows(
        quantized_model,
        tokenizer,
        resolved_profile,
        verification_rows,
        max_length,
        batch_size=8,
    )
    if not verification_outputs:
        raise ValueError("Verification produced no outputs.")
    observed_labels = {output["pred_label"] for output in verification_outputs}
    invalid_labels = [label for label in observed_labels if label < 0 or label >= profile_spec["num_labels"]]
    if invalid_labels:
        raise ValueError(f"Verification produced invalid labels: {invalid_labels}")

    config = AutoConfig.from_pretrained(source_head_dir)
    config.save_pretrained(target_head_dir)
    tokenizer.save_pretrained(target_head_dir)
    torch.save(quantized_model.state_dict(), target_head_dir / "quantized_model.pt")

    quantization_manifest = {
        "source_version": source_version,
        "target_version": target_version,
        "requested_profile": requested_profile,
        "resolved_profile": resolved_profile,
        "dataset_name": profile_spec["dataset_name"],
        "subtask": profile_spec["subtask"],
        "head": profile_spec["head"],
        "quantization_method": "post-training dynamic INT8",
        "calibration_sample_size": len(calibration_rows),
        "verification_sample_size": len(verification_rows),
        "seed": seed,
        "loader_spec": DEFAULT_LOADER_SPEC,
        "quantized_engine": quant_engine,
    }
    _write_json(target_head_dir / "quantization_manifest.json", quantization_manifest)
    _write_json(target_head_dir / "activation_ranges.json", activation_ranges)
    _write_json(target_head_dir / "verification_log.json", {"rows": verification_outputs})

    target_metadata = _build_target_metadata(
        source_metadata,
        resolved_profile=resolved_profile,
        target_version=target_version,
        source_version=source_version,
        calibration_size=len(calibration_rows),
        seed=seed,
        verification_size=len(verification_rows),
    )
    _write_json(target_version_dir / "metadata.json", target_metadata)
    _write_readme(
        target_version_dir / "README.md",
        resolved_profile=resolved_profile,
        source_version=source_version,
        target_version=target_version,
        calibration_size=len(calibration_rows),
        seed=seed,
    )

    before_mb = _directory_size_mb(source_head_dir)
    after_mb = _directory_size_mb(target_head_dir)
    summary = {
        "requested_profile": requested_profile,
        "resolved_profile": resolved_profile,
        "source_model_dir": str(source_version_dir),
        "target_model_dir": str(target_version_dir),
        "source_version": source_version,
        "target_version": target_version,
        "model_size_before_mb": round(before_mb, 3),
        "model_size_after_mb": round(after_mb, 3),
        "loader_spec": DEFAULT_LOADER_SPEC,
        "verification_outputs": verification_outputs,
    }
    _write_json(target_version_dir / "quantization_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Produce a v4 INT8 quantized pilot model from a saved v3 model.")
    parser.add_argument("--profile", required=True, choices=sorted(PROFILE_ALIASES))
    parser.add_argument("--source-version", default="v3_distilled")
    parser.add_argument("--target-version", default="v4_quantized")
    parser.add_argument("--calibration-size", type=int, default=256)
    parser.add_argument("--verification-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--source-model-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args()
    summary = quantize_profile(
        requested_profile=args.profile,
        source_version=args.source_version,
        target_version=args.target_version,
        calibration_size=args.calibration_size,
        verification_size=args.verification_size,
        seed=args.seed,
        force=args.force,
        source_model_dir=args.source_model_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
