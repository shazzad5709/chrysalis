from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification


MODEL_SPECS = {
    "bert": {"model_name": "bert-base-cased", "num_labels": 3},
    "distilbert": {"model_name": "distilbert-base-cased", "num_labels": 4},
}


def _log(message: str) -> None:
    print(message, flush=True)


def _parse_batch_sizes(value: str) -> list[int]:
    sizes = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        sizes.append(int(chunk))
    if not sizes:
        raise ValueError("At least one batch size is required.")
    return sizes


def _build_child_command(args, model_key: str, mode: str, batch_size: int) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--probe-child",
        "--model-key",
        model_key,
        "--mode",
        mode,
        "--batch-size",
        str(batch_size),
        "--max-length",
        str(args.max_length),
        "--steps",
        str(args.steps),
        "--device",
        args.device,
    ]


def _run_probe_subprocess(args, model_key: str, mode: str, batch_size: int) -> tuple[bool, str]:
    cmd = _build_child_command(args, model_key, mode, batch_size)
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = (result.stdout + "\n" + result.stderr).strip()
    return result.returncode == 0, output


def _benchmark_series(args, model_key: str, mode: str, batch_sizes: list[int]) -> tuple[int | None, dict[int, str]]:
    largest_safe: int | None = None
    outputs: dict[int, str] = {}
    for batch_size in batch_sizes:
        _log(f"[benchmark:{model_key}:{mode}] probing batch_size={batch_size}")
        ok, output = _run_probe_subprocess(args, model_key, mode, batch_size)
        outputs[batch_size] = output
        if ok:
            largest_safe = batch_size
            summary = output.splitlines()[-1] if output else "probe_ok"
            _log(f"[benchmark:{model_key}:{mode}] PASS batch_size={batch_size} {summary}")
        else:
            summary = output.splitlines()[-1] if output else "probe_failed"
            _log(f"[benchmark:{model_key}:{mode}] FAIL batch_size={batch_size} {summary}")
            if "CUDA out of memory" in output or "out of memory" in output.lower():
                break
    return largest_safe, outputs


def _recommend(batch_size: int | None, eval_batch_size: int | None) -> str:
    if batch_size is None:
        return "No safe train batch size found."
    if eval_batch_size is None:
        eval_batch_size = batch_size
    return (
        f"--batch-size {batch_size} "
        f"--eval-batch-size {eval_batch_size} "
        f"--gradient-accumulation-steps 2"
    )


def _parent_main(args) -> int:
    if args.device != "cuda":
        _log("This benchmark is intended for CUDA. Pass --device cuda on the training machine.")
        return 1
    if not torch.cuda.is_available():
        _log("CUDA is not available in this environment.")
        return 1

    train_sizes = _parse_batch_sizes(args.train_batch_sizes)
    eval_sizes = _parse_batch_sizes(args.eval_batch_sizes)

    _log(
        f"GPU benchmark starting on {torch.cuda.get_device_name(0)} "
        f"max_length={args.max_length} steps={args.steps}"
    )
    _log(f"Train probe sizes: {train_sizes}")
    _log(f"Eval probe sizes: {eval_sizes}")

    for model_key in args.models:
        if model_key not in MODEL_SPECS:
            _log(f"Unknown model key: {model_key}")
            return 1

        spec = MODEL_SPECS[model_key]
        _log(
            f"\n=== Benchmarking {model_key} ({spec['model_name']}, num_labels={spec['num_labels']}) ==="
        )
        largest_train, _ = _benchmark_series(args, model_key, "train", train_sizes)
        largest_eval, _ = _benchmark_series(args, model_key, "eval", eval_sizes)

        _log(f"[summary:{model_key}] largest safe train batch size: {largest_train}")
        _log(f"[summary:{model_key}] largest safe eval batch size: {largest_eval}")
        _log(f"[summary:{model_key}] suggested pipeline flags: {_recommend(largest_train, largest_eval)}")

    return 0


def _random_batch(batch_size: int, max_length: int, vocab_size: int, num_labels: int, device: str) -> dict[str, torch.Tensor]:
    input_ids = torch.randint(low=100, high=max(vocab_size - 1, 101), size=(batch_size, max_length), device=device)
    attention_mask = torch.ones((batch_size, max_length), dtype=torch.long, device=device)
    labels = torch.randint(low=0, high=num_labels, size=(batch_size,), device=device)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _child_main(args) -> int:
    if args.device != "cuda" or not torch.cuda.is_available():
        print("CUDA unavailable for child probe.", file=sys.stderr, flush=True)
        return 2

    spec = MODEL_SPECS[args.model_key]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.set_grad_enabled(args.mode == "train")

    try:
        config = AutoConfig.from_pretrained(spec["model_name"], num_labels=spec["num_labels"])
        model = AutoModelForSequenceClassification.from_config(config).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4) if args.mode == "train" else None
        scaler = torch.amp.GradScaler("cuda", enabled=True) if args.mode == "train" else None
        model.train(args.mode == "train")

        # Run a small number of realistic train/eval iterations with full sequence length.
        start = time.perf_counter()
        for _ in range(args.steps):
            batch = _random_batch(
                batch_size=args.batch_size,
                max_length=args.max_length,
                vocab_size=getattr(config, "vocab_size", 28996),
                num_labels=spec["num_labels"],
                device=args.device,
            )
            if args.mode == "train":
                assert optimizer is not None
                assert scaler is not None
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    outputs = model(**batch)
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                with torch.no_grad():
                    model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(
            f"probe_ok mode={args.mode} batch_size={args.batch_size} "
            f"peak_allocated_mb={peak_mb:.1f} elapsed_s={elapsed:.2f}",
            flush=True,
        )
        return 0
    except RuntimeError as exc:
        message = str(exc)
        if "out of memory" in message.lower():
            torch.cuda.empty_cache()
            print(f"CUDA out of memory for batch_size={args.batch_size} mode={args.mode}", file=sys.stderr, flush=True)
            return 10
        print(message, file=sys.stderr, flush=True)
        return 11


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark the largest safe CUDA batch size for the Chrysalis pilot.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--models", nargs="+", default=["bert", "distilbert"], choices=sorted(MODEL_SPECS))
    parser.add_argument("--train-batch-sizes", default="4,6,8,10,12,14,16")
    parser.add_argument("--eval-batch-sizes", default="8,12,16,20,24,28,32")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--probe-child", action="store_true")
    parser.add_argument("--model-key", choices=sorted(MODEL_SPECS))
    parser.add_argument("--mode", choices=["train", "eval"])
    parser.add_argument("--batch-size", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.probe_child:
        return _child_main(args)
    return _parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
