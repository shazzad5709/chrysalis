from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
import logging
from pathlib import Path
import sys
from typing import Any

import yaml

from alteron.config import SEED
from alteron.corpus.generator import CorpusGenerator
from alteron.registry.registry import RegistryLoader
from alteron.snapshot.engine import SnapshotEngine

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alteron",
        description="Alteron behavioral regression testing CLI.",
    )
    subparsers = parser.add_subparsers(dest="command_group", required=True)

    corpus_parser = subparsers.add_parser("corpus", help="Frozen corpus commands.")
    corpus_subparsers = corpus_parser.add_subparsers(dest="command", required=True)
    generate_parser = corpus_subparsers.add_parser(
        "generate",
        help="Generate a frozen MR corpus from labeled source examples.",
        description=(
            "Generate a frozen MR corpus. Source files may be .jsonl, .json, or .csv. "
            "SA/TOPIC rows should contain text or sentence plus label/source_label. "
            "NLI rows should contain premise, hypothesis, and label/source_label."
        ),
    )
    generate_parser.add_argument("--config", help="Optional YAML config file containing a top-level 'corpus' section.")
    generate_parser.add_argument("--mr-ids", nargs="+", help="MR IDs to generate, or 'all'.")
    generate_parser.add_argument("--sa-source", help="Optional SA source file: .jsonl, .json, or .csv.")
    generate_parser.add_argument("--nli-source", help="Optional NLI source file: .jsonl, .json, or .csv.")
    generate_parser.add_argument("--topic-source", help="Optional topic-classification source file: .jsonl, .json, or .csv.")
    generate_parser.add_argument("--output-dir", help="Directory for frozen corpus CSVs and manifest.")
    generate_parser.add_argument("--manual-validation-dir", help="Directory for manual-validation CSVs.")
    generate_parser.add_argument("--seed", type=int, help=f"Random seed. Default: {SEED}.")
    generate_parser.add_argument(
        "--tokenizer-loader",
        help=(
            "Optional import spec that returns a tokenizer, or a (model, tokenizer) pair. "
            "Useful for CHR-GEN-018 casing checks."
        ),
    )
    generate_parser.set_defaults(func=run_corpus_generate)

    snapshot_parser = subparsers.add_parser("snapshot", help="Snapshot commands.")
    snapshot_subparsers = snapshot_parser.add_subparsers(dest="command", required=True)
    create_parser = snapshot_subparsers.add_parser(
        "create",
        help="Create a behavioral snapshot for a model version.",
        description="Create a baseline or candidate behavioral snapshot from an existing frozen corpus.",
    )
    _add_model_snapshot_args(create_parser)
    create_parser.add_argument("--model-version", help="Snapshot version label, e.g. stable or candidate.")
    create_parser.set_defaults(func=run_snapshot_create)

    baseline_parser = snapshot_subparsers.add_parser(
        "baseline",
        help="Create a baseline snapshot for the current stable model.",
        description="Create a baseline behavioral snapshot from an existing frozen corpus.",
    )
    _add_model_snapshot_args(baseline_parser)
    baseline_parser.add_argument(
        "--baseline-version",
        "--model-version",
        dest="model_version",
        help="Baseline snapshot version label, e.g. v1_stable.",
    )
    baseline_parser.set_defaults(func=run_snapshot_baseline)

    return parser


def _add_model_snapshot_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="Optional YAML config file containing a top-level 'snapshot' section.")
    parser.add_argument("--model-loader", help="Import spec for model loader, e.g. package.module:load_model.")
    parser.add_argument("--model-dir", help="Local model directory passed to the model loader.")
    parser.add_argument("--corpus-dir", help="Frozen corpus directory with corpus_manifest.json.")
    parser.add_argument("--output-dir", help="Snapshot root directory.")


def run_corpus_generate(args: argparse.Namespace) -> None:
    config = _load_tool_config(args.config)
    corpus_config = dict(config.get("corpus") or {})
    sa_source = _resolve_config_value(args, corpus_config, "sa_source")
    nli_source = _resolve_config_value(args, corpus_config, "nli_source")
    topic_source = _resolve_config_value(args, corpus_config, "topic_source")
    if not any([sa_source, nli_source, topic_source]):
        raise ValueError("Provide at least one of --sa-source, --nli-source, or --topic-source.")
    mr_ids_value = _resolve_config_value(args, corpus_config, "mr_ids")
    if mr_ids_value is None:
        raise ValueError("Missing required setting 'mr_ids'. Provide it on the command line or under the 'corpus' section of the config.")
    output_dir = _require_config_value(args, corpus_config, "output_dir", section_name="corpus")
    manual_validation_dir = _require_config_value(args, corpus_config, "manual_validation_dir", section_name="corpus")
    tokenizer_loader = _resolve_config_value(args, corpus_config, "tokenizer_loader")
    seed = _resolve_config_value(args, corpus_config, "seed")
    if seed is None:
        seed = SEED

    registry_loader = RegistryLoader()
    mr_ids = _resolve_requested_mr_ids(_normalize_mr_ids(mr_ids_value), registry_loader)
    tokenizer = _load_tokenizer(tokenizer_loader) if tokenizer_loader else None
    generator = CorpusGenerator(
        registry_loader=registry_loader,
        tokenizer=tokenizer,
        manual_validation_dir=manual_validation_dir,
    )
    generator.generate(
        mr_ids=mr_ids,
        sa_source=_load_records(sa_source) if sa_source else [],
        nli_source=_load_records(nli_source) if nli_source else [],
        topic_source=_load_records(topic_source) if topic_source else [],
        output_dir=output_dir,
        seed=int(seed),
    )
    print(f"Frozen corpus written to {output_dir}")
    print(f"Manual validation artifacts written to {manual_validation_dir}")


def run_snapshot_create(args: argparse.Namespace) -> None:
    config = _load_tool_config(args.config)
    snapshot_config = dict(config.get("snapshot") or {})
    _run_snapshot(
        model_loader=_require_config_value(args, snapshot_config, "model_loader", section_name="snapshot"),
        model_dir=Path(_require_config_value(args, snapshot_config, "model_dir", section_name="snapshot")),
        model_version=_require_config_value(args, snapshot_config, "model_version", section_name="snapshot"),
        corpus_dir=_require_config_value(args, snapshot_config, "corpus_dir", section_name="snapshot"),
        output_dir=_require_config_value(args, snapshot_config, "output_dir", section_name="snapshot"),
    )


def run_snapshot_baseline(args: argparse.Namespace) -> None:
    config = _load_tool_config(args.config)
    snapshot_config = dict(config.get("snapshot") or {})
    _run_snapshot(
        model_loader=_require_config_value(args, snapshot_config, "model_loader", section_name="snapshot"),
        model_dir=Path(_require_config_value(args, snapshot_config, "model_dir", section_name="snapshot")),
        model_version=_require_snapshot_model_version(args, snapshot_config),
        corpus_dir=_require_config_value(args, snapshot_config, "corpus_dir", section_name="snapshot"),
        output_dir=_require_config_value(args, snapshot_config, "output_dir", section_name="snapshot"),
    )


def _run_snapshot(model_loader: str, model_dir: Path, model_version: str, corpus_dir: str, output_dir: str) -> None:
    model, tokenizer = _load_model_bundle(
        loader_spec=model_loader,
        model_version=model_version,
        model_dir=model_dir,
    )
    engine = SnapshotEngine()
    engine.run(
        model=model,
        tokenizer=tokenizer,
        model_version=model_version,
        corpus_dir=corpus_dir,
        output_dir=output_dir,
    )
    print(f"Snapshot for {model_version} written under {Path(output_dir) / model_version}")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    args.func(args)


def _resolve_requested_mr_ids(values: list[str], registry_loader: RegistryLoader) -> list[str]:
    if len(values) == 1 and values[0].lower() == "all":
        return [record["mr_id"] for record in registry_loader.load()]
    if len(values) == 1 and "," in values[0]:
        return [value.strip() for value in values[0].split(",") if value.strip()]
    return values


def _normalize_mr_ids(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    raise TypeError("mr_ids must be provided as a string or list of strings.")


def _load_tool_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve_config_value(args: argparse.Namespace, section: dict[str, Any], name: str) -> Any:
    cli_value = getattr(args, name, None)
    if cli_value is not None:
        return cli_value
    return section.get(name)


def _require_config_value(args: argparse.Namespace, config_section: dict[str, Any], name: str, *, section_name: str) -> Any:
    value = _resolve_config_value(args, config_section, name)
    if value is not None:
        return value
    raise ValueError(
        f"Missing required setting {name!r}. Provide it on the command line or under the '{section_name}' section of the config."
    )


def _require_snapshot_model_version(args: argparse.Namespace, config_section: dict[str, Any]) -> Any:
    if getattr(args, "model_version", None) is not None:
        return args.model_version
    if config_section.get("model_version") is not None:
        return config_section["model_version"]
    if config_section.get("baseline_version") is not None:
        return config_section["baseline_version"]
    raise ValueError(
        "Missing required setting 'model_version'. Provide it on the command line or under the 'snapshot' section of the config."
    )


def _load_records(path_value: str) -> list[dict[str, Any]]:
    path = Path(path_value)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            return data["data"]
        raise ValueError(f"JSON source must be a list or contain a list under 'data': {path}")
    if suffix == ".csv":
        with path.open("r", newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    raise ValueError(f"Unsupported source file extension for {path}. Use .jsonl, .json, or .csv.")


def _load_tokenizer(loader_spec: str):
    loaded = _call_loader(loader_spec)
    if isinstance(loaded, dict):
        return loaded["tokenizer"]
    if isinstance(loaded, tuple) and len(loaded) == 2:
        return loaded[1]
    return loaded


def _load_model_bundle(loader_spec: str, model_version: str, model_dir: Path):
    loaded = _call_loader(loader_spec, model_version=model_version, model_dir=model_dir)
    if isinstance(loaded, dict):
        return loaded["model"], loaded["tokenizer"]
    if isinstance(loaded, tuple) and len(loaded) == 2:
        return loaded
    raise TypeError("Model loader must return (model, tokenizer) or {'model': model, 'tokenizer': tokenizer}.")


def _call_loader(loader_spec: str, **kwargs):
    loader = _resolve_import_spec(loader_spec)
    try:
        return loader(**kwargs)
    except TypeError:
        if "model_dir" in kwargs:
            try:
                return loader(kwargs["model_dir"])
            except TypeError:
                pass
        if "model_version" in kwargs:
            try:
                return loader(kwargs["model_version"])
            except TypeError:
                pass
        return loader()


def _resolve_import_spec(spec: str):
    module_name, _, attr_name = spec.rpartition(":")
    if not module_name or not attr_name:
        raise ValueError(f"Invalid import spec: {spec}")

    if module_name.endswith(".py") or module_name.startswith("/"):
        module_path = Path(module_name).resolve()
        module_spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"Unable to load module from path: {module_path}")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)

    return getattr(module, attr_name)


if __name__ == "__main__":
    main(sys.argv[1:])
