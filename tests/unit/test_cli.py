from __future__ import annotations

import json
from pathlib import Path

from alteron import cli


def test_load_records_supports_json_jsonl_and_csv(tmp_path):
    json_path = tmp_path / "rows.json"
    jsonl_path = tmp_path / "rows.jsonl"
    csv_path = tmp_path / "rows.csv"

    json_path.write_text(json.dumps([{"text": "good", "label": 1}]), encoding="utf-8")
    jsonl_path.write_text(json.dumps({"text": "bad", "label": 0}) + "\n", encoding="utf-8")
    csv_path.write_text("text,label\nokay,1\n", encoding="utf-8")

    assert cli._load_records(str(json_path)) == [{"text": "good", "label": 1}]
    assert cli._load_records(str(jsonl_path)) == [{"text": "bad", "label": 0}]
    assert cli._load_records(str(csv_path)) == [{"text": "okay", "label": "1"}]


def test_corpus_generate_cli_uses_clear_file_inputs(tmp_path, monkeypatch):
    source_path = tmp_path / "sa.jsonl"
    source_path.write_text(json.dumps({"input_id": "sa-1", "text": "A good movie.", "label": 1}) + "\n", encoding="utf-8")
    captured = {}

    class FakeCorpusGenerator:
        def __init__(self, *, registry_loader, tokenizer, manual_validation_dir):
            captured["registry_loader"] = registry_loader
            captured["tokenizer"] = tokenizer
            captured["manual_validation_dir"] = manual_validation_dir

        def generate(self, *, mr_ids, sa_source, nli_source, topic_source, output_dir, seed):
            captured["generate"] = {
                "mr_ids": mr_ids,
                "sa_source": sa_source,
                "nli_source": nli_source,
                "topic_source": topic_source,
                "output_dir": output_dir,
                "seed": seed,
            }

    monkeypatch.setattr(cli, "CorpusGenerator", FakeCorpusGenerator)

    cli.main(
        [
            "corpus",
            "generate",
            "--mr-ids",
            "CHR-SA-001",
            "--sa-source",
            str(source_path),
            "--output-dir",
            str(tmp_path / "corpus"),
            "--manual-validation-dir",
            str(tmp_path / "manual"),
        ]
    )

    assert captured["manual_validation_dir"] == str(tmp_path / "manual")
    assert captured["generate"]["mr_ids"] == ["CHR-SA-001"]
    assert captured["generate"]["sa_source"] == [{"input_id": "sa-1", "text": "A good movie.", "label": 1}]
    assert captured["generate"]["nli_source"] == []
    assert captured["generate"]["topic_source"] == []
    assert captured["generate"]["output_dir"] == str(tmp_path / "corpus")
    assert captured["generate"]["seed"] == 42


def test_snapshot_create_cli_uses_model_loader_and_paths(tmp_path, monkeypatch):
    captured = {}

    def fake_load_model_bundle(*, loader_spec, model_version, model_dir):
        captured["loader_spec"] = loader_spec
        captured["model_version"] = model_version
        captured["model_dir"] = model_dir
        return object(), object()

    class FakeSnapshotEngine:
        def run(self, *, model, tokenizer, model_version, corpus_dir, output_dir):
            captured["snapshot"] = {
                "model": model,
                "tokenizer": tokenizer,
                "model_version": model_version,
                "corpus_dir": corpus_dir,
                "output_dir": output_dir,
            }

    monkeypatch.setattr(cli, "_load_model_bundle", fake_load_model_bundle)
    monkeypatch.setattr(cli, "SnapshotEngine", FakeSnapshotEngine)

    cli.main(
        [
            "snapshot",
            "create",
            "--model-loader",
            "project.loader:load_model",
            "--model-dir",
            str(tmp_path / "model"),
            "--model-version",
            "stable",
            "--corpus-dir",
            str(tmp_path / "corpus"),
            "--output-dir",
            str(tmp_path / "snapshots"),
        ]
    )

    assert captured["loader_spec"] == "project.loader:load_model"
    assert captured["model_version"] == "stable"
    assert captured["model_dir"] == tmp_path / "model"
    assert captured["snapshot"]["model_version"] == "stable"
    assert captured["snapshot"]["corpus_dir"] == str(tmp_path / "corpus")
    assert captured["snapshot"]["output_dir"] == str(tmp_path / "snapshots")


def test_snapshot_baseline_cli_uses_model_version_alias(tmp_path, monkeypatch):
    captured = {}

    def fake_run_snapshot(*, model_loader, model_dir, model_version, corpus_dir, output_dir):
        captured["run_snapshot"] = {
            "model_loader": model_loader,
            "model_dir": model_dir,
            "model_version": model_version,
            "corpus_dir": corpus_dir,
            "output_dir": output_dir,
        }

    monkeypatch.setattr(cli, "_run_snapshot", fake_run_snapshot)

    cli.main(
        [
            "snapshot",
            "baseline",
            "--model-loader",
            "project.loader:load_model",
            "--model-dir",
            str(tmp_path / "model"),
            "--model-version",
            "v1_stable",
            "--corpus-dir",
            str(tmp_path / "corpus"),
            "--output-dir",
            str(tmp_path / "snapshots"),
        ]
    )

    assert captured["run_snapshot"]["model_loader"] == "project.loader:load_model"
    assert captured["run_snapshot"]["model_dir"] == tmp_path / "model"
    assert captured["run_snapshot"]["model_version"] == "v1_stable"
    assert captured["run_snapshot"]["corpus_dir"] == str(tmp_path / "corpus")
    assert captured["run_snapshot"]["output_dir"] == str(tmp_path / "snapshots")


def test_corpus_generate_cli_reads_values_from_config(tmp_path, monkeypatch):
    source_path = tmp_path / "sa.jsonl"
    source_path.write_text(json.dumps({"input_id": "sa-1", "text": "A good movie.", "label": 1}) + "\n", encoding="utf-8")
    config_path = tmp_path / "alteron.yml"
    config_path.write_text(
        f"""
corpus:
  mr_ids:
    - CHR-SA-001
  sa_source: {source_path}
  output_dir: {tmp_path / "corpus"}
  manual_validation_dir: {tmp_path / "manual"}
  seed: 7
""",
        encoding="utf-8",
    )
    captured = {}

    class FakeCorpusGenerator:
        def __init__(self, *, registry_loader, tokenizer, manual_validation_dir):
            captured["registry_loader"] = registry_loader
            captured["tokenizer"] = tokenizer
            captured["manual_validation_dir"] = manual_validation_dir

        def generate(self, *, mr_ids, sa_source, nli_source, topic_source, output_dir, seed):
            captured["generate"] = {
                "mr_ids": mr_ids,
                "sa_source": sa_source,
                "nli_source": nli_source,
                "topic_source": topic_source,
                "output_dir": output_dir,
                "seed": seed,
            }

    monkeypatch.setattr(cli, "CorpusGenerator", FakeCorpusGenerator)

    cli.main(["corpus", "generate", "--config", str(config_path)])

    assert captured["manual_validation_dir"] == str(tmp_path / "manual")
    assert captured["generate"]["mr_ids"] == ["CHR-SA-001"]
    assert captured["generate"]["sa_source"] == [{"input_id": "sa-1", "text": "A good movie.", "label": 1}]
    assert captured["generate"]["output_dir"] == str(tmp_path / "corpus")
    assert captured["generate"]["seed"] == 7


def test_corpus_generate_cli_prefers_explicit_args_over_config(tmp_path, monkeypatch):
    source_path = tmp_path / "sa.jsonl"
    source_path.write_text(json.dumps({"input_id": "sa-1", "text": "A good movie.", "label": 1}) + "\n", encoding="utf-8")
    alt_source_path = tmp_path / "sa_alt.jsonl"
    alt_source_path.write_text(json.dumps({"input_id": "sa-2", "text": "A bad movie.", "label": 0}) + "\n", encoding="utf-8")
    config_path = tmp_path / "alteron.yml"
    config_path.write_text(
        f"""
corpus:
  mr_ids:
    - CHR-SA-001
  sa_source: {source_path}
  output_dir: {tmp_path / "corpus"}
  manual_validation_dir: {tmp_path / "manual"}
  seed: 7
""",
        encoding="utf-8",
    )
    captured = {}

    class FakeCorpusGenerator:
        def __init__(self, *, registry_loader, tokenizer, manual_validation_dir):
            captured["manual_validation_dir"] = manual_validation_dir

        def generate(self, *, mr_ids, sa_source, nli_source, topic_source, output_dir, seed):
            captured["generate"] = {
                "mr_ids": mr_ids,
                "sa_source": sa_source,
                "output_dir": output_dir,
                "seed": seed,
            }

    monkeypatch.setattr(cli, "CorpusGenerator", FakeCorpusGenerator)

    cli.main(
        [
            "corpus",
            "generate",
            "--config",
            str(config_path),
            "--mr-ids",
            "CHR-SA-007",
            "--sa-source",
            str(alt_source_path),
            "--seed",
            "11",
        ]
    )

    assert captured["generate"]["mr_ids"] == ["CHR-SA-007"]
    assert captured["generate"]["sa_source"] == [{"input_id": "sa-2", "text": "A bad movie.", "label": 0}]
    assert captured["generate"]["seed"] == 11


def test_snapshot_baseline_cli_reads_values_from_config(tmp_path, monkeypatch):
    config_path = tmp_path / "alteron.yml"
    config_path.write_text(
        f"""
snapshot:
  model_loader: project.loader:load_model
  model_dir: {tmp_path / "model"}
  model_version: v1_stable
  corpus_dir: {tmp_path / "corpus"}
  output_dir: {tmp_path / "snapshots"}
""",
        encoding="utf-8",
    )
    captured = {}

    def fake_run_snapshot(*, model_loader, model_dir, model_version, corpus_dir, output_dir):
        captured["run_snapshot"] = {
            "model_loader": model_loader,
            "model_dir": model_dir,
            "model_version": model_version,
            "corpus_dir": corpus_dir,
            "output_dir": output_dir,
        }

    monkeypatch.setattr(cli, "_run_snapshot", fake_run_snapshot)

    cli.main(["snapshot", "baseline", "--config", str(config_path)])

    assert captured["run_snapshot"]["model_loader"] == "project.loader:load_model"
    assert captured["run_snapshot"]["model_dir"] == tmp_path / "model"
    assert captured["run_snapshot"]["model_version"] == "v1_stable"
    assert captured["run_snapshot"]["corpus_dir"] == str(tmp_path / "corpus")
    assert captured["run_snapshot"]["output_dir"] == str(tmp_path / "snapshots")


def test_snapshot_baseline_cli_still_accepts_legacy_baseline_version_in_config(tmp_path, monkeypatch):
    config_path = tmp_path / "alteron.yml"
    config_path.write_text(
        f"""
snapshot:
  model_loader: project.loader:load_model
  model_dir: {tmp_path / "model"}
  baseline_version: v1_legacy
  corpus_dir: {tmp_path / "corpus"}
  output_dir: {tmp_path / "snapshots"}
""",
        encoding="utf-8",
    )
    captured = {}

    def fake_run_snapshot(*, model_loader, model_dir, model_version, corpus_dir, output_dir):
        captured["run_snapshot"] = {
            "model_loader": model_loader,
            "model_dir": model_dir,
            "model_version": model_version,
            "corpus_dir": corpus_dir,
            "output_dir": output_dir,
        }

    monkeypatch.setattr(cli, "_run_snapshot", fake_run_snapshot)

    cli.main(["snapshot", "baseline", "--config", str(config_path)])

    assert captured["run_snapshot"]["model_version"] == "v1_legacy"
