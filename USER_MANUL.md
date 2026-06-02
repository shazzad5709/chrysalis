# Alteron User Manual

## Table of Contents

1. [Who This Manual Is For](#who-this-manual-is-for)
2. [Workflow Overview](#workflow-overview)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Required Inputs](#required-inputs)
6. [Configuration File](#configuration-file)
7. [Corpus Generation](#corpus-generation)
8. [Snapshot Generation](#snapshot-generation)
9. [CI Regression Checks](#ci-regression-checks)
10. [Model Loader Contract](#model-loader-contract)
11. [Output Artifacts](#output-artifacts)
12. [Troubleshooting](#troubleshooting)
13. [Best Practices](#best-practices)

---

## Who This Manual Is For

This manual is for users who want to run Alteron on their own NLP classifiers.

It assumes you want to:

- prepare a fixed metamorphic test corpus from labeled source data
- record the behavior of an accepted model version
- compare a candidate model version against that baseline in a CI-style check

The README is the landing page. This document is the operational guide.

For the canonical config template, see [alteron.example.yml](/Users/shazzad/Desktop/Codespace/chrysalis/alteron.example.yml). For the public interface details, see [API_REFERENCE.md](/Users/shazzad/Desktop/Codespace/chrysalis/API_REFERENCE.md).

---

## Workflow Overview

Alteron uses a three-stage workflow:

1. Generate a fixed test corpus with `alteron corpus generate`
2. Create a baseline snapshot with `alteron snapshot baseline`
3. Run a CI regression check with `alteron-ci`

You can optionally create non-baseline snapshots with `alteron snapshot create`, but the main CI flow only requires a baseline snapshot. During a CI run, Alteron generates the candidate snapshot automatically.

---

## Prerequisites

- Python 3.10+
- `uv`
- spaCy English model `en_core_web_sm`
- NLTK `words` corpus

Install the required NLP resources:

```bash
uv run python -m spacy download en_core_web_sm
uv run python -c "import nltk; nltk.download('words')"
```

---

## Installation

```bash
git clone https://github.com/shazzad5709/alteron.git
cd alteron
uv sync --dev
```

You can run commands through `uv run` or activate the environment directly:

```bash
source .venv/bin/activate
```

The main commands are:

```bash
uv run alteron --help
uv run alteron-ci --help
```

---

## Required Inputs

Alteron needs four categories of input.

### 1. Labeled source data

Corpus generation consumes labeled source examples for one or more tasks:

- sentiment analysis via `--sa-source`
- natural language inference via `--nli-source`
- topic classification via `--topic-source`

Supported file formats:

- `.csv`
- `.json`
- `.jsonl`

Expected fields:

- sentiment/topic rows should include `text` or `sentence`, plus `label` or `source_label`
- NLI rows should include `premise`, `hypothesis`, plus `label` or `source_label`

### 2. Selected metamorphic relations

You specify the MR set through `mr_ids`. This can be:

- a list of explicit MR IDs
- the string `all`

### 3. A model loader

Snapshot generation and CI runs require a callable loader that returns a model and tokenizer bundle.

### 4. Model directories

You provide filesystem paths for:

- the accepted baseline model
- the candidate model under evaluation

---

## Configuration File

Alteron supports a single YAML file that can drive the full workflow.

Example `alteron.yml`:

```yaml
seed: 42
regression_threshold: -0.05

corpus:
  mr_ids:
    - CHR-SA-001
    - CHR-GEN-005
    - CHR-GEN-018
  sa_source: path/to/sa_data.csv
  output_dir: artifacts/corpus
  manual_validation_dir: artifacts/manual_validation
  tokenizer_loader: path/to/model_loader.py:load_tokenizer

snapshot:
  model_loader: path/to/model_loader.py:load_model_bundle
  model_dir: path/to/accepted_model
  model_version: v1_base
  corpus_dir: artifacts/corpus
  output_dir: artifacts/snapshots

run:
  candidate_model_dir: path/to/candidate_model
  candidate_version: v2_candidate
  baseline_snapshot_dir: artifacts/snapshots/v1_base
  baseline_version: v1_base
  corpus_dir: artifacts/corpus
  output_dir: artifacts/ci_run
  model_loader: path/to/model_loader.py:load_model_bundle

profiles:
  pr-fast:
    mr_ids:
      - CHR-GEN-005
      - CHR-GEN-018
    max_records_per_mr: 100
    fail_on_severity:
      - hard-fail

  release-full:
    mr_ids: all
    fail_on_severity:
      - hard-fail
```

Section meanings:

- `corpus` configures `alteron corpus generate`
- `snapshot` configures `alteron snapshot baseline` and `alteron snapshot create`
- `run` configures `alteron-ci`
- `profiles` define CI MR subsets and blocking policy

For config-backed commands, explicit CLI flags override YAML values.

---

## Corpus Generation

Generate the fixed test corpus with:

```bash
uv run alteron corpus generate --config alteron.yml
```

You can also override individual settings on the command line:

```bash
uv run alteron corpus generate \
  --config alteron.yml \
  --mr-ids CHR-SA-001 CHR-GEN-005 \
  --sa-source path/to/other_data.csv
```

What this step produces:

- per-MR corpus CSV files
- `corpus_manifest.json`
- manual-validation CSV samples

Important options:

- `--mr-ids`
- `--sa-source`
- `--nli-source`
- `--topic-source`
- `--output-dir`
- `--manual-validation-dir`
- `--seed`
- `--tokenizer-loader`

Use `--tokenizer-loader` when corpus validation depends on tokenizer behavior, especially for casing-sensitive MRs such as `CHR-GEN-018`.

---

## Snapshot Generation

### Baseline snapshot

Create the accepted baseline snapshot with:

```bash
uv run alteron snapshot baseline --config alteron.yml
```

This reads the `snapshot` section and writes files under:

```text
artifacts/snapshots/<model_version>/
```

Example:

```text
artifacts/snapshots/v1_base/
```

### Generic snapshot creation

If you want to create a snapshot outside the CI flow, use:

```bash
uv run alteron snapshot create --config alteron.yml --model-version v2_candidate
```

This is useful for debugging or precomputing snapshots, but it is not required for the standard baseline-versus-candidate CI workflow.

Important options:

- `--config`
- `--model-loader`
- `--model-dir`
- `--model-version`
- `--corpus-dir`
- `--output-dir`

Note:

- the `snapshot` section uses `model_version` for both `snapshot create` and `snapshot baseline`
- `baseline_version` is still accepted in old configs for compatibility, but new configs should use `model_version`

---

## CI Regression Checks

Run the version-to-version check with:

```bash
uv run alteron-ci --config alteron.yml --profile pr-fast
```

Useful variants:

```bash
uv run alteron-ci --config alteron.yml --profile release-full
uv run alteron-ci --config alteron.yml --profile pr-fast --force
```

What this step does:

1. Verifies the frozen corpus manifest
2. Selects the MR subset for the chosen profile
3. Builds a working corpus, including profile sampling if configured
4. Loads the candidate model
5. Creates a candidate snapshot
6. Compares candidate behavior against the baseline snapshot
7. Writes CI summary and regression reports
8. Exits with code `0` or `1`

Important options:

- `--config`
- `--profile`
- `--candidate-model-dir`
- `--candidate-version`
- `--baseline-snapshot-dir`
- `--baseline-version`
- `--corpus-dir`
- `--output-dir`
- `--model-loader`
- `--regression-threshold`
- `--force`

Use `--force` when rerunning the same candidate version into an existing output directory.

---

## Model Loader Contract

The loader spec must resolve to a callable import target such as:

```text
path/to/model_loader.py:load_model_bundle
```

### Required return types

For snapshot and CI commands, the loader must return either:

- `(model, tokenizer)`
- `{"model": model, "tokenizer": tokenizer}`

For corpus generation, `--tokenizer-loader` may return:

- a tokenizer directly
- `(model, tokenizer)` where only the tokenizer is used
- `{"tokenizer": tokenizer}`

### Minimal example

```python
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_model_bundle(model_version: str, model_dir: str | Path):
    model_path = Path(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def load_tokenizer():
    return AutoTokenizer.from_pretrained("bert-base-cased")
```

The loader does not need to use `model_version`, but it should accept it for compatibility with the Alteron CLI.

---

## Output Artifacts

### Corpus outputs

Expected artifacts under the corpus output directory:

- `*_corpus.csv`
- `corpus_manifest.json`

Expected artifacts under the manual validation directory:

- `manual_validation_artifacts_<MR_ID>.csv`

### Snapshot outputs

Snapshots are written under:

```text
artifacts/snapshots/<model_version>/
```

Each directory contains per-MR snapshot CSVs.

### CI outputs

Expected artifacts under the CI output directory:

- `ci_summary.json`
- `working_corpus/`
- `snapshots/<candidate_version>/`
- `regression_reports/regression_report_<transition>.csv`
- `regression_reports/fairness_regression_report_<transition>.csv` when fairness regressions are present

### What to inspect first

If a run fails:

1. Open `ci_summary.json`
2. Check `blocking_regressions`
3. Open `regression_report_<transition>.csv`
4. Find rows where `release_blocked` is `True`
5. Compare `pass_rate_old`, `pass_rate_new`, and `matched_pass_rate_delta`

---

## Troubleshooting

### The CLI says a required setting is missing

Check whether the missing setting belongs under:

- `corpus`
- `snapshot`
- `run`

Also check for key mismatches such as:

- `model_version` in `snapshot`
- `baseline_version` in `run`

### The CI run fails because the candidate model directory does not exist

Make sure `run.candidate_model_dir` points to a real local directory.

### The CI run fails because the baseline snapshot directory does not exist

Run baseline snapshot generation first:

```bash
uv run alteron snapshot baseline --config alteron.yml
```

### The CI run fails on corpus hash verification

The corpus files or manifest changed after generation. Regenerate the corpus or restore the original corpus directory.

### The model loader fails at runtime

Check:

- the import spec is valid
- the target function is importable
- the function returns the expected bundle shape
- the function can read the provided model directory

### `CHR-GEN-018` behaves unexpectedly

Check tokenizer casing behavior. This MR is meaningful only when the tokenizer preserves case distinctions.

---

## Best Practices

- Keep one canonical `alteron.yml` per experiment or CI target.
- Store baseline snapshots in a stable location and do not overwrite accepted versions casually.
- Treat the fixed corpus as immutable once you start comparing versions against it.
- Start with a small CI profile such as `pr-fast`, then use a fuller profile for release checks.
- Keep your model loader simple and deterministic.
- Inspect regression reports, not only exit codes.
- Use explicit output directories so artifacts from different runs do not overwrite each other.
