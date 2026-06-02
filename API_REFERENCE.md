# Alteron API Reference

## Scope

This document describes the public interfaces that users interact with when running Alteron:

- CLI commands
- YAML configuration structure
- model loader contract
- output schemas

It does not describe internal MR implementation details.

## CLI Commands

### `alteron corpus generate`

Builds a fixed metamorphic test corpus from labeled source data.

```bash
uv run alteron corpus generate --config alteron.yml
```

Important flags:

| Flag | Meaning |
|---|---|
| `--config` | YAML file containing a top-level `corpus` section |
| `--mr-ids` | One or more MR IDs, or `all` |
| `--sa-source` | Sentiment analysis source file |
| `--nli-source` | Natural language inference source file |
| `--topic-source` | Topic-classification source file |
| `--output-dir` | Corpus output directory |
| `--manual-validation-dir` | Manual-validation sample directory |
| `--seed` | Random seed override |
| `--tokenizer-loader` | Optional tokenizer loader for corpus validation |

Notes:

- At least one source file must be provided.
- Supported source formats are `.csv`, `.json`, and `.jsonl`.
- CLI flags override config values.

### `alteron snapshot baseline`

Creates the stored behavioral snapshot for an accepted model version.

```bash
uv run alteron snapshot baseline --config alteron.yml
```

Important flags:

| Flag | Meaning |
|---|---|
| `--config` | YAML file containing a top-level `snapshot` section |
| `--model-loader` | Loader import spec for model and tokenizer |
| `--model-dir` | Local model directory |
| `--model-version` | Version label written into the snapshot |
| `--corpus-dir` | Frozen corpus directory |
| `--output-dir` | Snapshot root directory |

Compatibility note:

- `--baseline-version` is still accepted as an alias for `--model-version`
- old configs using `snapshot.baseline_version` still work, but new configs should use `snapshot.model_version`

### `alteron snapshot create`

Creates a non-baseline snapshot outside the CI flow.

```bash
uv run alteron snapshot create --config alteron.yml --model-version v2_candidate
```

Important flags are the same as `snapshot baseline`, except this command only documents `--model-version`.

### `alteron-ci`

Runs the version-to-version behavioral regression check.

```bash
uv run alteron-ci --config alteron.yml --profile pr-fast
```

Important flags:

| Flag | Meaning |
|---|---|
| `--config` | YAML file containing `run` and `profiles` |
| `--profile` | CI profile name |
| `--candidate-model-dir` | Candidate model directory |
| `--candidate-version` | Candidate version label |
| `--baseline-snapshot-dir` | Baseline snapshot directory |
| `--baseline-version` | Baseline version label |
| `--corpus-dir` | Frozen corpus directory |
| `--output-dir` | CI output directory |
| `--model-loader` | Loader import spec for candidate model |
| `--regression-threshold` | Override configured threshold |
| `--force` | Remove existing candidate snapshot directory before running |

Exit codes:

| Exit code | Meaning |
|---|---|
| `0` | No blocking behavioral regression was found |
| `1` | At least one blocking behavioral regression was found |

## YAML Configuration

The recommended starting point is [alteron.example.yml](/Users/shazzad/Desktop/Codespace/chrysalis/alteron.example.yml).

### Top-level keys

| Key | Used by | Purpose |
|---|---|---|
| `seed` | `alteron-ci` | Default sampling seed |
| `regression_threshold` | `alteron-ci` | Default matched-pass-rate regression threshold |
| `corpus` | `alteron corpus generate` | Corpus generation settings |
| `snapshot` | `alteron snapshot baseline`, `alteron snapshot create` | Snapshot generation settings |
| `run` | `alteron-ci` | Runtime model and path settings |
| `profiles` | `alteron-ci` | CI MR subsets and blocking policy |

### `corpus` section

| Key | Type | Meaning |
|---|---|---|
| `mr_ids` | string or list | MR selection, or `all` |
| `sa_source` | string | Path to sentiment-analysis source data |
| `nli_source` | string | Path to NLI source data |
| `topic_source` | string | Path to topic-classification source data |
| `output_dir` | string | Directory for frozen corpus files |
| `manual_validation_dir` | string | Directory for manual-validation samples |
| `seed` | integer | Optional corpus-generation seed override |
| `tokenizer_loader` | string | Optional tokenizer loader import spec |

### `snapshot` section

| Key | Type | Meaning |
|---|---|---|
| `model_loader` | string | Model loader import spec |
| `model_dir` | string | Local model directory |
| `model_version` | string | Snapshot version label |
| `corpus_dir` | string | Frozen corpus directory |
| `output_dir` | string | Snapshot root directory |

Compatibility:

- `baseline_version` is still accepted as a legacy fallback key in this section

### `run` section

| Key | Type | Meaning |
|---|---|---|
| `candidate_model_dir` | string | Candidate model directory |
| `candidate_version` | string | Candidate version label |
| `baseline_snapshot_dir` | string | Baseline snapshot directory |
| `baseline_version` | string | Baseline version label |
| `corpus_dir` | string | Frozen corpus directory |
| `output_dir` | string | CI output directory |
| `model_loader` | string | Model loader import spec |

### `profiles` section

Each profile is a mapping from a profile name to:

| Key | Type | Meaning |
|---|---|---|
| `mr_ids` | string or list | MR selection, or `all` |
| `max_records_per_mr` | integer or null | Sampling cap per MR |
| `fail_on_severity` | list | Severity values that should block the run |
| `regression_threshold` | float or null | Optional profile-level threshold override |

## Model Loader Contract

### Import spec format

The loader spec uses:

```text
path/to/module.py:function_name
```

or

```text
package.module:function_name
```

### Snapshot and CI loaders

For `snapshot` and `alteron-ci`, the loader must return one of:

- `(model, tokenizer)`
- `{"model": model, "tokenizer": tokenizer}`

Accepted callable shapes include:

```python
def load_model_bundle(model_version, model_dir):
    ...

def load_model_bundle(model_dir):
    ...

def load_model_bundle(model_version):
    ...

def load_model_bundle():
    ...
```

Alteron first tries keyword arguments and then falls back to simpler call shapes.

### Tokenizer loaders

For corpus generation, `--tokenizer-loader` may return:

- a tokenizer
- `(model, tokenizer)`
- `{"tokenizer": tokenizer}`

## Output Schemas

### Corpus record

Source: [alteron/corpus/schemas.py](/Users/shazzad/Desktop/Codespace/chrysalis/alteron/corpus/schemas.py)

Serialized CSV columns:

| Field | Type | Meaning |
|---|---|---|
| `mr_id` | string | Metamorphic relation ID |
| `input_id` | string | Unique source input identifier |
| `subtask` | string | Task family such as `SA`, `NLI`, or topic |
| `source_text` | string | Original source input |
| `source_label` | integer | Ground-truth source label |
| `followup_text` | string | Transformed input |
| `expected_output_relation` | string | Expected MR relation |
| `variant` | string or empty | Optional variant label |
| `skip_reason` | string or empty | Populated only when serialized for skipped items |

### Snapshot record

Source: [alteron/corpus/schemas.py](/Users/shazzad/Desktop/Codespace/chrysalis/alteron/corpus/schemas.py)

Serialized CSV columns:

| Field | Type | Meaning |
|---|---|---|
| `model_version` | string | Snapshot version label |
| `mr_id` | string | Metamorphic relation ID |
| `input_id` | string | Source input identifier |
| `variant` | string or empty | Optional variant label |
| `source_pred_label` | integer | Predicted label for the source input |
| `source_pred_score` | float | Predicted score for the source input |
| `followup_pred_label` | integer | Predicted label for the follow-up input |
| `followup_pred_score` | float | Predicted score for the follow-up input |
| `mr_pass` | boolean | Whether the MR relation passed |
| `fairness_regression` | boolean | Whether the failure is routed as a fairness regression |
| `timestamp` | string | ISO-style timestamp written at snapshot time |

### Regression report row

Source: [alteron/regression/differ.py](/Users/shazzad/Desktop/Codespace/chrysalis/alteron/regression/differ.py)

Serialized CSV columns:

| Field | Type | Meaning |
|---|---|---|
| `transition` | string | Version transition label |
| `mr_id` | string | Metamorphic relation ID |
| `n_total` | integer | Number of comparable records for the MR |
| `source_accuracy_old` | float | Source-side accuracy for the baseline version |
| `source_accuracy_new` | float | Source-side accuracy for the candidate version |
| `source_accuracy_delta` | float | Source-side accuracy difference |
| `n_matched` | integer | Number of matched records used for pass-rate comparison |
| `pass_rate_old` | float | Baseline pass rate on the matched subset |
| `pass_rate_new` | float | Candidate pass rate on the matched subset |
| `matched_pass_rate_delta` | float | Candidate minus baseline matched pass rate |
| `behavioral_regression_flag` | boolean | Whether the MR regressed beyond threshold |
| `pipeline_severity` | string | Severity from the MR registry |
| `release_blocked` | boolean | Whether this row blocks the CI outcome |

### CI summary JSON

Source: [alteron/ci/runner.py](/Users/shazzad/Desktop/Codespace/chrysalis/alteron/ci/runner.py)

JSON fields:

| Field | Type | Meaning |
|---|---|---|
| `profile` | string | Selected CI profile |
| `baseline_version` | string | Baseline version label |
| `candidate_version` | string | Candidate version label |
| `mr_ids` | list | MR IDs used in the run |
| `corpus_dir` | string | Original corpus directory |
| `working_corpus_dir` | string | Sampled working corpus used in the run |
| `baseline_snapshot_dir` | string | Input baseline snapshot directory |
| `candidate_snapshot_dir` | string | Generated candidate snapshot directory |
| `output_dir` | string | CI output root |
| `regression_threshold` | float | Effective threshold used for regression detection |
| `reports_written` | list | Report file paths |
| `blocking_regressions` | integer | Number of blocking regressions |
| `fairness_alerts` | integer | Number of fairness alerts |
| `exit_code` | integer | Final process exit code |

## Severity and Blocking Semantics

`pipeline_severity` is defined in the MR registry. The CI profile chooses which severities should block the run through `fail_on_severity`.

Current behavior:

- `hard-fail` can block the CI run
- `soft-warning` is reported but does not block unless explicitly listed in `fail_on_severity`
- fairness regressions are routed into a separate fairness report

## Stability Notes

The interfaces described in this document are the user-facing surfaces to rely on first:

- CLI commands
- YAML config sections
- output artifact shapes

Internal module structure may change more freely than these interfaces.
