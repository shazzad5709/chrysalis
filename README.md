# Alteron

**A Tool for Behavioral Regression Testing Across NLP Model Versions**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

Alteron is a behavioral regression testing tool for NLP classifiers. It helps detect cases where a new model version preserves or improves aggregate benchmark performance, but still behaves worse on transformed inputs that should preserve the same decision.

The main CLI entry points are:

- `alteron`
- `alteron-ci`

## What Alteron Does

Alteron is designed for version-to-version evaluation of NLP classifiers in continuous integration workflows. Instead of checking only whether a candidate model still performs well on a benchmark, it checks whether the candidate preserves the expected behavior of a previously accepted version on metamorphically transformed inputs.

This is useful when standard metrics such as accuracy are not enough to reveal prediction-level regressions.

## Core Features

- Fixed test corpus generation from labeled source examples and selected metamorphic relations
- Validation checks for generated source and follow-up pairs
- Manual-validation artifact generation for MR review
- Baseline and candidate behavioral snapshot generation
- Matched-subset regression differencing across model versions
- CI profiles for short PR checks and fuller release checks
- Machine-readable CI summaries and MR-level regression reports
- Separate fairness-alert routing for fairness-specific regressions

## How It Works

Alteron uses a three-stage workflow:

1. **Corpus generation**  
   `alteron corpus generate --config alteron.yml` builds a fixed test corpus from labeled source data and selected metamorphic relations.

2. **Baseline snapshot generation**  
   `alteron snapshot baseline --config alteron.yml` evaluates a previously accepted model version on that corpus and stores its behavior as the reference snapshot.

3. **CI regression check**  
   `alteron-ci --config alteron.yml --profile pr-fast` evaluates a new model version on the same corpus, compares the two versions, and reports whether any behavioral regression should block the run.

## Installation

### Prerequisites

- Python 3.10+
- `uv`
- spaCy English model `en_core_web_sm`
- NLTK `words` corpus

### Install the project

```bash
git clone https://github.com/shazzad5709/alteron.git
cd alteron
uv sync --dev
uv run python -m spacy download en_core_web_sm
uv run python -c "import nltk; nltk.download('words')"
```

You can also activate the environment directly if needed:

```bash
source .venv/bin/activate
```

## Quick Start

This is a minimal example of the intended workflow.

Create a single YAML file, for example `alteron.yml`:

```yaml
corpus:
  mr_ids:
    - CHR-SA-001
    - CHR-SA-007
  sa_source: path/to/sa_data.csv
  output_dir: artifacts/corpus
  manual_validation_dir: artifacts/manual_validation

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
```

### 1. Generate a fixed test corpus

```bash
uv run alteron corpus generate --config alteron.yml
```

### 2. Create a baseline snapshot

```bash
uv run alteron snapshot baseline --config alteron.yml
```

### 3. Run the CI regression check

```bash
uv run alteron-ci --config alteron.yml --profile pr-fast
```

At the end of the run, inspect:

- `ci_summary.json`
- `regression_reports/regression_report_<transition>.csv`

## Further Documentation

- [USER_MANUL.md](/Users/shazzad/Desktop/Codespace/chrysalis/USER_MANUL.md) for the step-by-step operational guide
- [API_REFERENCE.md](/Users/shazzad/Desktop/Codespace/chrysalis/API_REFERENCE.md) for the public CLI, config, loader, and output reference
- [alteron.example.yml](/Users/shazzad/Desktop/Codespace/chrysalis/alteron.example.yml) for the canonical config template
- [example_model_loader.py](/Users/shazzad/Desktop/Codespace/chrysalis/example_model_loader.py) for a minimal loader example
- [ARCHITECTURE.md](/Users/shazzad/Desktop/Codespace/chrysalis/ARCHITECTURE.md) for the system architecture and data flow

## Outputs

The two most important outputs are:

### `ci_summary.json`

This is the machine-readable summary of the CI run. It records:

- the selected profile
- the compared model versions
- the selected MR set
- the regression threshold
- the number of blocking regressions
- the report paths
- the final exit code

Check this file first to see whether the run passed or failed.

### `regression_report_<transition>.csv`

This is the main diagnostic artifact for understanding why a run failed.

Each row corresponds to one metamorphic relation and includes:

- `mr_id`
- `n_matched`
- `pass_rate_old`
- `pass_rate_new`
- `matched_pass_rate_delta`
- `behavioral_regression_flag`
- `pipeline_severity`
- `release_blocked`

If a run is blocked, read the report in this order:

1. Find rows where `release_blocked` is `True`
2. Check the `mr_id`
3. Check `matched_pass_rate_delta`
4. Compare `pass_rate_old` and `pass_rate_new`

This tells you which behavioral property regressed and by how much.

## Architecture

The architectural overview is available in [ARCHITECTURE.md](/Users/shazzad/Desktop/Codespace/chrysalis/ARCHITECTURE.md). The original diagram document is also available in [alteron/architecture.pdf](/Users/shazzad/Desktop/Codespace/chrysalis/alteron/architecture.pdf).

Core packages:

| Path | Purpose |
|---|---|
| `alteron/mrs/` | MR implementations and shared MR base classes |
| `alteron/registry/` | MR metadata and registry loading |
| `alteron/corpus/` | Corpus generation, schemas, and validation |
| `alteron/snapshot/` | Snapshot creation and storage |
| `alteron/regression/` | Regression differencing |
| `alteron/ci/` | CI runner and profile handling |
| `tests/` | Unit and integration tests |

## References

Selected references behind Alteron:

- Ribeiro, Wu, Guestrin, and Singh. *Beyond Accuracy: Behavioral Testing of NLP Models with CheckList*. ACL, 2020.
- Cho, Ruberto, and Terragni. *A Catalog of Metamorphic Relations for NLP and LLMs*. arXiv, 2025.
- Cho et al. *LLMORPH: Automated Metamorphic Testing for Large Language Models*. 2025.
- Chen et al. *Metamorphic Testing: A Review of Challenges and Opportunities*. ACM Computing Surveys, 2018.

## Citation

If you use Alteron in research, cite the paper if available. The repository citation below is provisional until final paper metadata is added.

```bibtex
@software{alteron2026,
  title={Alteron: Behavioral Regression Testing Across NLP Model Versions},
  year={2026},
  url={https://github.com/shazzad5709/alteron}
}
```

## License

The source code in this repository is licensed under the MIT License. See [LICENSE](/Users/shazzad/Desktop/Codespace/chrysalis/LICENSE).

Datasets, model artifacts, and other third-party materials may be subject to separate terms and are not necessarily covered by the repository license.

## Contributing

If you want to contribute:

1. Open an issue first.
2. Keep the change scoped and testable.
3. Add or update tests when modifying MR logic, corpus generation, snapshot behavior, or CI behavior.

## Support

For bug reports, usage questions, or artifact issues, open a GitHub issue in this repository.
