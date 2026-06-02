# Alteron Pilot Setup and Results

This directory documents the pilot evaluation used to exercise Alteron across multiple task families, model versions, and model-update transitions.

The generated pilot artifacts are intentionally not kept in the main repository tree. The pilot was used to validate the tool and summarize its behavior, but the repository keeps only the pilot scripts, training entry points, and this documentation.

## Pilot Goal

The pilot was designed to answer one practical question:

Can Alteron detect behavioral regressions across realistic NLP model lifecycle changes, even when ordinary task accuracy alone does not fully explain the change?

## Pilot Scope

The pilot covered nine dataset profiles across three task families:

| Task Family | Profiles |
|---|---|
| Sentiment analysis | `sa_sst2`, `sa_imdb` |
| Natural language inference | `nli_snli`, `nli_multinli` |
| Generic robustness | `gen_sst2`, `gen_imdb`, `gen_snli`, `gen_multinli`, `gen_agnews` |

The pilot compared four model versions:

| Version | Meaning |
|---|---|
| `v1_base` | accepted baseline model |
| `v2_retrain` | retrained model |
| `v3_distilled` | distilled model |
| `v4_quantized` | post-training quantized model |

This produced three version transitions:

| Transition | Meaning |
|---|---|
| `v1_base -> v2_retrain` | retraining |
| `v2_retrain -> v3_distilled` | distillation |
| `v3_distilled -> v4_quantized` | deployment-oriented quantization |

## Pilot Procedure

The pilot followed the same workflow used by the main tool:

1. prepare model versions for one dataset profile
2. generate a fixed corpus for the selected MRs
3. create behavioral snapshots for each model version
4. compare snapshots across version transitions
5. inspect regression reports and CI-style outcomes

Useful pilot entry points in this directory:

- [pipeline.py](/Users/shazzad/Desktop/Codespace/chrysalis/pilot/pipeline.py)
- [model_loader.py](/Users/shazzad/Desktop/Codespace/chrysalis/pilot/model_loader.py)
- [quantized_model_loader.py](/Users/shazzad/Desktop/Codespace/chrysalis/pilot/quantized_model_loader.py)
- [run_fullscale.sh](/Users/shazzad/Desktop/Codespace/chrysalis/pilot/run_fullscale.sh)
- [run_all_profiles.sh](/Users/shazzad/Desktop/Codespace/chrysalis/pilot/run_all_profiles.sh)
- [run_quantize_v4.sh](/Users/shazzad/Desktop/Codespace/chrysalis/pilot/run_quantize_v4.sh)

Training entry points:

- [training/train_v1.py](/Users/shazzad/Desktop/Codespace/chrysalis/pilot/training/train_v1.py)
- [training/train_v2.py](/Users/shazzad/Desktop/Codespace/chrysalis/pilot/training/train_v2.py)
- [training/train_v3.py](/Users/shazzad/Desktop/Codespace/chrysalis/pilot/training/train_v3.py)

## Summary Results

The pilot evaluation produced the following aggregate outcomes:

| Metric | Count |
|---|---:|
| Dataset profiles | 9 |
| Model versions | 4 |
| Version transitions | 3 |
| Standard regression report rows | 81 |
| Behavioral regressions flagged | 16 |
| Release-blocking regressions | 11 |
| Fairness regression reports | 0 |

### Regressions by Transition

| Transition | Flagged Regressions | Release-Blocking Regressions |
|---|---:|---:|
| `v1_base -> v2_retrain` | 6 | 4 |
| `v2_retrain -> v3_distilled` | 9 | 6 |
| `v3_distilled -> v4_quantized` | 1 | 1 |

### Source Accuracy Context

| Transition | Average Source Accuracy Delta | Min Delta | Max Delta |
|---|---:|---:|---:|
| `v1_base -> v2_retrain` | +0.183 | +0.000 | +0.298 |
| `v2_retrain -> v3_distilled` | -0.027 | -0.106 | +0.000 |
| `v3_distilled -> v4_quantized` | -0.003 | -0.020 | +0.022 |

The most important pilot result is that the `v1_base -> v2_retrain` transition improved source accuracy on average while still producing multiple behavioral regressions and release-blocking outcomes.

### Regressions by MR

| MR | Meaning | Flagged Count |
|---|---|---:|
| `CHR-GEN-018` | Capitalization change | 7 |
| `CHR-GEN-019` | Keyboard typo simulation | 5 |
| `CHR-GEN-005` | Space injection | 4 |

### Regressions by Profile

| Profile | Flagged Count |
|---|---:|
| `gen_multinli` | 6 |
| `gen_sst2` | 4 |
| `gen_imdb` | 3 |
| `gen_snli` | 3 |
| `gen_agnews` | 0 |
| `sa_sst2` | 0 |
| `sa_imdb` | 0 |
| `nli_snli` | 0 |
| `nli_multinli` | 0 |

## Interpretation

The pilot supports the main Alteron claim:

- aggregate source-side accuracy is not enough to characterize version-to-version behavior
- retraining and compression can preserve or even improve task-level performance while still degrading robustness-oriented behavioral checks
- generic robustness MRs were the most sensitive indicators in this pilot

## Additional Notes

- `pilot/test_artifacts/` is kept because tests depend on those fixtures.
- Raw datasets, trained models, and generated pilot artifacts are intentionally not part of the main repository payload.
