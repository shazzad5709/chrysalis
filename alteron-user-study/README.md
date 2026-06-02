# Alteron User Study Guide

This guide explains what you need to know before running the Alteron user study.

The study bundle keeps only the study-specific scripts, inputs, and outputs. It does not keep a copied copy of the main Alteron tool code.

## What Alteron Is

Alteron checks whether a new NLP model version behaves consistently with a previously accepted version. It does not look only at overall accuracy. It also checks whether the new version keeps making the same kind of decisions when the input is changed in controlled ways.

In this study, you will run a prepared scenario and inspect the artifacts that Alteron produces after the check.

## How Alteron Works in This Study

The study uses four ideas:

1. A **fixed test corpus**: a prepared set of source inputs and transformed follow-up inputs.
2. A **previous accepted snapshot**: recorded predictions from the accepted model version on that fixed test corpus.
3. A **new candidate model**: the version currently being checked.
4. **Regression differencing**: a comparison step that checks whether the new version fails more often than the accepted version on transformed inputs that should preserve the same decision.

In simple terms, Alteron asks:

- Did the new version still get the original source input right?
- When the input was transformed in a controlled way, did the new version still behave as expected?
- Did it get worse than the previous accepted version badly enough to block the CI check?

## What the Two Study Scripts Do

### `setup_study`

This script prepares a local Python environment for the study and installs only the runtime dependency needed by the bundled study scenario.

What it does:

1. asks for your consent before installing anything,
2. installs `uv` if it is not already available,
3. creates a local `.venv`, and
4. installs the study dependency set from `requirements.txt`.

### `run_study`

This script runs the prepared Alteron scenario.

What it does:

1. loads the fixed test corpus from `user-study/study_input/corpus/`,
2. loads the stored snapshot of the previous accepted version from `user-study/study_input/snapshots/stable/`,
3. loads the prepared candidate behavior from `user-study/study_model/candidate/`,
4. runs the CI check, and
5. writes the output artifacts to `user-study/study_output/`.

## What the Terminal Output Means

When you run `run_study`, you will usually see:

- `INFO` lines: progress messages from Alteron
- a JSON block: the CI summary printed to the terminal
- a final shell result: pass or fail

The important part is the CI summary and the exit status:

- Exit code `0`: no blocking regression was found
- Exit code `1`: at least one blocking regression was found

For this bundled study scenario, a blocking result is expected. That does **not** mean you did anything wrong. It means the study is set up so that you can inspect a realistic blocked CI outcome.

## Which Files to Check First

After the run completes, inspect these two files first:

```text
user-study/study_output/regression_reports/regression_report_stable_to_candidate.csv
user-study/study_output/ci_summary.json
```

Use them in this order:

1. **`ci_summary.json`** tells you the overall CI outcome.
2. **`regression_report_stable_to_candidate.csv`** tells you why the run was blocked.

## What the CI Summary Is

The CI summary is the machine-readable run summary. It gives you the top-level outcome of the study run.

The most useful fields are:

- `profile`: which CI profile was used
- `baseline_version`: the previous accepted version
- `candidate_version`: the version under test
- `blocking_regressions`: how many blocking regressions were found
- `reports_written`: which report files were generated
- `exit_code`: the final CI result

How to use it:

- Start here to confirm whether the run passed or failed.
- If `blocking_regressions` is greater than `0`, move to the regression report to see which MR caused the problem.

## What the Regression Report Is

The regression report is the main file for understanding **why** the CI run passed or failed.

It has one row for each metamorphic relation (MR) tested in the run. Each row compares the previous accepted version with the new candidate version for that MR and summarizes three things:

- how source-side accuracy changed,
- how many examples were used for the behavioral comparison, and
- whether the observed change was serious enough to be flagged as a behavioral regression.

### What the Columns Mean

- `transition`: which two versions were compared, for example `stable→candidate`
- `mr_id`: which MR was tested
- `n_total`: how many total test rows were available for that MR
- `source_accuracy_old`: source-input accuracy of the previous accepted version
- `source_accuracy_new`: source-input accuracy of the new candidate version
- `source_accuracy_delta`: how much source-input accuracy changed from old to new
- `n_matched`: how many examples were actually used for the behavioral comparison
- `pass_rate_old`: how often the previous accepted version passed this MR on that comparison set
- `pass_rate_new`: how often the new candidate version passed this MR on that comparison set
- `matched_pass_rate_delta`: how much the MR pass rate changed from old to new
- `behavioral_regression_flag`: whether Alteron flagged this MR as a behavioral regression
- `pipeline_severity`: whether this MR is treated as `hard-fail` or `soft-warning`
- `release_blocked`: whether this row contributes to a blocked CI result

## How to Use the Regression Report

Use the regression report as a short checklist:

1. Look for rows where `release_blocked` is `True`.
2. If none are `True`, the run did not fail because of a blocking MR.
3. For each blocking row, check `mr_id` to see which MR caused the problem.
4. Check `matched_pass_rate_delta` to see whether the new candidate version passed this MR less often than the previous accepted version.
5. Check `source_accuracy_delta` to see whether the new candidate version improved, stayed the same, or got worse on the original source inputs.
6. Read `pipeline_severity` to understand whether the MR is configured to block the CI run or only warn about the issue.

This matters because a model can improve on the original source inputs while still getting worse on transformed inputs that should preserve the same decision. When that happens, Alteron treats it as a behavioral regression.

## What to Remember During the Study

- The study scenario is deterministic.
- A failing CI result can be expected and still be correct.
- The CI summary tells you **what happened overall**.
- The regression report tells you **why it happened**.

## Study File Layout

```text
user-study/
  README.md
  setup_study.sh
  setup_study.ps1
  run_study.sh
  run_study.ps1
  study_ci.yml
  study_loader.py
  study_input/
    corpus/
    snapshots/
  study_model/
    candidate/
  study_output/
```
