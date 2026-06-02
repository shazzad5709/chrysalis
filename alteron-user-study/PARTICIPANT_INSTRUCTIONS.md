# Alteron User Study - Participant Instructions

Thank you for participating in this study.

This study evaluates the usability and interpretability of **Alteron**, a behavioral regression testing tool for NLP classifiers. You will run a prepared Alteron study scenario built from a real pilot slice, inspect the generated outputs, and answer questions in a Google Form.

**Expected time:** 20-30 minutes

## 1. What Alteron Is

Alteron is a tool for checking whether a new NLP model version preserves expected behavior when compared with an accepted baseline model. In addition to ordinary task accuracy, Alteron evaluates how model behavior changes under controlled input transformations.

Alteron is intended for use in model maintenance and CI/CD workflows, where a team needs to decide whether a candidate model should be released.

## 2. Key Terms

**Behavioral regression.** A behavioral regression occurs when a candidate model behaves worse than the baseline under one or more metamorphic tests, even if ordinary task accuracy does not decrease.

**Accuracy up but behavior regress.** A model can improve standard accuracy on original inputs while still becoming less robust or less behaviorally consistent under transformed inputs. Alteron is designed to detect this kind of change.

**Fixed test corpus.** A fixed test corpus is a fixed set of validated source and follow-up test pairs reused across model versions so that changes in results come from model differences rather than test-set differences.

**Snapshot.** A snapshot is the recorded behavior of a model on the frozen corpus, including predictions, confidence scores, and MR pass/fail outcomes.

**Regression report.** A regression report summarizes the comparison between the baseline and candidate for each MR. It includes the MR identifier, the change in source accuracy, the change in MR pass rate on the matched subset, whether a behavioral regression was flagged, and whether the result blocks release.

## 3. What You Will Do

You will:

1. run Alteron on a prepared candidate model,
2. inspect the generated output artifacts,
3. determine whether the candidate would pass or fail the regression gate,
4. identify the cause of the result, and
5. complete a Google Form based on what you observe.

You do **not** need to train a model, modify code, or understand the entire repository.

## 4. What You Need

Please make sure you have:

- the provided Alteron study bundle,
- a terminal,
- internet access during setup,
- the Google Form link.

If the provided environment does not run correctly, please note the issue in the form and continue with the rest of the questions as best you can.

## 5. Setup

1. Extract the provided study bundle.
2. Open a terminal in the root directory of the extracted bundle.
3. Run the setup command for your platform:

On macOS/Linux:
```bash
./user-study/setup_study.sh
```

On Windows PowerShell:
```powershell
.\user-study\setup_study.ps1
```

4. Do not modify any files.

This command creates a local virtual environment and installs the required dependencies.
It uses `uv` to provision a compatible Python environment for the study. If `uv` is not already installed, the script will attempt to install it automatically. The script now asks for your consent before installing anything.

## 6. Run the Study

Run the command for your platform:

On macOS/Linux:
```bash
./user-study/run_study.sh
```

On Windows PowerShell:
```powershell
.\user-study\run_study.ps1
```

This executes the prepared Alteron study scenario.

## 7. Expected Behavior

The study scenario is deterministic. It uses:

- a pilot-derived frozen corpus slice,
- a prebuilt stable snapshot, and
- a deterministic replay of candidate behavior from the retrained pilot model.

The command is expected to produce a CI result and generate output files under:

```text
user-study/study_output/
```

If the command indicates a failing result, that is part of the prepared study scenario and does **not** mean you made a mistake.

For a step-by-step explanation of the workflow, the study scripts, the output files, and how to interpret the regression report, see:

```text
user-study/README.md
```

## 8. Files to Inspect

After the run completes, please inspect these two files first:

```text
user-study/study_output/regression_reports/regression_report_stable_to_candidate.csv
user-study/study_output/ci_summary.json
```

These are the primary files you will need to answer the study questions.

You may inspect other files in `user-study/study_output/` if helpful, but the two files above should be enough for most tasks.

## 9. Complete the Google Form

After inspecting the outputs, complete the provided Google Form.

Please answer based on:

- what you observed while running the study,
- the contents of the generated output files,
- your interpretation of the tool's behavior.

## 10. Important Notes

- Do **not** train models.
- Do **not** regenerate the corpus.
- Do **not** edit the repository.
- Do **not** spend time debugging the internals unless the command cannot run at all.
- If something goes wrong, record the issue briefly in the Google Form.

## 11. What We Are Evaluating

We are **not** evaluating your performance. We are evaluating:

- whether the workflow is understandable,
- whether the generated outputs are easy to interpret,
- whether the artifacts are easy to navigate, and
- whether Alteron seems useful in a model maintenance or CI/CD setting.

## 12. Google Form Link

Please complete the form here:

**https://forms.gle/kuDqM63jWrsrBV9NA**

## 13. Thank You

Thank you for your time and feedback.
