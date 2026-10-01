# MANIS: Multi-Agent Nanofluid Intelligence System (Electrical Conductivity)

Code and data accompanying the manuscript *MANIS: a multi-agent AI framework for predicting nanofluid electrical conductivity* (CHEMOLAB-D-26-01083, revised version).

This repository is provided to the reviewers for the purpose of peer review.

## Overview

MANIS is a sequence of seven agents. Each agent reads the structured file produced by the previous one, performs its computation with fixed code, and (except for data extraction) adds an AI-generated commentary that interprets the computed results. All language model calls use Claude Sonnet 4.5 (`claude-sonnet-4-5`) through the Anthropic API.

| File | Agent | Input | Output |
|---|---|---|---|
| `agents/01_data_extraction.py` | Data Extraction | figure or table images | `nanofluid_ec_data.csv` |
| `agents/02_data_cleaning.py` | Data Cleaning | `nanofluid_ec_data.csv` | `nanofluid_ec_data_clean.csv` |
| `agents/03_theoretical_benchmarking.py` | Theoretical Model Benchmarking | clean data | `ec_model_metrics.csv`, `ec_model_predictions.csv` (Fig. 2) |
| `agents/04_feature_engineering.py` | Feature Engineering | clean data | `nanofluid_ec_features.xlsx` |
| `agents/05a_ml_development_conventional.py` | ML Development, 5 parameters | features file | trained models, test set, metrics |
| `agents/05b_ml_development_shen.py` | ML Development, 9 parameters (Shen et al.) | features file | trained models, test set, metrics |
| `agents/05c_ml_development_newcorr.py` | ML Development, 10 parameters (New Correlation) | features file | trained models, test set, metrics |
| `agents/06_ml_evaluation.py` | ML Evaluation | ten-parameter models and test set | per-model diagnostics (Figs. 4 to 6) |
| `agents/07_physical_validation.py` | Physical Validation | ten-parameter models and features | physical-consistency figures (Figs. 8 to 11) |
| `analysis/repeated_splits.py` | Robustness analysis | features file | 11 repeated splits (Section 3.5, Fig. 7) |
| `figures/` | Figure scripts | saved results | manuscript figures |

## Data

`data/nanofluid_ec_data.csv` is the dataset used for every result in the revised manuscript: 509 records, 22 nanofluid systems, 7 nanoparticle types and 13 base fluids. Its SHA256 fingerprint is recorded in `data/SHA256.txt`:

```
6ba8ece5f49791fd1dee6deb7599a527fc90c49541b7b4b8cd31a010788d2f9b
```

Several agents check this fingerprint (or the content of the derived features file) before running, and stop if the data differ.

## How to run

The agents were written for Google Colab with Google Drive. Paths point to `MyDrive/MANIS_ELECTRICAL/` and, for the revision checks, `MyDrive/MANIS_REVISION/00_inputs_v3/`; adjust them for another environment.

1. Store an Anthropic API key in Colab Secrets as `ANTHROPIC_API_KEY`. No key is included in this repository. Model training and all metrics run without a key; only the AI commentary steps need one.
2. To reproduce the results from the shared dataset, start at agent 02 (agent 01 is needed only to extract new data from images).
3. Run the agents in numerical order. Agents 05a, 05b and 05c use the same train/test split and can run in any order.
4. Run `analysis/repeated_splits.py` for the robustness analysis.

## Settings that determine the results

* **Target:** models are trained on log10(σ_nf); R², MAE, RMSE and MAPE are computed after back-transformation to S/m. The robustness analysis also reports the RMSE of log10(σ_nf).
* **Split:** within each nanofluid system, about 20% of the records are held out at random (412 training and 97 test records on the reference split). The split is at the record level, so every system appears in both sets.
* **Seeds:** model seed 42; reference split seed 42; repeated splits use seeds 42 and 0 to 9, fixed before any result was examined.
* **Hyperparameters:** fixed, not tuned (see `build_models()` in the ML Development agents and Section 2.4 of the manuscript).
* **Software:** Python 3.13.15, scikit-learn 1.6.1, XGBoost 3.4.1 (recorded in `analysis/run_info.json`).

## Changes made during the revision

Each file states its changes in its header. In summary:

* **Agent 01:** `parse_response()` corrected. The original version discarded the final data point of every extraction call. The dataset was not re-extracted; this is disclosed in Section 3.8 of the manuscript.
* **Agents 02 and 03:** input checks added (frozen data fingerprint; cleaned file matches the frozen data; no records removed as duplicates). The calculations are unchanged.
* **Agents 05a to 05c:** the metrics table is saved to file, and a check confirms that the features file matches the verified copy. The New Correlation agent also checks that all three feature sets use identical held-out records.
* **Agent 06:** one axis label corrected. A known limitation of its AI comparison (the Conventional baseline is not found) is stated in its header; it does not affect any reported number or figure.
* **Agent 07:** experimental points are shown as training (filled) or held-out test (open) markers. Candidate systems for the figures are ranked by data coverage and the number of held-out points; no model predictions or errors are used in the ranking.

## Notes for reviewers

* The physical-consistency figures (Figs. 8 to 11) mostly show points that were used in training, as stated in the manuscript.
* The AI commentary steps assess checklists of concerns specified in each agent's prompt; their outputs are advisory and do not change the data, features or models.
