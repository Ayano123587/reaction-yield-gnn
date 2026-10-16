# Reaction yield model

The model estimates percentage yield from input molecular graphs, reaction roles and temperature. It uses three GINE layers with 48 hidden units, role-wise mean pooling, a sigmoid regression head and MSE training. Products, measured outcomes, notes and source identifiers never enter its predictors. The selected configuration is in [selected-model.json](docs/selected-model.json).

## Data and selection

The complete named AstraZeneca ORD ELN source has 750 records. Six ambiguous desired-product yields and four nonfinite temperatures leave 740 accepted reactions. A principal-reactant-scaffold split reserves 130 records for evaluation. All 610 development records enter three grouped validation folds. The final model trains on those 610 records for the validation-selected 47 epochs.

Grouped validation MAE is **23.5668 percentage points**, compared with **24.9940** for the matched fingerprint forest. Replacement requires at least 0.5-point mean improvement and wins on two of three folds. Controlled representation, robust-loss, seed-ensemble and descriptor checks did not meet that gate. [Selection verification](docs/selection-check.json) records the compact integrity check; no held-out score selected a candidate.

## Results

| Cohort | Valid reactions | GNN MAE | Forest MAE | GNN R^2 |
|---|---:|---:|---:|---:|
| ELN scaffold holdout | 130 | 24.6419 | 23.8645 | 0.0270 |
| Complete independent HTE source | 50,668 | 23.6374 | 29.8568 | -17.5100 |

The forest is better on the ELN holdout. The GNN's large negative HTE R^2 means poor transfer despite lower MAE than the forest. The HTE source has 50,688 rows, all processed; 20 fail the fixed unambiguous desired-yield rule. It comprises two substrate pairs and has scaffold overlap with ELN. HTE target spread is about 6.01 points, versus ELN 29.68; the GNN's HTE squared error is roughly 18.51 times the cohort-mean reference. That hindsight reference is not a deployable predictor.

Both evaluations include previously observed records and are retrospective, not blind benchmark evidence. HTE results did not guide development selection. Exact full-source counts, hashes and metrics are in [source coverage](docs/source-coverage.json) and [full HTE results](docs/full-hte/results.json). No synthetic data are used.

## Train and use

Use an isolated environment and prepare the real ORD inputs as described in the README. `python train.py --output models/my-model` reproduces the fixed 47-epoch selected model and forest on development records only. It writes actual training history, configuration hashes and a checkpoint reload check; it never scores the 130-row holdout. Output directories must be fresh.

`reaction-gnn predict models/my-model/selected.pt YOUR_REACTION.json` performs inference. Explicit evaluation uses `python evaluation_scripts/evaluate.py --checkpoint-directory models/my-model --report docs/my-observed-eln.json --acknowledge-observed-eln`; the report must be fresh. This reads already trained models, without refitting.

The retained checkpoint is ignored `models/loss-v4/selected.pt`, SHA256 `a23bf986ffaa53d6003dd6648f98141b697438c790c9bb8ebc60adab50ec2e97`. Its production loader reproduces the public manual example at 28.5084%. The deployment notebook checks 64 real development inputs for save/load, batching, order and unused-metadata invariance. The four executed notebooks show real source EDA, core model/training code, actual curves, evaluation and deployment diagnostics. Figures are in `docs/figures`.

ORD declares CC-BY-SA 4.0; see [license](docs/ORD-LICENSE.txt) and [provenance](docs/dataset-provenance.json). Data, row predictions and weights remain private ignored artifacts. Python 3.12 is recommended; recorded CPU environments and source hashes are preserved with aggregate run metadata. Small heterogeneous ELN coverage and narrow HTE chemistry limit generalization.
