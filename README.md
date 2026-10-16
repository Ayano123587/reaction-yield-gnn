# Reaction Yield GNN

Predict percentage yield from input-only molecular graphs, reaction roles and temperature using PyTorch Geometric. Real ORD ELN experiments compare GINE graph regressors with fingerprint forests. Quantity-aware representations and robust losses are implemented and tested; selection retains the simpler procedure when gains fail the gate.

## Run

Run these commands from this repository’s root.

Python 3.12 and a separate environment are recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,notebooks]"
reaction-gnn fetch
reaction-gnn prepare
python -c "from reaction_gnn.quantities import prepare_quantitative; prepare_quantitative('data/external/az.parquet','data/processed/reactions-quantity-v2.json')"
python -m pytest -q
python train.py --output docs/NEW_QUANTITY_RUN
```

Root `train.py` scores development folds only. Followups use `--study morgan` or `--study loss` with an explicit preceding `--source` selection directory. Use fresh output paths. `evaluation_scripts/evaluate.py` requires acknowledgement of the previously observed ELN holdout and refuses to overwrite an evaluation. After selection, release explicitly with `python evaluation_scripts/evaluate.py --selection-directory docs/NEW_QUANTITY_RUN --acknowledge-observed-eln`. Then use `reaction-gnn predict models/NEW_QUANTITY_RUN/selected.pt YOUR_REACTION.json`. Saved weights and raw/processed records are ignored; model inference requires local weights. On Linux/macOS activate `.venv/bin/activate`.

Legacy `reaction-gnn train`, `optimize` and `transfer-eval` also score already-observed cohorts. They require `--acknowledge-observed-evaluation` and fresh outputs; Python callers must pass `acknowledge_observed=True`. Legacy optimization needs a fresh report directory for its protocol/progress sidecars. Exclusive writes protect artifacts that appear during computation. `make train` uses validation only.

## Notebooks and results

Launch `python -m jupyter lab` from the activated environment.

- [Real ORD inputs and EDA](notebooks/01_real_ord_inputs_and_eda.ipynb)
- [Main graph model and training](notebooks/02_main_graph_model_and_training.ipynb): visible features, GINE pooling/training loop and actual training curves.
- [Evaluation and domain transfer](notebooks/03_evaluation_and_domain_transfer.ipynb)
- [Checkpoint deployment and transfer limits](notebooks/04_checkpoint_deployment_and_transfer_limits.ipynb): actual development replay, input contracts and target-spread diagnostics.

Three development rounds tested quantities, molecular fingerprints and robust graph losses. None met the fixed **0.5-point/two-fold-win** replacement gate. The retained GNN's retrospective ELN MAE is **24.642 percentage points**, versus forest **23.864**. On 2,048 hash-selected independent-source HTE reactions, GNN MAE is **23.591**, but R² is **−18.145**: broad chemistry generalization is not established.

See [Report.md](Report.md), the [experiment ledger](docs/experiments.md), [dataset provenance](docs/dataset-provenance.json) and [ORD license](docs/ORD-LICENSE.txt). The 740-row ELN cohort is small; grouping uses the principal reactant scaffold. The separate HTE source is narrow assay chemistry, with scaffold overlap. Products, measured outcomes, notes and source identifiers never enter predictors. No synthetic benchmark score is claimed. Data, row predictions, weights and secrets stay ignored.

![Recorded aggregate result](docs/figures/reaction_observed_eln_errors.png)
