# Reaction yield: results and reproduction

The retained role-aware GINE model was trained on real AstraZeneca ORD ELN inputs. Three resumed development rounds compared quantities/component pooling, larger molecular fingerprints and robust graph losses. None met the fixed replacement gate of at least 0.5-point mean MAE improvement plus two of three fold wins. The gate was not relaxed after seeing small gains.

## Development results

| Procedure | Mean validation MAE (percentage points) |
|---|---:|
| Matched legacy GNN, MSE | 23.566776 |
| Quantity atom-role GNN, MSE | 23.872097 |
| Quantity component-role GNN, MSE | 23.585006 |
| Quantity component-role GNN, Huber 0.1 | 23.342044 |
| Legacy GNN, Huber 0.1 / 0.01 | 23.989624 / 24.066958 |
| Quantity component GNN, Huber 0.01 | 23.718275 |
| Old RDK256 forest, leaf 8 | 24.993998 |
| Morgan1024 forest, best | 24.882222 |

The original 750-row raw file yields 740 accepted records. Six ambiguous desired-product targets and four nonfinite-temperature records remain excluded. Quantity preparation proved all accepted labels, IDs, scaffold groups and structural signatures unchanged. The same 610 development rows and three principal-reactant-scaffold folds were used throughout. Only reported pre-reaction inputs enter models; unknown quantity units remain missing. The concentration field is a proxy, not a reconstructed procedure.

## Released evidence

Selection was frozen at commit `15cf75a` before final evaluation. The 130-row ELN test was already observed in prior work, so this release is retrospective. The retained GNN refit for 47 epochs scored MAE **24.641868**, RMSE **29.273375**, R² **0.027048**. The forest remained better: MAE **23.864459**, RMSE **28.987835**, R² **0.045936**. The original 43-epoch result (MAE 25.2029, R² −0.0244) remains in historical aggregate JSON; it was not overwritten.

The independent-source HTE check retained all 2,048 hash-selected candidate identifiers, with no exact ELN structural/temperature clones. Frozen ELN-trained GNN MAE was **23.591075**, forest **29.885383**, and ELN-development median **34.945738**. GNN R² was **−18.144531** and forest **−26.735837**. One of two principal scaffolds overlapped ELN. This assay source comprises [two documented substrate pairs](https://pubs.acs.org/doi/10.1021/jacs.6c05959) and differs in measurement domain; lower MAE than the transferred forest does not establish broad or useful generalization. No adaptation or tuning followed these outcomes.

## Reproduce and verify

The quantity, Morgan and loss plans were committed before their scores (`28fa125`, `81fb38f`, `6d390cf`). Run the root training entrypoint with fresh output directories, then followups with explicit preceding selections. Final ELN evaluation requires `--acknowledge-observed-eln`. Independent-source evaluation verifies frozen data, checkpoint and selection hashes and refuses an existing report. Exact aggregates are under `docs/quantity-v2`, `docs/morgan-v3` and `docs/loss-v4`.

Actual GNN checkpoint: ignored `models/loss-v4/selected.pt`, SHA256 `a23bf986ffaa53d6003dd6648f98141b697438c790c9bb8ebc60adab50ec2e97`. Production reload matched the refit example exactly. The real-data two-epoch smoke also matched checkpoint inference exactly and produced finite gradients. The current source suite passed **40 tests**, including component batching/permutation invariance, target exclusion, quantity units and forest routing/save-load.

A fresh venv installed, but Windows Application Control blocked its RDKit DLL import. Executed runs therefore used the existing working Python 3.14/Torch 2.14/PyG 2.8/RDKit 2026.03 environment; exact versions are in `docs/quantity-smoke.json` and selection metadata. Upstream Python 3.14 compatibility warnings remain; Python 3.12 is recommended. No policy bypass or security change occurred.

The notebook series is fully executed on real cached data/aggregates. The main notebook exposes features, GINE pooling and the training loop directly. Figures and tables do not contain private row-level predictions or weights. Source hashes, run durations and failures are retained in the ledger; raw data and all binaries stay ignored. [ORD license](docs/ORD-LICENSE.txt) and [provenance](docs/dataset-provenance.json) describe the public sources.

Legacy CLI/API acknowledgement and fresh-output guards were added after all scientific runs. The 40-test suite and 16 focused guard tests passed; scoring runners were mocked. Exclusive report/checkpoint writes preserve existing artifacts, including files created after preflight. No study was rerun and no metric or learned checkpoint changed.
