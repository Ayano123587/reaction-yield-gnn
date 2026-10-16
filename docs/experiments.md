# Experiment ledger

| Round | Preregistered commit | Fits | Scientific wall time | Outcome |
|---|---|---:|---:|---|
| Quantity/component pooling | 28fa125 | 12 GNN +27 forest | 925.56s | Small Huber gain fails 0.5 pp/two-fold gate |
| Morgan1024 fingerprints | 81fb38f | 12 forest | 212.32s | Best gain 0.112 pp; gate fails |
| Robust graph loss | 6d390cf | 9 GNN | 485.11s | All worse than matched MSE baseline |
| Final release | Freeze 15cf75a | One47-epoch refit +forest | See execution attestation | Observed ELN once; independent HTE once; no subsequent tuning |

No scientific numerical failures occurred. Source and processed-data hashes, configs, fold metrics and aggregate training histories remain in `docs/quantity-v2`, `docs/morgan-v3` and `docs/loss-v4`. The fresh-venv RDKit Application Control import failure is an environment failure, not a discarded model result. The first source/real-data smoke checkpoint and production inference route were validated before training. Forest production routing guards were added after the first study; they do not change its executed scores. All runners are readable Python. Raw reactions, row predictions and checkpoint binaries remain ignored.
