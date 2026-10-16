# Preflight — 16 October 2026

Status: ready with limitations for a bounded CPU experiment, not a competition submission.

- Task: percentage yield regression on the official ORD AstraZeneca 750-reaction ELN dataset. Lower MAE in percentage points is better; RMSE and R2 are secondary. Scorer is `reaction_gnn.train.metrics` and has an exact known-answer unit test.
- Data: 274285-byte downloaded Parquet SHA256 is pinned in `docs/dataset-provenance.json`. Protobuf decoding succeeded. 740 usable reactions: six ambiguous/missing/out-of-range yield targets and four nonfinite temperature records were rejected.
- Evaluation: seed-42 grouping by the largest-heavy-atom reactant Murcko scaffold, with canonical acyclic fallback; 563 train / 47 validation / 130 test reactions and 90 / 19 / 20 distinct groups. Scaffold groups are disjoint. Only the principal substrate scaffold is protected; reagents and smaller reactants can overlap.
- Features: allow-list of input molecule SMILES and roles plus pre-reaction temperature setpoint. Notes, products, outcomes, reaction IDs, source names and dates are excluded. No product or yield-derived graph features.
- Baselines: training-median yield, and a fixed 100-tree role-aware 256-bit-per-role RDKit path fingerprint RandomForest (min leaf 3), evaluated on exactly the GNN test split.
- Environment: local Python 3.14.2, Torch 2.14 CPU, PyG 2.8.1, RDKit 2026.03.6, ord-schema 0.9.3 and PyArrow 25.0.1. Ten unit tests passed. Upstream Torch/PyG issue Python-3.14 warnings; CI targets Python 3.12. Optional RDKit descriptor DLL imports are blocked by Windows Application Control. Production code uses supported core RDKit `Chem.MurckoDecompose` and `Chem.RDKFingerprint` instead; no policy changes made.
- Compute/storage: CPU only, two Torch threads, approximately 142 GB free on D:. Data under ignored `data/`, checkpoints under ignored `models/`, aggregate reports under `docs/`. No Kaggle/GPU job is necessary for this bounded sample.

Smoke command: `reaction-gnn train --epochs 2 --checkpoint models/smoke.pt --report docs/smoke-results.json`. Full benchmark command after smoke acceptance: `reaction-gnn train --epochs 30 --checkpoint models/yield.pt --report docs/results.json`.

Hypothesis: input graph message passing can outperform a train-only constant baseline on unseen principal-substrate scaffolds. The fixed fingerprint forest is a stronger control and may outperform the GNN. Hold data, split and seed fixed; do not tune on test performance. Budget: one 30-epoch CPU benchmark, at most 30 minutes, stop on nonfinite loss or invalid outputs. The short run checks mechanics, not model quality. The benchmark will be reported whether the hypothesis succeeds or fails.
