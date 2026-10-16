# Bounded training check

Executed command: `python -u -m reaction_gnn.cli train --epochs 2 --checkpoint models/smoke.pt --report docs/smoke-results.json`.

Exit status0. Seed42, clean code revision1e4d5a3469e507fe2ff7dc19fbcf06d66da2559d, processed-data SHA256cff11c38744ec83ef635d21464e6316690d59468ad823d234db7dbda54363eb0. Training/runtime including graph construction and baselines:8.62seconds on local CPU, two threads. Checkpoint and aggregate report created. The13unit tests separately verify finite message-passing gradients, checkpoint serialization/reload and scalar inference, percentage-point MAE, original scaffold and exact-clone split integrity, isolated atoms, input/label separation and a protobuf/Parquet round trip.

The exploratory smoke produced finite losses and predictions. It is a mechanics check, not full model-quality evidence. Its held-out aggregate was exposed before the stricter optimization plan was adopted; the repository explicitly discloses that historical exposure. No smoke-based test tuning is used in subsequent model selection. The separately sourced transfer cohort is evaluated only after selected-model refit.

Subsequent experiment: `reaction-gnn optimize --epochs 60` with four configurations and three scaffold-disjoint development folds, then final refit and evaluation; see optimization-plan.md and generated selection-protocol.json. Final metrics must be read from actual generated reports, not this readiness note.
