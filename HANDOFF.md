# Reaction yield GNN: resumed research

The user resumed and authorized publication of this project. The original experiment below remains historical evidence. Quantity-aware three-fold validation is running under plan `28fa125` and tested source `9cf085a`; independent-source HTE remains unscored. A larger-fingerprint followup was preregistered as `81fb38f` before any holdout release. No new quality result is claimed until aggregate reports complete.

The end-to-end notebook has passed schema validation and all six default code cells ran successfully. Training, acquisition and inference are explicit opt-ins; the notebook never silently rescores the observed ELN holdout. Data/weights/row predictions are ignored. Current real-data smoke and 21 tests passed; see docs/quantity-preflight.md.

## Historical checkpoint (prior paused scope)

# Local checkpoint — reaction yield GNN

Latest user scope: finish only the five repositories already created on GitHub (leakage audit, complexity predictor, literary RAG, regime quant, overlapping audio), then pause the goal. Reaction yield is local only and deferred. No additional experiments, source-transfer jobs or projects were launched after this instruction.

## Authoritative process status

- Current optimization exec session22455: **terminal, exit0**, confirmed by final process output.
- Four GNN configurations × three scaffold-disjoint development folds completed. The existing atomic process included its already-started final refit and report generation; no separate follow-up training/refit job was started.
- Actual runtime: 624.71seconds, CPU two threads.
- Selected configuration: role-specific pooling, hidden48, three GINE layers, learning rate0.003; refit43epochs on610development rows.
- Checkpoint: `D:\CodexData\portfolio\reaction-yield-gnn\models\optimized.pt` (108884bytes), ignored binary retained locally.
- Checkpoint SHA256: `781a53a19d5b705401ef44e23eb176f387bf36b47346ceb1005e17d957b40b88`.
- Reports: `docs/optimized-results.json`, `docs/selection-protocol.json`, `docs/selection-progress.jsonl`, `docs/run-status.json`.

## Measured result and limits

130principal-scaffold-heldout ELN rows: GNN MAE25.2029percentage points; development-median25.6229; selected role-fingerprint forest23.8645. GNN test R2−0.0244. The stronger baseline wins, and useful cross-scaffold generalization is not established. Four bounded GNN configurations were selected by three-fold validation, not by test score.

An earlier2epoch mechanics smoke exposed an aggregate on the same ELN test before the stricter selection protocol. This historical exposure is disclosed and the test is not described as completely untouched. The separately sourced50,688C–N cohort was downloaded and SHA-verified, but source-transfer evaluation is **unexecuted**. No metrics for it exist.

## Validated software

15pytest tests passed after final parser/finite-input guards. Editable installation, wheel building and compilation checks passed. The trained GNN checkpoint is state-dict based and architecture/version tagged; checkpoint save/load and scalar inference are covered by meaningful tests. All text is explicitly UTF-8. Raw/processed data and model binaries are ignored. No GitHub repo or publication was created for this local project.

## Deferred work if user later resumes

Review whether the representation should incorporate unit-safe stoichiometry/concentration and richer pre-reaction conditions, then use a new validation protocol to compare against stronger fingerprints and uncertainty-aware models. Evaluate the preserved separate-source cohort without source-driven tuning, retaining any poor result and measurement-domain limitations. The narrow two-amine/four-aryl HTE source is not an indiscriminate training augmentation for diverse ELN scaffolds. Do not claim this checkpoint finishes the original reaction project or reaches a globally optimal score.
