# Validation selection

The selected role-aware GINE configuration uses all 610 ELN development records in three scaffold-grouped folds. Mean MAE is 23.5668 points. Candidate adoption requires at least 0.5-point mean improvement and two fold wins; the gate remained fixed. Quantities, alternative pooling/losses, fingerprint forests, equal seed averaging and matched chemical descriptors did not replace the selected model. No HTE score selected model parameters.

The descriptor check used six actual CPU fits, a parameter-matched control, fold-training-only descriptor scaling, finite-gradient checks and exact candidate checkpoint reloads. It took 549.88 seconds. Its source was frozen at `fa251f7`; the aggregate SHA256 is `e2664a6a912a3641f4e30710a0b56d421576567d999c2a25bb72c2f62be454a7`. Compact selection evidence is in [selection-check.json](selection-check.json). Detailed candidate histories stay in private run archives.

Full HTE evaluation processes the complete 50,688-row source with frozen ELN-trained models and unchanged quality rules. It is evaluation, not training or model selection. Source coverage and current aggregate results are linked from the report. Previously observed test exposure is a current limitation.
