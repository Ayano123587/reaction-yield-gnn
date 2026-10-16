# Validation selection

The selected role-aware GINE configuration uses all 610 ELN development records in three scaffold-grouped folds. Mean MAE is 23.5668 percentage points. Candidate adoption requires at least 0.5-point mean improvement and two fold wins. No HTE score selects model parameters.

The final model fits the development cohort for 47 epochs. [Selection verification](selection-check.json) preserves the configuration and validation checks.

Full HTE evaluation processes the complete 50,688-row source with frozen ELN-trained models and unchanged quality rules. It is evaluation, not training or model selection. Source coverage and current aggregate results are linked from the report. Previously observed test exposure is a current limitation.
