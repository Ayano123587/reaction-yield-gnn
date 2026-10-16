# Sources, chemistry and model assumptions

## Primary sources

- [Open Reaction Database official data](https://github.com/open-reaction-database/ord-data) and [pinned repository revision](https://github.com/open-reaction-database/ord-data/tree/83f971f586f6ad18f358ae4ae99d045e94ed2066): authoritative source, licensing and Parquet schema. Official storage is one serialized Reaction protobuf per Parquet row; the loader reads those bytes with `ord_schema.proto.reaction_pb2` and PyArrow rather than assuming JSON columns.
- [ORD schema](https://docs.open-reaction-database.org/en/stable/schema.html): input roles, conditions and outcome measurements. Only `YIELD` measurements with an explicit `percentage` message for one desired product become targets; AREA and arbitrary float measurements are not silently treated as yields.
- [AstraZeneca ELN source dataset](https://open-reaction-database.org/dataset/ord_dataset-00005539a1e04c809a9a78647bea649c) and [original real-world yield prediction paper](https://pubs.rsc.org/en/content/articlelanding/2023/sc/d2sc06041h):750 Buchwald–Hartwig ELN reactions and the challenges of real-world outcome prediction.
- [Additional C–N source](https://open-reaction-database.org/dataset/ord_dataset-805ad863feef48579d95d86a728035f4):50,688 Pd/Ni/Cu reactions from only2 amines×4 aryl halides. Different source and assay methodology mean transfer metrics cannot be merged with ELN scaffold-test metrics.
- [RDKit molecular operations](https://www.rdkit.org/docs/source/rdkit.Chem.rdmolops.html): core molecular parsing and Murcko decomposition. Core `Chem.MurckoDecompose` gives the same scaffold operation without importing optional descriptors blocked in the executed Windows environment. No machine security setting is altered.
- [PyTorch Geometric GINEConv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINEConv.html): edge-feature-aware graph message passing.
- [RDKit fingerprints](https://www.rdkit.org/docs/source/rdkit.Chem.rdmolops.html#rdkit.Chem.rdmolops.RDKFingerprint): the baseline uses256-bit path fingerprints independently per role, not Morgan fingerprints or learned embeddings.

## Leakage controls and limits

Features are an allow-list extracted from `Reaction.inputs` plus temperature setpoint. Free text often contains measured yields, product identity or purification outcomes, so all notes and procedures are excluded. Product structures can be useful for yield modeling but are deliberately excluded here to enforce the requested pre-reaction-input contract. Stoichiometric amounts are omitted rather than mixing inconsistent units.

The principal substrate grouping is a declared evaluation assumption, not a proof that every molecular motif is unseen. Exact input signatures are separately checked; the check includes canonical molecules with atom maps removed, roles and temperature. Cloned reactant combinations remain in the same principal-scaffold partition regardless of duplicated records or condition changes. Reagents/catalysts and smaller substrate scaffolds can recur across splits and are disclosed.

Four GNN configurations and a three-point forest grid use the same development folds. Results are selected by mean fold validation MAE. Early stopping and refit epochs are based on development labels only. Three-fold estimates from610 rows are noisy, and the original preliminary smoke exposed one test aggregate; independent prospective or laboratory validation is still necessary.

## Target semantics

The original ELN contains heterogeneous reaction conditions and reporting quality. We keep only a single unambiguous desired-product percentage target and drop nonfinite temperatures rather than silently imputing corrupted values. Six target records and four temperature records were rejected. No yield value is normalized based on test-set statistics.

The transfer cohort uses calibrated UPLC-MS assay yields and much narrower substrates. Explicit no-ligand/no-catalyst entries are not real molecules and are skipped only when named as absent. Any other missing SMILES causes rejection. Selection for source-transfer evaluation ranks SHA256(reaction_id) before reading outcomes, avoiding yield-stratified cherry-picking. Exact input clones from the ELN cohort are removed; scaffold overlap is measured and reported.

## What is not established

These experiments do not prove a universal reaction model, a best achievable chemistry score, calibrated uncertainty or reliable synthesis planning. They do not optimize chemical conditions. External assay transfer may fail badly despite valid software and development validation. Such a result is retained as an engineering/scientific limitation, not replaced by a synthetic success metric.

## Repeated input feature patterns

[Executed duplicate diagnostics](feature-duplicate-diagnostics.json) found119/6/22 duplicate rows beyond first within the original train/validation/test partitions. There are444/41/108unique input-structure/role/temperature signatures among563/47/130rows. These can be genuine repeat experiments or conditions that this feature representation omits. They are deliberately kept in the same substrate-scaffold partition; no exact feature clone crosses splits. Rows within a repeated-feature group are not independent evidence, and row-level errors weight common conditions more heavily. Expanding the quantitative condition representation is a future model improvement, not a solved issue.
