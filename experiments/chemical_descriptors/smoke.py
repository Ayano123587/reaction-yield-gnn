"""Two-epoch real-development mechanics smoke; never scores held-out cohorts."""

import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from reaction_gnn.quantity_study import partitions
from reaction_gnn.descriptor_study import fit, predict, save, load

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--input", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
output = Path(a.output)
if output.exists():
    raise ValueError("Choose a fresh smoke output directory")
dataset = json.loads(Path(a.input).read_text(encoding="utf-8"))
dev, _, fold = partitions(dataset)
training = [r for r in dev if fold[r["group"]] != 0][:32]
validation = [r for r in dev if fold[r["group"]] == 0][:8]
fixed = {
    "lr": 0.003,
    "weight_decay": 0.0001,
    "batch_size": 24,
    "max_epochs": 120,
    "minimum_epochs": 20,
    "early_stopping_patience": 20,
}
output.mkdir(parents=True)
torch.set_num_threads(2)
started = time.monotonic()
rows = []
for include in [False, True]:
    model, epoch, history = fit(training, validation, include, fixed, 42, epochs=2)
    values = predict(model, validation)
    path = output / ("chemical.pt" if include else "control.pt")
    save(model, path)
    replay = predict(load(path), validation)
    assert np.array_equal(values, replay)
    rows.append(
        {
            "include_descriptors": include,
            "training_rows": len(training),
            "validation_rows": len(validation),
            "epochs": 2,
            "history": history,
            "reload_delta_pp": 0,
            "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "parameters": sum(p.numel() for p in model.parameters()),
        }
    )
assert rows[0]["parameters"] == rows[1]["parameters"]
result = {
    "scope": "Real development mechanics only, not accuracy evidence",
    "torch": torch.__version__,
    "results": rows,
    "runtime_seconds": time.monotonic() - started,
    "source_sha256": {
        name: hashlib.sha256((Path("src/reaction_gnn") / name).read_bytes()).hexdigest()
        for name in ["descriptor_model.py", "descriptor_study.py"]
    },
}
(output / "aggregate.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(
    json.dumps(
        {
            "status": "passed",
            "parameters_per_model": rows[0]["parameters"],
            "reload_delta_pp": 0,
            "runtime_seconds": result["runtime_seconds"],
        }
    )
)
