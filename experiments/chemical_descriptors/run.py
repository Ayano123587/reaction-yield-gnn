"""Private CPU descriptor study, development-only; readable source files."""

from pathlib import Path
import hashlib, json, shutil, subprocess, sys, time, platform

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "torch-geometric==2.8.1",
        "ord-schema==0.9.3",
        "rdkit",
        "pyarrow",
    ],
    check=True,
)
mount = next(p.parent for p in Path("/kaggle/input").rglob("ensemble-payload.json"))
manifest = json.loads((mount / "manifest.json").read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for relative, digest in manifest["files"].items():
    assert sha(mount / relative) == digest, relative
source = Path("/kaggle/working/source/reaction_gnn")
source.mkdir(parents=True)
for p in mount.glob("*.py"):
    shutil.copyfile(p, source / p.name)
sys.path.insert(0, str(source.parent))
import numpy as np
import torch
from reaction_gnn.descriptor_study import fit, predict, save, load
from reaction_gnn.graph import reaction_graph
from reaction_gnn.train import load_model, metrics
from reaction_gnn.quantity_study import predictions

torch.set_num_threads(2)
payload = json.loads((mount / "ensemble-payload.json").read_text())
plan = json.loads((mount / "descriptor-plan.json").read_text())
records = payload["development_records"]
folds = payload["fold_assignment"]
fixed = payload["fixed_training"]
assert len(records) == 610 and len({r["reaction_id"] for r in records}) == 610
out = Path("/kaggle/working")
started = time.monotonic()
report = {
    "plan_sha256": sha(mount / "descriptor-plan.json"),
    "input_manifest_sha256": sha(mount / "manifest.json"),
    "scope": "610developmentrecords only; noELNtest/HTEinputs or targets",
    "environment": {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": "cpu",
        "threads": 2,
    },
    "smoke": [],
    "folds": [],
}
training = [r for r in records if folds[r["group"]] != 0][:32]
validation = [r for r in records if folds[r["group"]] == 0][:8]
for include in [False, True]:
    model, epoch, history = fit(training, validation, include, fixed, 42, epochs=2)
    p = predict(model, validation)
    path = out / ("smoke-descriptors.pt" if include else "smoke-control.pt")
    save(model, path)
    assert np.array_equal(p, predict(load(path), validation))
    report["smoke"].append(
        {
            "include_descriptors": include,
            "training_rows": 32,
            "validation_rows": 8,
            "epochs": 2,
            "reload_delta_pp": 0,
            "finite_gradients": all(
                np.isfinite(h["maximum_gradient_norm_before_clip"]) for h in history
            ),
        }
    )
assert all(r["finite_gradients"] for r in report["smoke"])
for fold in range(3):
    training = [r for r in records if folds[r["group"]] != fold]
    validation = [r for r in records if folds[r["group"]] == fold]
    baseline_rows = json.loads((mount / f"fold{fold}-predictions.json").read_text())
    baseline = {r["reaction_id"]: r["prediction"] for r in baseline_rows}
    assert set(baseline) == {r["reaction_id"] for r in validation}
    y = np.array([r["yield_percent"] for r in validation])
    p = np.array([baseline[r["reaction_id"]] for r in validation])
    replay = predictions(
        load_model(mount / f"fold{fold}.pt"), [reaction_graph(r) for r in validation], 1
    )
    assert np.max(abs(replay - p)) < 0.001
    item = {
        "fold": fold,
        "training_rows": len(training),
        "validation_rows": len(validation),
        "champion_mae": metrics(y, p)["mae_percentage_points"],
        "baseline_replay_delta_pp": float(np.max(abs(replay - p))),
        "candidates": [],
    }
    for config in plan["candidates"]:
        model, epoch, history = fit(
            training, validation, config["include_descriptors"], fixed, 42 + fold
        )
        p = predict(model, validation)
        path = out / f"{config['name']}-fold{fold}.pt"
        save(model, path)
        assert np.array_equal(p, predict(load(path), validation))
        (out / f"{config['name']}-fold{fold}-predictions.json").write_text(
            json.dumps(
                [
                    {
                        "reaction_id": r["reaction_id"],
                        "target": float(t),
                        "prediction": float(v),
                    }
                    for r, t, v in zip(validation, y, p)
                ]
            )
        )
        item["candidates"].append(
            {
                "name": config["name"],
                "include_descriptors": config["include_descriptors"],
                "mae": metrics(y, p)["mae_percentage_points"],
                "best_epoch": epoch,
                "history": history,
                "checkpoint_sha256": sha(path),
                "reload_delta_pp": 0,
            }
        )
    report["folds"].append(item)
    print(json.dumps({k: v for k, v in item.items() if k != "candidates"}), flush=True)
base = np.array([f["champion_mae"] for f in report["folds"]])
summary = []
for config in plan["candidates"]:
    errors = np.array(
        [
            next(c["mae"] for c in f["candidates"] if c["name"] == config["name"])
            for f in report["folds"]
        ]
    )
    gain = float(base.mean() - errors.mean())
    wins = int((errors < base).sum())
    summary.append(
        {
            "name": config["name"],
            "fold_mae": errors.tolist(),
            "mean_mae": float(errors.mean()),
            "gain_pp": gain,
            "fold_wins": wins,
            "eligible": gain >= 0.5 and wins >= 2,
        }
    )
eligible = [r for r in summary if r["eligible"]]
report["selection"] = {
    "champion_mean_mae": float(base.mean()),
    "candidates": summary,
    "selected": (
        min(eligible, key=lambda r: r["mean_mae"])["name"]
        if eligible
        else "legacy_role48"
    ),
}
# Serialize the validation decision before fitting a final model; no test data exist here.
(out / "selection.json").write_text(json.dumps(report["selection"], indent=2))
selected = report["selection"]["selected"]
if selected != "legacy_role48":
    config = next(c for c in plan["candidates"] if c["name"] == selected)
    epochs = int(
        np.median(
            [
                next(c["best_epoch"] for c in f["candidates"] if c["name"] == selected)
                for f in report["folds"]
            ]
        )
    )
    model, _, history = fit(
        records, None, config["include_descriptors"], fixed, 42, epochs=epochs
    )
    save(model, out / "selected.pt")
    example = {
        "components": [
            {"smiles": "Brc1ccccc1", "role": 0},
            {"smiles": "Nc1ccccc1", "role": 0},
            {"smiles": "CCO", "role": 2},
        ],
        "temperature_c": 80.0,
    }
    point = predict(load(out / "selected.pt"), [example])
    report["final_model"] = {
        "training_rows": len(records),
        "epochs": epochs,
        "history": history,
        "checkpoint_sha256": sha(out / "selected.pt"),
        "public_example_percent": float(point[0]),
        "include_descriptors": config["include_descriptors"],
    }
report["runtime_seconds"] = time.monotonic() - started
(out / "aggregate.json").write_text(json.dumps(report, indent=2, allow_nan=False))
(out / "completion.json").write_text(
    json.dumps({"status": "complete", "aggregate_sha256": sha(out / "aggregate.json")})
)
print("Complete development selection:", report["selection"], flush=True)
