"""Training and safe checkpoint replay for matched descriptor GNN candidates."""

from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from .descriptor_model import DescriptorGNN, reaction_graph, scaling
from .train import metrics
from .artifacts import save_torch_new


def predict(model, records):
    graphs = [reaction_graph(r, model.include_descriptors) for r in records]
    if not graphs:
        raise ValueError("Nonempty inference records required")
    model.eval()
    with torch.no_grad():
        values = (
            torch.cat([model(b) for b in DataLoader(graphs, batch_size=32)]).numpy()
            * 100
        )
    if not np.isfinite(values).all() or (values < 0).any() or (values > 100).any():
        raise ValueError("Invalid predictions")
    return values


def fit(training, validation, include_descriptors, fixed, seed, epochs=None):
    if not training or (validation is not None and not validation):
        raise ValueError("Nonempty fitting partitions required")
    partitions = [training] + ([validation] if validation is not None else [])
    for rows in partitions:
        identifiers = [r["reaction_id"] for r in rows]
        targets = np.asarray([r["yield_percent"] for r in rows], dtype=float)
        if (
            len(set(identifiers)) != len(identifiers)
            or not np.isfinite(targets).all()
            or (targets < 0).any()
            or (targets > 100).any()
        ):
            raise ValueError("Unique IDs and finite percentage targets required")
    if validation is not None and (
        {r["reaction_id"] for r in training} & {r["reaction_id"] for r in validation}
        or {r["group"] for r in training} & {r["group"] for r in validation}
    ):
        raise ValueError("Training/validation identifiers or scaffold groups overlap")
    torch.manual_seed(seed)
    graphs = [reaction_graph(r, include_descriptors) for r in training]
    mean, scale = scaling(graphs) if include_descriptors else (None, None)
    model = DescriptorGNN(mean, scale, include_descriptors)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=fixed["lr"], weight_decay=fixed["weight_decay"]
    )
    loader = DataLoader(
        graphs,
        batch_size=fixed["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    maximum = epochs if epochs is not None else fixed["max_epochs"]
    if maximum < 1:
        raise ValueError("Positive epoch count required")
    best = float("inf")
    state = None
    best_epoch = 0
    history = []
    for epoch in range(maximum):
        model.train()
        total = 0.0
        norm_max = 0.0
        for batch in loader:
            loss = torch.nn.functional.mse_loss(model(batch), batch.y)
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite training loss")
            optimizer.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0, error_if_nonfinite=True
            )
            norm_max = max(norm_max, float(norm))
            optimizer.step()
            total += float(loss.detach()) * batch.num_graphs
        item = {
            "epoch": epoch + 1,
            "train_loss_fraction": total / len(graphs),
            "maximum_gradient_norm_before_clip": norm_max,
        }
        if validation is not None:
            score = metrics(
                [r["yield_percent"] for r in validation], predict(model, validation)
            )["mae_percentage_points"]
            item["validation_mae"] = score
            if score < best:
                best = score
                state = deepcopy(model.state_dict())
                best_epoch = epoch + 1
        history.append(item)
        if (
            validation is not None
            and epoch + 1 >= fixed["minimum_epochs"]
            and epoch + 1 - best_epoch >= fixed["early_stopping_patience"]
        ):
            break
    if validation is not None:
        if state is None:
            raise ValueError("No finite validation checkpoint")
        model.load_state_dict(state)
    return model, best_epoch if validation is not None else maximum, history


def save(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_torch_new(
        {
            "model_kind": "descriptor_gnn",
            "feature_version": model.feature_version,
            "include_descriptors": model.include_descriptors,
            "state_dict": model.state_dict(),
        },
        path,
    )


def load(path):
    return from_checkpoint(torch.load(path, map_location="cpu", weights_only=True))


def from_checkpoint(checkpoint):
    if checkpoint["model_kind"] != "descriptor_gnn":
        raise ValueError("Unexpected model kind")
    include = checkpoint["include_descriptors"]
    if type(include) is not bool or checkpoint["feature_version"] != 3:
        raise ValueError("Feature contract changed")
    model = DescriptorGNN(include_descriptors=include)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    if (
        any(not torch.isfinite(v).all() for v in model.state_dict().values())
        or (model.head.scale <= 0).any()
    ):
        raise ValueError("Invalid checkpoint weights/normalization")
    return model
