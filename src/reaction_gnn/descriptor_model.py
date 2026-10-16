"""Input-only chemical summaries alongside the retained molecular graph."""

import numpy as np
import torch
from torch import nn
from rdkit.Chem import Crippen, Lipinski, rdMolDescriptors, Descriptors
from .data import canonical
from .graph import reaction_graph as legacy_graph
from .model import YieldGNN

NAMES = (
    "heavy_atoms",
    "molecular_weight",
    "logP",
    "TPSA",
    "H_donors",
    "H_acceptors",
    "rotatable_bonds",
    "rings",
    "fraction_CSP3",
    "nitrogen",
    "oxygen",
    "phosphorus",
    "sulfur",
    "halogens",
    "formal_charge",
    "aromatic_atoms",
)
DIM = 4 * (len(NAMES) + 1)


def descriptors(record):
    by_role = [[] for _ in range(4)]
    for component in record["components"]:
        role = component["role"]
        if isinstance(role, bool) or role not in range(4):
            raise ValueError("Role must be0..3")
        _, mol = canonical(component["smiles"])
        atoms = list(mol.GetAtoms())
        values = [
            mol.GetNumHeavyAtoms(),
            Descriptors.MolWt(mol),
            Crippen.MolLogP(mol),
            rdMolDescriptors.CalcTPSA(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol),
            Lipinski.NumRotatableBonds(mol),
            rdMolDescriptors.CalcNumRings(mol),
            rdMolDescriptors.CalcFractionCSP3(mol),
        ]
        values.extend(sum(a.GetAtomicNum() == z for a in atoms) for z in [7, 8, 15, 16])
        values.extend(
            [
                sum(a.GetAtomicNum() in [9, 17, 35, 53] for a in atoms),
                sum(a.GetFormalCharge() for a in atoms),
                sum(a.GetIsAromatic() for a in atoms),
            ]
        )
        assert len(values) == len(NAMES)
        by_role[role].append(values)
    result = []
    for values in by_role:
        result.extend(
            np.mean(values, axis=0).tolist() if values else [0.0] * len(NAMES)
        )
        result.append(len(values))
    result = np.asarray(result, dtype=np.float32)
    if result.shape != (DIM,) or not np.isfinite(result).all():
        raise ValueError("Invalid chemical descriptor vector")
    return result


def reaction_graph(record, include_descriptors=True):
    graph = legacy_graph(record)
    values = (
        descriptors(record) if include_descriptors else np.zeros(DIM, dtype=np.float32)
    )
    graph.conditions = torch.cat(
        [graph.conditions, torch.from_numpy(values).reshape(1, -1)], dim=1
    )
    return graph


def scaling(training_graphs):
    """Fit descriptor normalization on fold-training graphs only."""
    if not training_graphs:
        raise ValueError("Nonempty training graphs required")
    values = np.vstack([g.conditions.numpy()[:, 2:] for g in training_graphs])
    if values.shape[1] != DIM or not np.isfinite(values).all():
        raise ValueError("Descriptor dimensions/values invalid")
    mean = values.astype(np.float64).mean(axis=0)
    scale = values.astype(np.float64).std(axis=0)
    scale[scale < 1e-8] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


class ScaledHead(nn.Module):
    def __init__(self, hidden, mean, scale):
        super().__init__()
        mean = torch.as_tensor(mean, dtype=torch.float32)
        scale = torch.as_tensor(scale, dtype=torch.float32)
        if (
            mean.shape != scale.shape
            or mean.ndim != 1
            or not torch.isfinite(mean).all()
            or not torch.isfinite(scale).all()
            or (scale <= 0).any()
        ):
            raise ValueError("Invalid training-only normalization")
        self.register_buffer("mean", mean.clone())
        self.register_buffer("scale", scale.clone())
        self.network = nn.Sequential(
            nn.Linear(hidden * 4 + 2 + mean.numel(), 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        count = self.mean.numel()
        if count:
            x = torch.cat(
                [x[:, :-count], (x[:, -count:] - self.mean) / self.scale], dim=1
            )
        return self.network(x)


class DescriptorGNN(YieldGNN):
    def __init__(self, mean=None, scale=None, include_descriptors=True):
        super().__init__(hidden=48, layers=3, pooling="role", dropout=0.0)
        self.include_descriptors = include_descriptors
        self.feature_version = 3
        if include_descriptors:
            mean = np.zeros(DIM, dtype=np.float32) if mean is None else mean
            scale = np.ones(DIM, dtype=np.float32) if scale is None else scale
        else:
            mean = np.zeros(DIM, dtype=np.float32)
            scale = np.ones(DIM, dtype=np.float32)
        if len(mean) != DIM:
            raise ValueError("Incorrect descriptor normalization length")
        self.head = ScaledHead(48, mean, scale)
