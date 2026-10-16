import numpy as np
import pytest
import torch
from torch_geometric.data import Batch
from reaction_gnn.descriptor_model import (
    descriptors,
    reaction_graph,
    scaling,
    DescriptorGNN,
    DIM,
)


def record(temp=20):
    return {
        "components": [
            {"smiles": "CCO", "role": 0},
            {"smiles": "Nc1ccccc1", "role": 1},
        ],
        "temperature_c": temp,
    }


def test_descriptors_exclude_metadata_and_preserve_component_order():
    r = record()
    original = descriptors(r)
    r["yield_percent"] = 999
    r["reaction_id"] = "unused"
    r["components"].reverse()
    np.testing.assert_allclose(original, descriptors(r))
    assert original.shape == (DIM,) and np.isfinite(original).all()


def test_scaling_uses_only_explicit_training_and_constant_columns_finite():
    training = [reaction_graph(record())]
    mean, scale = scaling(training)
    first = mean.copy()
    future = record()
    future["components"][0]["smiles"] = "CCCCCCCCCCCCCCCC"
    reaction_graph(future)
    np.testing.assert_array_equal(mean, first)
    assert np.isfinite(scale).all() and (scale > 0).all()
    with pytest.raises(ValueError):
        scaling([])


def test_descriptor_model_batch_and_checkpoint_roundtrip(tmp_path):
    graphs = [reaction_graph(record(20)), reaction_graph(record(40))]
    mean, scale = scaling(graphs)
    torch.manual_seed(1)
    model = DescriptorGNN(mean, scale).eval()
    with torch.no_grad():
        batch = model(Batch.from_data_list(graphs))
        single = torch.cat([model(Batch.from_data_list([g])) for g in graphs])
    torch.testing.assert_close(batch, single, atol=1e-6, rtol=1e-5)
    assert torch.isfinite(batch).all() and ((batch >= 0) & (batch <= 1)).all()
    path = tmp_path / "descriptor.pt"
    torch.save(model.state_dict(), path)
    loaded = DescriptorGNN().eval()
    loaded.load_state_dict(torch.load(path, weights_only=True))
    with torch.no_grad():
        torch.testing.assert_close(
            batch, loaded(Batch.from_data_list(graphs)), rtol=0, atol=0
        )
    assert loaded.head.mean.data_ptr() != model.head.mean.data_ptr()


def test_capacity_control_uses_legacy_graph_conditions():
    graph = reaction_graph(record(), include_descriptors=False)
    assert graph.conditions.shape == (1, 2 + DIM)
    assert torch.count_nonzero(graph.conditions[:, 2:]) == 0
    with torch.no_grad():
        value = DescriptorGNN(include_descriptors=False).eval()(
            Batch.from_data_list([graph])
        )
    assert value.shape == (1,) and torch.isfinite(value).all()


def test_descriptor_fit_rejects_group_overlap_before_optimization():
    from reaction_gnn.descriptor_study import fit

    a = {**record(), "reaction_id": "a", "group": "same", "yield_percent": 20}
    b = {**record(), "reaction_id": "b", "group": "same", "yield_percent": 40}
    with pytest.raises(ValueError, match="overlap"):
        fit([a], [b], True, {}, 42)


def test_descriptor_checkpoint_rejects_corrupted_normalization(tmp_path):
    from reaction_gnn.descriptor_study import save, load

    model = DescriptorGNN()
    clean = tmp_path / "clean.pt"
    save(model, clean)
    state = torch.load(clean, weights_only=True)
    state["state_dict"]["head.scale"][0] = -1
    broken = tmp_path / "broken.pt"
    torch.save(state, broken)
    with pytest.raises(ValueError, match="normalization"):
        load(broken)


def test_capacity_control_has_exact_matched_parameter_count():
    chemical = DescriptorGNN()
    control = DescriptorGNN(include_descriptors=False)
    assert sum(p.numel() for p in chemical.parameters()) == sum(
        p.numel() for p in control.parameters()
    )


def test_matched_control_initial_parameters_equal_chemical_branch():
    torch.manual_seed(123)
    chemical = DescriptorGNN(
        np.arange(DIM, dtype=np.float32), np.ones(DIM, dtype=np.float32)
    )
    torch.manual_seed(123)
    control = DescriptorGNN(include_descriptors=False)
    for (_, left), (_, right) in zip(
        chemical.named_parameters(), control.named_parameters()
    ):
        torch.testing.assert_close(left, right, rtol=0, atol=0)

def test_production_loader_and_inference_match_descriptor_api(tmp_path):
    from reaction_gnn.descriptor_study import save,predict
    from reaction_gnn.train import load_model,predict_record
    for include in [False,True]:
        model=DescriptorGNN(include_descriptors=include).eval()
        path=tmp_path/('chemical.pt' if include else 'control.pt');save(model,path)
        loaded=load_model(path)
        expected=float(predict(model,[record()])[0])
        assert predict_record(loaded,record())==pytest.approx(expected,abs=1e-5)
