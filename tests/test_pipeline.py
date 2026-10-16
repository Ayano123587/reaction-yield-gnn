import json
import numpy as np
import pytest
import torch
from ord_schema.proto import reaction_pb2 as pb
from torch_geometric.data import Batch
from reaction_gnn.data import canonical,extract_inputs,extract_target,scaffold_group,split_records,prepare
from reaction_gnn.graph import reaction_graph,NODE_DIM
from reaction_gnn.model import YieldGNN
from reaction_gnn.train import metrics,load_model,predict_record,fingerprint

torch.set_num_threads(2)

def example():
    reaction=pb.Reaction(reaction_id="ord-unit-test")
    component=reaction.inputs["reactant"].components.add()
    component.reaction_role=pb.ReactionRole.REACTANT
    component.identifiers.add(type=pb.CompoundIdentifier.SMILES,value="CCc1ccccc1")
    reaction.conditions.temperature.setpoint.value=373.15
    reaction.conditions.temperature.setpoint.units=pb.Temperature.KELVIN
    product=reaction.outcomes.add().products.add(is_desired_product=True)
    product.measurements.add(type=pb.ProductMeasurement.YIELD).percentage.value=25
    return reaction

def record():
    return {"components":[{"smiles":"CCc1ccccc1","role":0},{"smiles":"[Na+]","role":1}],"temperature_c":100,"yield_percent":25}

def test_allowlist_excludes_all_outcome_and_notes():
    reaction=example();before=extract_inputs(reaction)
    reaction.notes.procedure_details="Measured yield 99%; product CCCCCCC"
    reaction.outcomes[0].products[0].identifiers.add(type=pb.CompoundIdentifier.SMILES,value="CCCCCC")
    reaction.outcomes[0].products[0].measurements[0].percentage.value=99
    assert extract_inputs(reaction)==before
    assert before["temperature_c"]==pytest.approx(100)
    assert extract_target(reaction)==99

def test_zero_yield_valid_and_ambiguous_yield_rejected():
    reaction=example();reaction.outcomes[0].products[0].measurements[0].percentage.value=0
    assert extract_target(reaction)==0
    reaction.outcomes[0].products[0].measurements.add(type=pb.ProductMeasurement.YIELD).percentage.value=20
    with pytest.raises(ValueError,match="unambiguous"): extract_target(reaction)

def test_missing_smiles_invalid_molecule():
    reaction=example();reaction.inputs["reactant"].components[0].ClearField("identifiers")
    with pytest.raises(ValueError,match="SMILES"):extract_inputs(reaction)
    with pytest.raises(ValueError): canonical("not-a-molecule")

def test_scaffold_group_and_acyclic_fallback():
    assert scaffold_group([{ "smiles":"CCc1ccccc1","role":0}])==scaffold_group([{ "smiles":"OCc1ccccc1","role":0}])
    assert scaffold_group([{ "smiles":"CCC","role":0}])=="CCC"
    assert canonical("[CH3:1][OH:2]")[0]=="CO"

def test_group_split_disjoint_and_deterministic():
    records=[{"reaction_id":str(i),"group":f"scaffold-{i//3}"} for i in range(30)]
    first=split_records(records);assert first==split_records(list(reversed(records))) or {s:{r["reaction_id"] for r in rows} for s,rows in first.items()}=={s:{r["reaction_id"] for r in rows} for s,rows in split_records(list(reversed(records))).items()}
    groups={s:{r["group"] for r in rows} for s,rows in first.items()}
    assert not (groups["train"]&groups["test"] or groups["train"]&groups["validation"] or groups["validation"]&groups["test"])
    with pytest.raises(ValueError):split_records(records[:3])

def test_graph_edges_single_atom_and_targets():
    graph=reaction_graph(record())
    assert graph.x.shape[1]==NODE_DIM and graph.edge_index.shape[0]==2
    assert graph.edge_index.shape[1]==graph.edge_attr.shape[0]
    assert graph.edge_index.max()<len(graph.x)
    single=reaction_graph({"components":[{"smiles":"[Na+]","role":1}]})
    assert single.edge_index.shape==(2,0) and single.edge_attr.shape==(0,4)
    assert not hasattr(single,"y") or single.y is None
    assert graph.y.item()==pytest.approx(.25)

def test_message_passing_gradients_and_permutation():
    torch.manual_seed(42);model=YieldGNN(16)
    item=record();graph=reaction_graph(item);batch=Batch.from_data_list([graph,graph])
    output=model(batch);loss=((output-batch.y)**2).mean();loss.backward()
    assert output.shape==(2,) and torch.isfinite(output).all()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.convs.parameters())
    item["components"].reverse()
    model.eval()
    assert predict_record(model,item)==pytest.approx(predict_record(model,record()),abs=1e-4)

def test_checkpoint_roundtrip(tmp_path):
    model=YieldGNN(16);path=tmp_path/"model.pt"
    torch.save({"state_dict":model.state_dict(),"hidden":16,"feature_version":1},path)
    assert predict_record(load_model(path),record())==pytest.approx(predict_record(model,record()))

def test_metrics_and_baseline_features():
    result=metrics([10,30],[20,20]);assert result["mae_percentage_points"]==10
    assert result["rmse_percentage_points"]==10 and result["r2"]==0
    features=fingerprint(record());assert features.shape==(1026,)
    changed=record();changed["yield_percent"]=99
    assert np.array_equal(features,fingerprint(changed))

def test_parquet_loader_proto_roundtrip(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    reaction=example();path=tmp_path/"tiny.parquet"
    pq.write_table(pa.table({"reaction_id":[reaction.reaction_id],"reaction":[reaction.SerializeToString()]}),path)
    dataset=prepare(path,tmp_path/"prepared.json")
    assert dataset["raw_rows"]==1 and len(dataset["records"])==1
    assert dataset["records"][0]["yield_percent"]==25


def test_role_pooling_handles_absent_roles_and_checkpoint(tmp_path):
    model=YieldGNN(16,layers=2,pooling="role",dropout=.1)
    simple={"components":[{"smiles":"CCO","role":0}]}
    assert np.isfinite(predict_record(model,simple))
    path=tmp_path/"role.pt"
    torch.save({"state_dict":model.state_dict(),"hidden":16,"layers":2,"pooling":"role","dropout":.1,"feature_version":1},path)
    model.eval()
    assert predict_record(load_model(path),simple)==pytest.approx(predict_record(model,simple))

def test_exact_clone_overlap_rejected():
    from reaction_gnn.data import audit_split_integrity
    a={**record(),"group":"deliberately-wrong-a"};b={**record(),"group":"deliberately-wrong-b"}
    with pytest.raises(ValueError,match="clone"):
        audit_split_integrity({"train":[a],"test":[b]})

def test_missing_catalyst_sentinel_is_not_molecule():
    reaction=example();component=reaction.inputs["empty-catalyst"].components.add()
    component.reaction_role=pb.ReactionRole.CATALYST
    component.identifiers.add(type=pb.CompoundIdentifier.NAME,value="none")
    assert len(extract_inputs(reaction)["components"])==1


def test_missing_numeric_value_not_silent_zero():
    reaction=example();reaction.outcomes[0].products[0].measurements[0].percentage.ClearField("value")
    with pytest.raises(ValueError,match="unambiguous"):extract_target(reaction)
    reaction.conditions.temperature.setpoint.ClearField("value")
    assert extract_inputs(reaction)["temperature_c"] is None


def test_prediction_temperature_must_be_finite():
    item=record();item["temperature_c"]=float("nan")
    with pytest.raises(ValueError,match="finite"):reaction_graph(item)
