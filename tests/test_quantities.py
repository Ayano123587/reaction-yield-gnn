import copy
import numpy as np
import pytest
import torch
from ord_schema.proto import reaction_pb2 as pb
from torch_geometric.data import Batch
from reaction_gnn.quantities import extract_quantitative_inputs
from reaction_gnn.quantity_graph import reaction_graph,quantitative_conditions,NODE_DIM,CONDITION_DIM
from reaction_gnn.quantity_model import QuantityGNN
from reaction_gnn.train import load_model,predict_record


def example():
    r=pb.Reaction();c=r.inputs['substrate'].components.add(reaction_role=pb.ReactionRole.REACTANT,is_limiting=True)
    c.identifiers.add(type=pb.CompoundIdentifier.SMILES,value='CCc1ccccc1');c.amount.moles.value=1.;c.amount.moles.units=pb.Moles.MILLIMOLE
    solvent=r.inputs['solvent'].components.add(reaction_role=pb.ReactionRole.SOLVENT)
    solvent.identifiers.add(type=pb.CompoundIdentifier.SMILES,value='CO');solvent.amount.volume.value=1.;solvent.amount.volume.units=pb.Volume.MILLILITER
    return r


def test_reported_units_are_equivalent_and_outcomes_never_enter_features():
    r=example();first=extract_quantitative_inputs(r)
    r.inputs['substrate'].components[0].amount.moles.value=1000;r.inputs['substrate'].components[0].amount.moles.units=pb.Moles.MICROMOLE
    r.inputs['solvent'].components[0].amount.volume.value=1000;r.inputs['solvent'].components[0].amount.volume.units=pb.Volume.MICROLITER
    assert extract_quantitative_inputs(r)==first
    r.notes.procedure_details='Measured yield99%';r.outcomes.add().products.add(is_desired_product=True).measurements.add(type=pb.ProductMeasurement.YIELD).percentage.value=99
    assert extract_quantitative_inputs(r)==first
    assert next(c for c in first['components'] if c['role']==0)['moles']==pytest.approx(.001)


def test_invalid_amounts_and_missing_units():
    r=example();r.inputs['substrate'].components[0].amount.moles.value=-1
    with pytest.raises(ValueError,match='nonnegative'):extract_quantitative_inputs(r)
    r.inputs['substrate'].components[0].amount.moles.value=float('nan')
    with pytest.raises(ValueError,match='finite'):extract_quantitative_inputs(r)
    r.inputs['substrate'].components[0].amount.moles.value=1;r.inputs['substrate'].components[0].amount.moles.units=pb.Moles.UNSPECIFIED
    features=extract_quantitative_inputs(r)
    assert next(c for c in features['components'] if c['role']==0)['moles'] is None


def test_component_pooling_batch_boundaries_gradients_and_permutation(tmp_path):
    record=extract_quantitative_inputs(example());record['yield_percent']=25
    graph=reaction_graph(record);assert graph.x.shape[1]==NODE_DIM and graph.conditions.shape==(1,CONDITION_DIM)
    torch.manual_seed(1);model=QuantityGNN(16,layers=2);batch=Batch.from_data_list([graph,graph],follow_batch=['component_roles'])
    output=model(batch);((output-batch.y)**2).mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.convs.parameters())
    assert output.shape==(2,) and torch.isfinite(output).all()
    changed=copy.deepcopy(record);changed['components'].reverse();model.eval()
    assert predict_record(model,record)==pytest.approx(predict_record(model,changed),abs=1e-4)
    path=tmp_path/'quant.pt';torch.save({'state_dict':model.state_dict(),'feature_version':2,'hidden':16,'layers':2,'pooling':'component_role'},path)
    assert predict_record(load_model(path),record)==pytest.approx(predict_record(model,record),abs=1e-7)


def test_graph_feature_target_invariance_and_quantity_effect():
    r=extract_quantitative_inputs(example());first=reaction_graph(r);changed=copy.deepcopy(r);changed['yield_percent']=99
    second=reaction_graph(changed);torch.testing.assert_close(first.x,second.x);torch.testing.assert_close(first.conditions,second.conditions)
    next(c for c in changed['components'] if c['role']==2)['volume_l']=.002
    third=reaction_graph(changed);assert not torch.equal(first.conditions,third.conditions)


def test_absent_roles_and_single_atom_missing_amounts():
    record={'components':[{'smiles':'[Na+]','role':1}],'temperature_c':None}
    model=QuantityGNN(16);assert np.isfinite(predict_record(model,record))
    conditions,_=quantitative_conditions(record);assert np.isfinite(conditions).all()


def test_finite_metric_alignment_and_two_fold_adoption_gate():
    from reaction_gnn.train import metrics
    from reaction_gnn.quantity_study import select_gnn
    for y,p in [([],[]),([1,2],[1]),([1],[float('nan')])]:
        with pytest.raises(ValueError,match='Aligned'):metrics(y,p)
    baseline={'config':{'feature_version':1},'mean_validation_mae':25.,'fold_validation_mae':[25.,25.,25.]}
    new={'config':{'feature_version':2},'mean_validation_mae':24.4,'fold_validation_mae':[23.,25.1,25.1]}
    assert select_gnn([baseline,new]) is baseline
    new['fold_validation_mae']=[24.4]*3
    assert select_gnn([baseline,new]) is new


def test_morgan_role_permutation_target_invariance():
    from reaction_gnn.morgan import representation
    record=extract_quantitative_inputs(example())
    first=representation(record)
    assert first.shape==(4112,) and np.isfinite(first).all()
    changed=copy.deepcopy(record);changed["components"].reverse();changed["yield_percent"]=100
    np.testing.assert_array_equal(first,representation(changed))
