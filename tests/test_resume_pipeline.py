"""Mocked evaluator contracts; no scientific corpus or checkpoint is scored."""
import hashlib,json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from ord_schema.proto import reaction_pb2 as pb
from reaction_gnn import full_transfer_resume as module
from reaction_gnn.resume import snapshot


def fixture(tmp_path,monkeypatch,wrong_target=False):
    monkeypatch.chdir(tmp_path)
    def write(path,value):
        p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return p
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    source=Path('source.parquet')
    pq.write_table(pa.Table.from_pylist([{'reaction_id':f'r{i}','reaction':pb.Reaction(reaction_id=f'r{i}').SerializeToString()} for i in range(3)]),source)
    eln=write('data/processed/reactions-quantity-v2.json',{'records':[{'components':[{'smiles':'CC','role':0}],'temperature_c':20,'yield_percent':50,'group':'CC'}]})
    selection=write('docs/loss-v4/selection.json',{'feature_sha256':{},'selected_gnn':{'feature_version':1},'selected_forest':{}})
    for name in ['selected.pt','forest.joblib']:
        p=Path('models/loss-v4')/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'mocked checkpoint')
    write('docs/loss-v4/ELN-results.json',{'selection_sha256':sha(selection)})
    plan={'selection_directory':'docs/loss-v4','source_rows_expected':3,'source_sha256':sha(source),'processed_eln_sha256':sha(eln),'gnn_checkpoint_sha256':sha('models/loss-v4/selected.pt'),'forest_checkpoint_sha256':sha('models/loss-v4/forest.joblib'),'threads':1,'batch_rows':2,'scope':'unit fixture','test_exposure':'unit fixture'}
    p=write('plan.json',plan)
    prefix=write('host.jsonl',{'reaction_id':'r0','target':11 if wrong_target else 10,'gnn':20,'forest':30})
    prefix.write_text(prefix.read_text()+'\n')
    manifest={k:plan[k] for k in ['source_sha256','gnn_checkpoint_sha256','forest_checkpoint_sha256','processed_eln_sha256']}
    manifest.update(prefix=snapshot(prefix,'prefix.jsonl'),plan_sha256=sha(p))
    calls=[]
    monkeypatch.setattr(module,'partitions',lambda d:(d['records'],[],{}))
    monkeypatch.setattr(module,'extract_quantitative_inputs',lambda r:{'components':[{'smiles':'C','role':0}],'temperature_c':20})
    monkeypatch.setattr(module,'extract_target',lambda r:10+10*int(r.reaction_id[-1]))
    monkeypatch.setattr(module,'load_model',lambda p:object())
    monkeypatch.setattr(module.joblib,'load',lambda p:object())
    monkeypatch.setattr(module,'legacy_graph',lambda r:r['reaction_id'])
    def predict(model,graphs,version):
        calls.extend(graphs);return np.full(len(graphs),40.)
    monkeypatch.setattr(module,'predictions',predict)
    monkeypatch.setattr(module,'forest_predictions',lambda m,r,c:np.full(len(r),50.))
    return source,manifest,calls


def test_resumed_pipeline_skips_prefix_and_accounts_entire_source(tmp_path,monkeypatch):
    source,manifest,calls=fixture(tmp_path,monkeypatch)
    result=module.evaluate(source,'results.json','data/new/predictions.jsonl',True,'plan.json',manifest,'prefix.jsonl')
    assert calls==['r1','r2']
    assert result['source_rows']==result['retained_rows']==3
    assert result['prefix_rows_reused_without_model_inference']==1
    assert result['gnn']['mae_percentage_points']==pytest.approx(40/3)
    assert Path('data/new/predictions.jsonl').read_bytes().startswith(Path('prefix.jsonl').read_bytes())
    assert len(Path('data/new/predictions.jsonl').read_text().splitlines())==3


def test_changed_prefix_target_prevents_any_tail_inference(tmp_path,monkeypatch):
    source,manifest,calls=fixture(tmp_path,monkeypatch,wrong_target=True)
    with pytest.raises(ValueError,match='source order or target'):
        module.evaluate(source,'results.json','data/new/predictions.jsonl',True,'plan.json',manifest,'prefix.jsonl')
    assert calls==[] and not Path('results.json').exists()
