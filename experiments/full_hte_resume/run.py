"""Resume full frozen inference from immutable host prefix; no training."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys
subprocess.run([sys.executable,'-m','pip','install','-q','torch-geometric==2.8.1','ord-schema==0.9.3','rdkit==2026.3.6','scikit-learn==1.8.0','pyarrow'],check=True)
mount=next(p.parent for p in Path('/kaggle/input').rglob('resume-manifest.json'))
manifest=json.loads((mount/'manifest.json').read_text())
for relative,digest in manifest['files'].items():
    assert hashlib.sha256((mount/relative).read_bytes()).hexdigest()==digest,relative
root=Path('/kaggle/working/project');package=root/'src/reaction_gnn';package.mkdir(parents=True)
for p in mount.glob('*.py'):shutil.copyfile(p,package/p.name)
paths={'reactions-quantity-v2.json':'data/processed/reactions-quantity-v2.json',
       'selection.json':'docs/loss-v4/selection.json','ELN-results.json':'docs/loss-v4/ELN-results.json',
       'full-hte-plan.json':'docs/full-hte-plan.json','selected.pt':'models/loss-v4/selected.pt',
       'forest.joblib':'models/loss-v4/forest.joblib'}
for source,dest in paths.items():
    target=root/dest;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(mount/source,target)
os.chdir(root);sys.path.insert(0,str(root/'src'))
import torch
from reaction_gnn.train import load_model,predict_record
torch.set_num_threads(2)
example={'components':[{'smiles':'Brc1ccccc1','role':0},{'smiles':'Nc1ccccc1','role':0},{'smiles':'CCO','role':2}],'temperature_c':80.}
value=predict_record(load_model(root/'models/loss-v4/selected.pt'),example)
assert abs(value-28.50843071937561)<0.001
from reaction_gnn.full_transfer_resume import evaluate
resume=json.loads((mount/'resume-manifest.json').read_text())
result=evaluate(mount/'cn-50688.parquet',root/'docs/full-hte/results.json',
                root/'data/interim/full-hte/predictions.jsonl',True,
                resume_manifest=resume,prefix_path=mount/'prefix.jsonl')
result['public_example_replay']={'predicted_yield_percent':value,'host_reference_percent':28.50843071937561,
                                 'absolute_delta_pp':abs(value-28.50843071937561)}
Path('/kaggle/working/aggregate.json').write_text(json.dumps(result,indent=2,allow_nan=False))
Path('/kaggle/working/completion.json').write_text(json.dumps({'status':'complete','source_rows':result['source_rows'],
      'retained_rows':result['retained_rows'],'prefix_rows':result['prefix_rows_reused_without_model_inference']}))
print('Full frozen source inference complete:',result['retained_rows'],flush=True)
