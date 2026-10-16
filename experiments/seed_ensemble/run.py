"""Readable private Kaggle CPU runner. Source/data are a private mounted dataset."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time

# Version 1 is smoke-only; full training is submitted only after smoke evidence review.
MODE = 'smoke'
subprocess.run([sys.executable,'-m','pip','install','-q','torch-geometric==2.8.1','ord-schema==0.9.3','rdkit','pyarrow'],check=True)
mount=next(p.parent for p in Path('/kaggle/input').rglob('ensemble-payload.json'))
import shutil
source=Path('/kaggle/working/source/reaction_gnn');source.mkdir(parents=True,exist_ok=True)
for module in mount.glob('*.py'):shutil.copyfile(module,source/module.name)
sys.path.insert(0,str(source.parent))
import numpy as np
import torch
from reaction_gnn.graph import reaction_graph
from reaction_gnn.quantity_study import fit_graph,predictions,save_model
from reaction_gnn.train import load_model,metrics
from reaction_gnn.seed_ensemble import decide

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=json.loads((mount/'manifest.json').read_text())
for relative,digest in manifest['files'].items():
    assert sha(mount/relative)==digest,relative
payload=json.loads((mount/'ensemble-payload.json').read_text())
records=payload['development_records'];folds=payload['fold_assignment']
assert len(records)==610 and len({r['reaction_id'] for r in records})==610
assert not set(folds)-{r['group'] for r in records}
config={'name':'legacy_role48','feature_version':1,'pooling':'role','loss':'mse'}
fixed=payload['fixed_training']
torch.set_num_threads(2)
out=Path('/kaggle/working');started=time.monotonic()
graphs={r['reaction_id']:reaction_graph(r) for r in records}
report={'mode':MODE,'manifest_sha256':sha(mount/'manifest.json'),
        'plan_sha256':sha(mount/'ensemble-plan.json'),
        'environment':{'python':platform.python_version(),'torch':torch.__version__,
                       'packages':{p:importlib.metadata.version(p) for p in ['torch-geometric','rdkit','ord-schema','numpy']},
                       'device':'cpu','threads':2},'folds':[]}
for fold in range(1 if MODE=='smoke' else 3):
    training=[r for r in records if folds[r['group']]!=fold]
    validation=[r for r in records if folds[r['group']]==fold]
    baseline_rows=json.loads((mount/f'fold{fold}-predictions.json').read_text())
    baseline={r['reaction_id']:r['prediction'] for r in baseline_rows}
    assert set(baseline)=={r['reaction_id'] for r in validation}
    target=np.array([r['yield_percent'] for r in validation])
    frozen=np.array([baseline[r['reaction_id']] for r in validation])
    saved=load_model(mount/f'fold{fold}.pt')
    replay=predictions(saved,[graphs[r['reaction_id']] for r in validation],1)
    replay_delta=float(np.max(np.abs(replay-frozen)))
    assert replay_delta<0.001,replay_delta
    item={'fold':fold,'training_rows':len(training),'validation_rows':len(validation),
          'baseline_replay_max_delta_pp':replay_delta,'baseline_mae':metrics(target,frozen)['mae_percentage_points'],'seeds':[]}
    values=[frozen]
    for seed in ([7] if MODE=='smoke' else [7,101]):
        tr=training[:32] if MODE=='smoke' else training
        va=validation[:8] if MODE=='smoke' else validation
        y=np.array([r['yield_percent'] for r in va])
        model,epoch,history=fit_graph([graphs[r['reaction_id']] for r in tr],
            ([graphs[r['reaction_id']] for r in va],y),config,fixed,seed+fold,
            epochs=2 if MODE=='smoke' else None)
        prediction=predictions(model,[graphs[r['reaction_id']] for r in va],1)
        checkpoint=out/f'seed{seed}-fold{fold}.pt';save_model(model,config,fixed,checkpoint)
        reload=predictions(load_model(checkpoint),[graphs[r['reaction_id']] for r in va],1)
        assert np.array_equal(prediction,reload)
        assert all(np.isfinite(h['train_loss_fraction']) and np.isfinite(h['maximum_gradient_norm_before_clip']) for h in history)
        item['seeds'].append({'seed':seed+fold,'fitted_training_rows':len(tr),'scored_validation_rows':len(va),'best_epoch':epoch,'history':history,
             'mae':metrics(y,prediction)['mae_percentage_points'],'checkpoint_sha256':sha(checkpoint),'reload_delta_pp':float(np.max(np.abs(prediction-reload)))})
        (out/f'seed{seed}-fold{fold}-predictions.json').write_text(json.dumps([
             {'reaction_id':r['reaction_id'],'target':float(t),'prediction':float(p)} for r,t,p in zip(va,y,prediction)]))
        if MODE!='smoke':values.append(prediction)
    if MODE!='smoke':
        combined=np.mean(values,axis=0)
        item['ensemble_mae']=metrics(target,combined)['mae_percentage_points']
    report['folds'].append(item)
    print(json.dumps({k:v for k,v in item.items() if k!='seeds'}),flush=True)
if MODE!='smoke':report['selection']=decide([f['baseline_mae'] for f in report['folds']],[f['ensemble_mae'] for f in report['folds']])
report['runtime_seconds']=time.monotonic()-started
(out/'aggregate.json').write_text(json.dumps(report,indent=2,allow_nan=False))
(out/'completion.json').write_text(json.dumps({'status':'complete','mode':MODE,'aggregate_sha256':sha(out/'aggregate.json')}))
print('Complete:',MODE,report['runtime_seconds'],flush=True)
