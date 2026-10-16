from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
from rdkit import Chem,DataStructs
import torch
from torch_geometric.loader import DataLoader
from .data import split_records
from .graph import reaction_graph
from .model import YieldGNN


def metrics(target,prediction):
    target=np.asarray(target,dtype=float);prediction=np.asarray(prediction,dtype=float)
    if target.ndim!=1 or not len(target) or target.shape!=prediction.shape or not np.isfinite(target).all() or not np.isfinite(prediction).all():raise ValueError("Aligned nonempty finite target/prediction vectors required")
    residual=target-prediction
    total=float(((target-target.mean())**2).sum())
    return {"mae_percentage_points":float(np.abs(residual).mean()),"rmse_percentage_points":float(np.sqrt((residual**2).mean())),"r2":float(1-(residual**2).sum()/total) if total else None,"rows":len(target)}

def fingerprint(record):
    values=np.zeros(4*256+2,dtype=np.float32)
    for component in record["components"]:
        bitvector=Chem.RDKFingerprint(Chem.MolFromSmiles(component["smiles"]),fpSize=256)
        bits=np.zeros(256,dtype=np.float32);DataStructs.ConvertToNumpyArray(bitvector,bits)
        offset=component["role"]*256
        values[offset:offset+256]=np.maximum(values[offset:offset+256],bits)
    temp=record.get("temperature_c");values[-2:]=[temp/100 if temp is not None else 0.,float(temp is None)]
    return values

def train(path,checkpoint,report,epochs=30,seed=42,hidden=48,*,acknowledge_observed=False):
    from .artifacts import observed_outputs,train_outputs,write_text_new,save_torch_new
    observed_outputs(acknowledge_observed,'ELN',train_outputs(report,checkpoint))
    if epochs<1: raise ValueError("epochs must be positive")
    from sklearn.ensemble import RandomForestRegressor
    torch.set_num_threads(2);torch.manual_seed(seed);np.random.seed(seed)
    started=time.perf_counter();raw=Path(path).read_bytes();dataset=json.loads(raw)
    splits=split_records(dataset["records"],seed)
    graphs={split:[reaction_graph(row) for row in rows] for split,rows in splits.items()}
    loader=DataLoader(graphs["train"],batch_size=24,shuffle=True,generator=torch.Generator().manual_seed(seed))
    model=YieldGNN(hidden);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    history=[];best_mae=float("inf");best=None;best_epoch=0
    def predict(split):
        model.eval()
        with torch.no_grad():
            return torch.cat([model(batch) for batch in DataLoader(graphs[split],batch_size=32)]).numpy()*100
    targets={s:np.array([r["yield_percent"] for r in rows]) for s,rows in splits.items()}
    for epoch in range(epochs):
        model.train();loss_sum=0.
        for batch in loader:
            output=model(batch);loss=torch.nn.functional.mse_loss(output,batch.y)
            if not torch.isfinite(loss): raise ValueError("Nonfinite training loss")
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            loss_sum+=float(loss.detach())*batch.num_graphs
        val=metrics(targets["validation"],predict("validation"))["mae_percentage_points"]
        history.append({"epoch":epoch+1,"train_mse_fraction":loss_sum/len(graphs["train"]),"validation_mae_percentage_points":val})
        if val<best_mae: best_mae=val;best=deepcopy(model.state_dict());best_epoch=epoch+1
    model.load_state_dict(best)
    test_prediction=predict("test")
    median=float(np.median(targets["train"]))
    forest=RandomForestRegressor(n_estimators=100,min_samples_leaf=3,random_state=seed,n_jobs=2)
    forest.fit(np.stack([fingerprint(r) for r in splits["train"]]),targets["train"])
    forest_prediction=forest.predict(np.stack([fingerprint(r) for r in splits["test"]]))
    checkpoint=Path(checkpoint);checkpoint.parent.mkdir(parents=True,exist_ok=True)
    save_torch_new({"state_dict":best,"hidden":hidden,"feature_version":1,"dataset_id":dataset["dataset_id"]},checkpoint)
    try:
        revision=subprocess.check_output(["git","rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
        dirty=bool(subprocess.check_output(["git","status","--porcelain"],text=True).strip())
    except subprocess.CalledProcessError: revision=None;dirty=None
    result={"dataset_id":dataset["dataset_id"],"processed_sha256":hashlib.sha256(raw).hexdigest(),"seed":seed,"epochs":epochs,"best_epoch_selected_on_validation":best_epoch,"feature_contract":dataset["feature_contract"],"split_rule":"largest-reactant Murcko scaffold; acyclic fallback canonical SMILES; group-heldout 70/15/15 by group count", "splits":{s:{"rows":len(rows),"scaffold_groups":len({r["group"] for r in rows}),"yield_mean":float(targets[s].mean()),"yield_std":float(targets[s].std())} for s,rows in splits.items()},"test_gnn":metrics(targets["test"],test_prediction),"test_training_median":metrics(targets["test"],np.full(len(targets["test"]),median)),"test_role_fingerprint_random_forest":metrics(targets["test"],forest_prediction),"history":history,"environment":{"python":platform.python_version(),"torch":torch.__version__,"device":"cpu","threads":2},"code_revision":revision,"code_dirty_at_run":dirty,"runtime_seconds":time.perf_counter()-started,"limitations":["Single 750-row ELN source; no independent laboratory or prospective validation.","Grouping protects largest-reactant scaffolds only; smaller reactants/reagents may overlap.","No product structures, notes, analytical outcomes, yield measurements or outcome reaction time enter model features.","Input quantities and detailed procedures are omitted; conditions include only temperature setpoint.","A measured model is not necessarily better than tabular or constant baselines."]}
    destination=Path(report);destination.parent.mkdir(parents=True,exist_ok=True);write_text_new(destination,json.dumps(result,indent=2,allow_nan=False)+"\n")
    split_ids={s:[{"reaction_id":r["reaction_id"],"group_sha256":hashlib.sha256(r["group"].encode()).hexdigest()} for r in rows] for s,rows in splits.items()}
    write_text_new(destination.with_name(destination.stem+"-split-ids.json"),json.dumps(split_ids,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="history"},indent=2));return result

def load_model(path):
    checkpoint=torch.load(path,map_location="cpu",weights_only=True)
    kind=checkpoint.get("model_kind")
    if kind is not None:
        if kind!="descriptor_gnn":raise ValueError("Unsupported checkpoint model kind")
        from .descriptor_study import from_checkpoint
        return from_checkpoint(checkpoint)
    version=checkpoint["feature_version"]
    if version==1:
        model=YieldGNN(checkpoint["hidden"],checkpoint.get("layers",3),checkpoint.get("pooling","global"),checkpoint.get("dropout",0.))
    elif version==2:
        from .quantity_model import QuantityGNN
        model=QuantityGNN(checkpoint["hidden"],checkpoint.get("layers",3),checkpoint.get("pooling","component_role"),checkpoint.get("dropout",0.))
    else:raise ValueError("Unsupported graph feature version")
    model.load_state_dict(checkpoint["state_dict"]);model.eval()
    return model

def predict_record(model,record):
    from torch_geometric.data import Batch
    if getattr(model,'feature_version',1)==3:
        from .descriptor_model import reaction_graph as descriptor_graph
        batch=Batch.from_data_list([descriptor_graph(record,model.include_descriptors)])
    elif getattr(model,'feature_version',1)==2:
        from .quantity_graph import reaction_graph as quantity_graph
        batch=Batch.from_data_list([quantity_graph(record)],follow_batch=['component_roles'])
    else:batch=Batch.from_data_list([reaction_graph(record)])
    with torch.no_grad():value=float(model(batch)[0])*100
    if not np.isfinite(value) or not 0<=value<=100:raise ValueError('Nonfinite or out-of-range model prediction')
    return value
