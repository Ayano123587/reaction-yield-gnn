"""Validation-only controlled ablations, with final held-out evaluation after selection."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from torch_geometric.loader import DataLoader
from .data import split_records,audit_split_integrity
from .graph import reaction_graph
from .model import YieldGNN
from .train import metrics,fingerprint

CONFIGS=[
 {"name":"global48","hidden":48,"layers":3,"pooling":"global","dropout":0.,"lr":.001},
 {"name":"role48","hidden":48,"layers":3,"pooling":"role","dropout":0.,"lr":.001},
 {"name":"role96","hidden":96,"layers":3,"pooling":"role","dropout":0.,"lr":.001},
 {"name":"role48_fast","hidden":48,"layers":3,"pooling":"role","dropout":0.,"lr":.003},
]

def predictions(model,graphs):
    model.eval()
    with torch.no_grad():return torch.cat([model(batch) for batch in DataLoader(graphs,batch_size=32)]).numpy()*100

def fit(graphs,validation,config,epochs,seed,patience=10):
    torch.manual_seed(seed)
    model=YieldGNN(config["hidden"],config["layers"],config["pooling"],config["dropout"])
    optimizer=torch.optim.AdamW(model.parameters(),lr=config["lr"],weight_decay=.0001)
    loader=DataLoader(graphs,batch_size=24,shuffle=True,generator=torch.Generator().manual_seed(seed))
    best=float("inf");state=None;best_epoch=0;history=[]
    for epoch in range(epochs):
        model.train();total=0.
        for batch in loader:
            loss=torch.nn.functional.mse_loss(model(batch),batch.y)
            if not bool(torch.isfinite(loss)):raise ValueError("Nonfinite training loss")
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            total+=float(loss.detach())*batch.num_graphs
        if validation:
            val_graphs,val_y=validation
            score=metrics(val_y,predictions(model,val_graphs))["mae_percentage_points"]
            history.append({"epoch":epoch+1,"train_mse_fraction":total/len(graphs),"validation_mae":score})
            if score<best:
                best=score;state=deepcopy(model.state_dict());best_epoch=epoch+1
            if epoch+1>=20 and epoch+1-best_epoch>=patience:break
        else:history.append({"epoch":epoch+1,"train_mse_fraction":total/len(graphs)})
    if validation:model.load_state_dict(state)
    return model,best_epoch if validation else epochs,history

def optimize(path,output="docs/optimized-results.json",checkpoint="models/optimized.pt",epochs=60,folds=3,*,acknowledge_observed=False):
    from .artifacts import observed_outputs,optimize_outputs,write_text_new,save_torch_new
    observed_outputs(acknowledge_observed,'ELN',optimize_outputs(output,checkpoint))
    if epochs<1 or folds<2:raise ValueError("positive epochs and >=2 folds required")
    torch.set_num_threads(2);np.random.seed(42);started=time.perf_counter()
    raw=Path(path).read_bytes();dataset=json.loads(raw);splits=split_records(dataset["records"],42)
    integrity=audit_split_integrity(splits)
    development=splits["train"]+splits["validation"]
    groups=sorted({r["group"] for r in development});np.random.default_rng(42).shuffle(groups)
    fold_assignment={g:i%folds for i,g in enumerate(groups)}
    graph_cache={r["reaction_id"]:reaction_graph(r) for r in dataset["records"]}
    fingerprints={r["reaction_id"]:fingerprint(r) for r in dataset["records"]}
    destination=Path(output);destination.parent.mkdir(parents=True,exist_ok=True)
    lock={"processed_sha256":hashlib.sha256(raw).hexdigest(),"test_ids":[r["reaction_id"] for r in splits["test"]],"configs":CONFIGS,"max_epochs":epochs,"folds":folds,"seed":42,"test_policy":"No final-test predictions during model/config selection. An earlier2epoch exploratory smoke did report aggregate test metrics; this historical exposure is disclosed."}
    write_text_new(destination.with_name("selection-protocol.json"),json.dumps(lock,indent=2)+"\n")
    write_text_new(destination.with_name("selection-progress.jsonl"),"")
    try:revision=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    except subprocess.CalledProcessError:revision=None
    trials=[]
    for config in CONFIGS:
        scores=[];best_epochs=[]
        for fold in range(folds):
            training=[r for r in development if fold_assignment[r["group"]]!=fold]
            validation=[r for r in development if fold_assignment[r["group"]]==fold]
            audit_split_integrity({"train":training,"validation":validation,"test":splits["test"]})
            train_graphs=[graph_cache[r["reaction_id"]] for r in training];val_graphs=[graph_cache[r["reaction_id"]] for r in validation]
            targets=np.array([r["yield_percent"] for r in validation])
            model,best_epoch,history=fit(train_graphs,(val_graphs,targets),config,epochs,42+fold)
            score=metrics(targets,predictions(model,val_graphs))["mae_percentage_points"]
            scores.append(score);best_epochs.append(best_epoch)
            partial={"config":config["name"],"fold":fold,"validation_mae":score,"best_epoch":best_epoch,"executed_epochs":len(history)}
            print(json.dumps(partial),flush=True)
            with destination.with_name("selection-progress.jsonl").open("a",encoding="utf-8") as progress:
                progress.write(json.dumps(partial)+"\n")
        trials.append({"config":config,"fold_validation_mae":scores,"mean_validation_mae":float(np.mean(scores)),"best_epochs":best_epochs})
    selected=min(trials,key=lambda t:t["mean_validation_mae"])
    forest_trials=[]
    for leaf in [1,3,8]:
        scores=[]
        for fold in range(folds):
            tr=[r for r in development if fold_assignment[r["group"]]!=fold];va=[r for r in development if fold_assignment[r["group"]]==fold]
            forest=RandomForestRegressor(n_estimators=200,min_samples_leaf=leaf,random_state=42,n_jobs=2)
            forest.fit(np.stack([fingerprints[r["reaction_id"]] for r in tr]),[r["yield_percent"] for r in tr])
            scores.append(metrics([r["yield_percent"] for r in va],forest.predict(np.stack([fingerprints[r["reaction_id"]] for r in va])))["mae_percentage_points"])
        forest_trials.append({"min_samples_leaf":leaf,"fold_validation_mae":scores,"mean_validation_mae":float(np.mean(scores))})
    forest_config=min(forest_trials,key=lambda t:t["mean_validation_mae"])
    final_epochs=max(1,round(float(np.median(selected["best_epochs"]))))
    model,_,history=fit([graph_cache[r["reaction_id"]] for r in development],None,selected["config"],final_epochs,42)
    path=Path(checkpoint);path.parent.mkdir(parents=True,exist_ok=True)
    config=selected["config"]
    save_torch_new({"state_dict":model.state_dict(),"hidden":config["hidden"],"layers":config["layers"],"pooling":config["pooling"],"dropout":config["dropout"],"feature_version":1,"dataset_id":dataset["dataset_id"]},path)
    # The held-out test is first used for predictions only after all selection/refitting.
    y=np.array([r["yield_percent"] for r in splits["test"]]);gnn=predictions(model,[graph_cache[r["reaction_id"]] for r in splits["test"]])
    forest=RandomForestRegressor(n_estimators=200,min_samples_leaf=forest_config["min_samples_leaf"],random_state=42,n_jobs=2)
    forest.fit(np.stack([fingerprints[r["reaction_id"]] for r in development]),[r["yield_percent"] for r in development])
    forest_pred=forest.predict(np.stack([fingerprints[r["reaction_id"]] for r in splits["test"]]))
    median=float(np.median([r["yield_percent"] for r in development]))
    result={"dataset_id":dataset["dataset_id"],"processed_sha256":lock["processed_sha256"],"code_revision":revision,"seed":42,"protocol":lock["test_policy"],"selection":"3fold principal-substrate-scaffold-disjoint validation; fixed final test; refit all development rows with median validation best epoch", "split_integrity":integrity,"development_rows":len(development),"final_test_rows":len(y),"gnn_trials":trials,"forest_trials":forest_trials,"selected_gnn":config,"selected_forest":forest_config,"final_refit_epochs":final_epochs,"final_refit_history":history,"test_gnn":metrics(y,gnn),"test_training_median":metrics(y,np.full(len(y),median)),"test_role_fingerprint_random_forest":metrics(y,forest_pred),"runtime_seconds":time.perf_counter()-started,"environment":{"python":platform.python_version(),"torch":torch.__version__,"device":"cpu","threads":2},"limitations":["Earlier exploratory smoke already exposed an aggregate on this test; final selection used validation only, but a genuinely untouched external source evaluation is still needed.","740 usable ELN reactions from one source; this is not a universal reaction predictor.","Largest-reactant scaffolds and exact input clones are disjoint; smaller substrates and reagents can overlap.","Mean3fold validation is noisy and no best-achievable-global-score claim is made."]}
    write_text_new(destination,json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:result[k] for k in ["selected_gnn","final_refit_epochs","test_gnn","test_training_median","test_role_fingerprint_random_forest","runtime_seconds"]},indent=2),flush=True)
    return result
