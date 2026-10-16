"""Independent-source evaluation, after model selection, without adaptation."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import urllib.request
import numpy as np
import pyarrow.parquet as pq
from ord_schema.proto import reaction_pb2 as pb
import torch
from .data import canonical,extract_inputs,extract_target,scaffold_group,input_signature,split_records
from .graph import reaction_graph
from .optimize import predictions
from .train import load_model,metrics,fingerprint

TRANSFER_ID="ord_dataset-805ad863feef48579d95d86a728035f4"
TRANSFER_SHA="31b7c4762145e5dad973050373cd967bdd85caf3a9858767c83a17fed6c4af36"

def fetch_transfer(path):
    url=f"https://huggingface.co/datasets/open-reaction-database/ord-data/resolve/main/data/80/{TRANSFER_ID}.parquet"
    with urllib.request.urlopen(url,timeout=90) as response:content=response.read()
    if hashlib.sha256(content).hexdigest()!=TRANSFER_SHA:raise ValueError("Transfer dataset digest mismatch")
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
    return {"dataset_id":TRANSFER_ID,"sha256":TRANSFER_SHA,"bytes":len(content),"url":url,"license":"CC-BY-SA-4.0"}

def evaluate_transfer(path,checkpoint,development,report,limit=2048,selection_report="docs/optimized-results.json"):
    if limit<1:raise ValueError("limit must be positive")
    torch.set_num_threads(2)
    content=Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest()!=TRANSFER_SHA:raise ValueError("Transfer digest mismatch")
    table=pq.read_table(path,columns=["reaction_id","reaction"])
    # Hash selection uses identifiers only, before looking at outcomes or labels.
    items=sorted(table.to_pylist(),key=lambda item:hashlib.sha256(item["reaction_id"].encode()).hexdigest())[:limit]
    training=json.loads(Path(development).read_text(encoding="utf-8"))["records"]
    seen={input_signature(r) for r in training};records=[];dropped=Counter()
    for item in items:
        reaction=pb.Reaction();reaction.ParseFromString(item["reaction"])
        try:
            record={"reaction_id":item["reaction_id"],**extract_inputs(reaction),"yield_percent":extract_target(reaction)}
            record["group"]=scaffold_group(record["components"])
            if input_signature(record) in seen:raise ValueError("Exact input clone from ELN cohort")
            records.append(record)
        except ValueError as error:dropped[str(error)]+=1
    if not records:raise ValueError("No valid transfer rows")
    model=load_model(checkpoint);target=np.array([r["yield_percent"] for r in records])
    score=metrics(target,predictions(model,[reaction_graph(r) for r in records]))
    selection=split_records(training,42)
    development_rows=selection["train"]+selection["validation"]
    median=float(np.median([r["yield_percent"] for r in development_rows]))
    from sklearn.ensemble import RandomForestRegressor
    selected=json.loads(Path(selection_report).read_text(encoding="utf-8"))
    forest=RandomForestRegressor(n_estimators=200,min_samples_leaf=selected["selected_forest"]["min_samples_leaf"],random_state=42,n_jobs=2)
    forest.fit(np.stack([fingerprint(r) for r in development_rows]),[r["yield_percent"] for r in development_rows])
    forest_metrics=metrics(target,forest.predict(np.stack([fingerprint(r) for r in records])))
    groups={r["group"] for r in records};training_groups={r["group"] for r in training}
    result={"dataset_id":TRANSFER_ID,"sha256":TRANSFER_SHA,"selection":"lowest SHA256(reaction_id), no target-based sampling","candidate_rows":len(items),"retained_rows":len(records),"dropped":dict(dropped),"test_gnn":score,"test_role_fingerprint_random_forest":forest_metrics,"test_eln_median":metrics(target,np.full(len(target),median)),"target_mean":float(target.mean()),"target_std":float(target.std()),"distinct_principal_scaffolds":len(groups),"principal_scaffolds_overlapping_eln":len(groups&training_groups),"exact_input_clones_overlapping_eln":0,"interpretation":"Separate-source assay-yield transfer only; substrate scaffolds may overlap and this narrow2amine4aryl dataset is not a broad chemistry benchmark. No tuning or adaptation uses this cohort."}
    destination=Path(report);destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2));return result
