from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import urllib.request
import pyarrow.parquet as pq
from ord_schema.proto import reaction_pb2 as pb
from rdkit import Chem

DATASET_ID="ord_dataset-00005539a1e04c809a9a78647bea649c"
REVISION="83f971f586f6ad18f358ae4ae99d045e94ed2066"
EXPECTED_SHA256="bf1686ca6edac300a61acb5a5ee006ab5fd61007962ae19062b3d3c7879f84a9"
PATH=f"data/00/{DATASET_ID}.parquet"
ROLES={pb.ReactionRole.REACTANT:0,pb.ReactionRole.REAGENT:1,pb.ReactionRole.SOLVENT:2,pb.ReactionRole.CATALYST:3}

def fetch(output):
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    url=f"https://huggingface.co/datasets/open-reaction-database/ord-data/resolve/main/{PATH}"
    request=urllib.request.Request(url,headers={"User-Agent":"reaction-yield-gnn/0.1"})
    with urllib.request.urlopen(request,timeout=90) as response: content=response.read()
    digest=hashlib.sha256(content).hexdigest()
    # Immutable expected digest prevents accepting upstream drift at a mutable mirror URL.
    if digest!=EXPECTED_SHA256: raise ValueError("ORD mirror content changed; review provenance before updating hash")
    output.write_bytes(content)
    return {"dataset_id":DATASET_ID,"github_revision":REVISION,"sha256":digest,"bytes":len(content),"url":url,"license":"CC-BY-SA-4.0 dataset; upstream ord-data code Apache-2.0"}

def canonical(smiles):
    molecule=Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms()==0: raise ValueError("Invalid input SMILES")
    for atom in molecule.GetAtoms(): atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(molecule),molecule

def scaffold_group(components):
    reactants=[canonical(c["smiles"]) for c in components if c["role"]==0]
    if not reactants: raise ValueError("No reactant input")
    # Predeclared substrate rule: largest heavy-atom reactant, tie break by canonical SMILES.
    smi,mol=max(reactants,key=lambda pair:(pair[1].GetNumHeavyAtoms(),pair[0]))
    scaffold=Chem.MolToSmiles(Chem.MurckoDecompose(mol),isomericSmiles=False)
    return scaffold or smi

def extract_inputs(reaction):
    """Allow-list inputs and temperature setpoint; never read outcome-derived features."""
    components=[]
    for key in sorted(reaction.inputs):
        for component in reaction.inputs[key].components:
            if component.reaction_role not in ROLES: continue
            identifiers=[i.value for i in component.identifiers if i.type==pb.CompoundIdentifier.SMILES]
            if not identifiers:
                names=[i.value.strip().lower() for i in component.identifiers if i.type==pb.CompoundIdentifier.NAME]
                if names and all(name in {"none","no catalyst","no ligand"} for name in names): continue
                raise ValueError("Allowed-role component lacks SMILES")
            smi,_=canonical(identifiers[0])
            components.append({"smiles":smi,"role":ROLES[component.reaction_role]})
    if not components: raise ValueError("No usable components")
    temperature=None
    if reaction.conditions.temperature.HasField("setpoint") and reaction.conditions.temperature.setpoint.HasField("value"):
        setpoint=reaction.conditions.temperature.setpoint
        value=float(setpoint.value)
        if setpoint.units==pb.Temperature.CELSIUS: temperature=value
        elif setpoint.units==pb.Temperature.KELVIN: temperature=value-273.15
        elif setpoint.units==pb.Temperature.FAHRENHEIT: temperature=(value-32)*5/9
        if temperature is not None and not math.isfinite(temperature): raise ValueError("Nonfinite temperature")
    return {"components":components,"temperature_c":temperature}

def extract_target(reaction):
    yields=[]
    for outcome in reaction.outcomes:
        for product in outcome.products:
            if not product.is_desired_product: continue
            for measurement in product.measurements:
                if measurement.type==pb.ProductMeasurement.YIELD and measurement.HasField("percentage") and measurement.percentage.HasField("value"):
                    yields.append(float(measurement.percentage.value))
    if len(yields)!=1 or not math.isfinite(yields[0]) or not 0<=yields[0]<=100:
        raise ValueError("Need one unambiguous desired-product percentage yield in [0,100]")
    return yields[0]

def prepare(path,output):
    table=pq.read_table(path,columns=["reaction_id","reaction"])
    records=[];dropped=Counter()
    for item in table.to_pylist():
        reaction=pb.Reaction();reaction.ParseFromString(item["reaction"])
        try:
            inputs=extract_inputs(reaction);target=extract_target(reaction)
            group=scaffold_group(inputs["components"])
            records.append({"reaction_id":item["reaction_id"],**inputs,"yield_percent":target,"group":group})
        except ValueError as error: dropped[str(error)]+=1
    result={"dataset_id":DATASET_ID,"records":records,"raw_rows":len(table),"dropped":dict(dropped),"feature_contract":"input molecular SMILES/roles and temperature setpoint only"}
    destination=Path(output);destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result

def split_records(records,seed=42):
    groups=sorted({r["group"] for r in records})
    if len(groups)<7: raise ValueError("At least seven distinct substrate scaffold groups required")
    random.Random(seed).shuffle(groups)
    a=max(1,int(.7*len(groups)));b=max(a+1,int(.85*len(groups)))
    assignment={g:("train" if i<a else "validation" if i<b else "test") for i,g in enumerate(groups)}
    result={s:[r for r in records if assignment[r["group"]]==s] for s in ["train","validation","test"]}
    if any(not rows for rows in result.values()): raise ValueError("Empty split")
    return result


def input_signature(record):
    return tuple(sorted((c["role"],canonical(c["smiles"])[0]) for c in record["components"])),record.get("temperature_c")

def audit_split_integrity(splits):
    signatures={s:{input_signature(r) for r in rows} for s,rows in splits.items()}
    groups={s:{r["group"] for r in rows} for s,rows in splits.items()}
    for a in splits:
        for b in splits:
            if a>=b:continue
            if groups[a]&groups[b]:raise ValueError("Scaffold overlap across splits")
            if signatures[a]&signatures[b]:raise ValueError("Exact input/condition clone crosses splits")
    return {"scaffold_overlap":0,"exact_input_clone_overlap":0}
