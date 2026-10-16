import math
import torch
from torch_geometric.data import Data
from .data import canonical

NODE_DIM=11
EDGE_DIM=4

def reaction_graph(record):
    nodes=[];edges=[];bonds=[]
    for component in record["components"]:
        role=component["role"]
        if role not in range(4): raise ValueError("Input role must be 0..3")
        _,mol=canonical(component["smiles"]);offset=len(nodes)
        for atom in mol.GetAtoms():
            features=[atom.GetAtomicNum()/100,atom.GetDegree()/4,atom.GetFormalCharge()/4,atom.GetTotalNumHs()/4,float(atom.GetIsAromatic()),float(atom.IsInRing()),atom.GetMass()/200]+[float(role==i) for i in range(4)]
            nodes.append(features)
        for bond in mol.GetBonds():
            a,b=bond.GetBeginAtomIdx()+offset,bond.GetEndAtomIdx()+offset
            attributes=[bond.GetBondTypeAsDouble()/3,float(bond.GetIsConjugated()),float(bond.IsInRing()),float(bond.GetIsAromatic())]
            edges.extend([(a,b),(b,a)]);bonds.extend([attributes,attributes])
    if not nodes: raise ValueError("Empty reaction graph")
    temperature=record.get("temperature_c")
    if temperature is not None and not math.isfinite(float(temperature)):
        raise ValueError("temperature_c must be finite or null")
    graph=Data(x=torch.tensor(nodes,dtype=torch.float32),edge_index=torch.tensor(edges,dtype=torch.long).t().contiguous() if edges else torch.empty((2,0),dtype=torch.long),edge_attr=torch.tensor(bonds,dtype=torch.float32).reshape(-1,EDGE_DIM),conditions=torch.tensor([[temperature/100 if temperature is not None else 0.,float(temperature is None)]],dtype=torch.float32))
    if "yield_percent" in record: graph.y=torch.tensor([record["yield_percent"]/100],dtype=torch.float32)
    return graph
