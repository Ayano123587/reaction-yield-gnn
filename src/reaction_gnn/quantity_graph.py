"""Quantity-aware graph version2 with explicit component boundaries."""
import math
import numpy as np
import torch
from .graph import reaction_graph as legacy_graph

NODE_DIM=13
CONDITION_DIM=16


def quantitative_conditions(record):
    components=record['components'];moles=[];volumes=[]
    for c in components:
        if isinstance(c['role'],bool) or not isinstance(c['role'],int) or c['role'] not in range(4):raise ValueError('Input role must be 0..3')
        if not isinstance(c.get('is_limiting',False),bool):raise ValueError('is_limiting must be boolean')
        values=[]
        for key in ['moles','volume_l']:
            value=c.get(key)
            if value is not None and (isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0):raise ValueError('Input amount must be finite nonnegative or null')
            values.append(value)
        moles.append(values[0]);volumes.append(values[1])
    reference=[m for c,m in zip(components,moles) if c['role']==0 and c.get('is_limiting',False) and m is not None and m>0]
    if not reference:reference=[m for c,m in zip(components,moles) if c['role']==0 and m is not None and m>0]
    reference=min(reference) if reference else None
    equivalents=[m/reference if m is not None and reference is not None else None for m in moles]
    if any(v is not None and not math.isfinite(v) for v in equivalents):raise ValueError('Nonfinite stoichiometric ratio')
    def scaled(value):return math.log1p(min(value,1000.))/math.log(1001.)
    temperature=record.get('temperature_c')
    if temperature is not None and (not isinstance(temperature,(int,float)) or isinstance(temperature,bool) or not math.isfinite(temperature)):raise ValueError('Finite temperature required')
    condition=[float(temperature)/100 if temperature is not None else 0.,float(temperature is None)]
    for role in range(4):condition.append(scaled(sum(v for c,v in zip(components,equivalents) if c['role']==role and v is not None)))
    for role in range(4):
        selected=[v for c,v in zip(components,equivalents) if c['role']==role]
        condition.append(sum(v is None for v in selected)/len(selected) if selected else 1.)
    for role in range(4):condition.append(math.log1p(sum(c['role']==role for c in components))/math.log(11.))
    volume=sum(v for v in volumes if v is not None)
    concentration=reference/volume if reference is not None and volume>0 else None
    if concentration is not None and not math.isfinite(concentration):raise ValueError('Nonfinite input concentration')
    condition.extend([scaled(concentration) if concentration is not None else 0.,float(concentration is None)])
    if len(condition)!=CONDITION_DIM or not np.isfinite(condition).all():raise ValueError('Nonfinite quantity conditions')
    node_quantities=[[scaled(v) if v is not None else 0.,float(v is None)] for v in equivalents]
    return np.asarray(condition,dtype=np.float32),node_quantities


def reaction_graph(record):
    conditions,quantities=quantitative_conditions(record);graph=legacy_graph(record)
    from .data import canonical
    local=[];amounts=[]
    for index,(component,quantity) in enumerate(zip(record['components'],quantities)):
        _,mol=canonical(component['smiles']);count=mol.GetNumAtoms()
        local.extend([index]*count);amounts.extend([quantity]*count)
    graph.x=torch.cat([graph.x,torch.tensor(amounts,dtype=torch.float32)],dim=1)
    graph.conditions=torch.tensor(conditions[None,:],dtype=torch.float32)
    graph.component_local=torch.tensor(local,dtype=torch.long)
    graph.component_roles=torch.tensor([c['role'] for c in record['components']],dtype=torch.long)
    return graph
