"""Bond-aware quantitative GNN with atom or component-balanced role pooling."""
import torch
from torch import nn
from torch_geometric.nn import GINEConv,global_mean_pool
from .quantity_graph import NODE_DIM,CONDITION_DIM


class QuantityGNN(nn.Module):
    feature_version=2
    def __init__(self,hidden=48,layers=3,pooling='component_role',dropout=0.):
        super().__init__()
        if pooling not in ['atom_role','component_role']:raise ValueError('Unknown quantity pooling')
        self.hidden=hidden;self.layers=layers;self.pooling=pooling;self.dropout=dropout
        self.encoder=nn.Linear(NODE_DIM,hidden)
        self.convs=nn.ModuleList([GINEConv(nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,hidden)),edge_dim=4) for _ in range(layers)])
        self.head=nn.Sequential(nn.Linear(hidden*4+CONDITION_DIM,hidden),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden,1))

    def forward(self,batch):
        x=torch.relu(self.encoder(batch.x))
        for conv in self.convs:x=torch.relu(conv(x,batch.edge_index,batch.edge_attr)+x)
        if self.pooling=='component_role':
            index=batch.component_local+batch.component_roles_ptr[batch.batch]
            x=global_mean_pool(x,index,size=len(batch.component_roles));roles=batch.component_roles;assignment=batch.component_roles_batch
        else:roles=batch.x[:,7:11].argmax(1);assignment=batch.batch
        pooled=[]
        for role in range(4):
            selected=roles==role
            pooled.append(global_mean_pool(x[selected],assignment[selected],size=batch.num_graphs))
        pooled=torch.cat(pooled,dim=1)
        return torch.sigmoid(self.head(torch.cat([pooled,batch.conditions],dim=1))).flatten()
