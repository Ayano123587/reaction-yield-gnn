import torch
from torch import nn
from torch_geometric.nn import GINEConv,global_mean_pool
from .graph import NODE_DIM,EDGE_DIM

class YieldGNN(nn.Module):
    def __init__(self,hidden=48,layers=3,pooling="global",dropout=0.):
        super().__init__();self.hidden=hidden;self.layers=layers;self.pooling=pooling;self.dropout=dropout
        self.encoder=nn.Linear(NODE_DIM,hidden)
        self.convs=nn.ModuleList([GINEConv(nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,hidden)),edge_dim=EDGE_DIM) for _ in range(layers)])
        self.head=nn.Sequential(nn.Linear(hidden*(4 if pooling=="role" else 1)+2,hidden),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden,1))
    def forward(self,batch):
        x=torch.relu(self.encoder(batch.x))
        for conv in self.convs: x=torch.relu(conv(x,batch.edge_index,batch.edge_attr)+x)
        if self.pooling=="role":
            # Distinguish substrates/reagents/solvents/catalysts instead of diluting them.
            roles=batch.x[:,7:11].argmax(1);pooled=[]
            for role in range(4):
                selected=roles==role
                pooled.append(global_mean_pool(x[selected],batch.batch[selected],size=batch.num_graphs))
            pooled=torch.cat(pooled,dim=1)
        else: pooled=global_mean_pool(x,batch.batch)
        return torch.sigmoid(self.head(torch.cat([pooled,batch.conditions],dim=1))).flatten()
