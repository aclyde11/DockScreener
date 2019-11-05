import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch

class GAT(nn.Module):
    def __init__(self, in_dim, edge_feats):
        super(GAT, self).__init__()
        in_feats = in_dim
        out_feats = 32
        self.edge_layer = nn.Linear(edge_feats, edge_feats * out_feats)

        self.conv1 = dgl.nn.pytorch.conv.SAGEConv(
            in_feats=in_feats,
            out_feats=out_feats,
            aggregator_type='lstm')
        self.conv2 = dgl.nn.pytorch.conv.NNConv(
            in_feats=out_feats,
            out_feats=out_feats,
            edge_func=self.edge_layer,
            aggregator_type='sum')

        self.final_layer = nn.Sequential(
            nn.Linear(out_feats, 32),
            nn.ReLU(),
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32,1)
        )
        self.pooling = dgl.nn.pytorch.glob.MaxPooling()

        # self.gate_nn = nn.Sequential(
        #     nn.Linear(out_feats, 1)
        # )
        # self.pooling = dgl.nn.pytorch.glob.GlobalAttentionPooling(gate_nn=self.gate_nn)


    def forward(self, g, n, e):
        h = self.conv1(g,n)
        h = F.relu(h)
        h = self.conv2(g,h,e)
        h = self.pooling(g,h)
        h = F.elu(h)
        h = self.final_layer(h)
        return h
