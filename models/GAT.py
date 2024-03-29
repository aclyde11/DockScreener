import dgl
import dgl.nn.pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F

class GAT(nn.Module):
    def __init__(self, in_dim, edge_feats):
        super(GAT, self).__init__()
        in_feats = in_dim
        out_feats = 128

        self.edge1 = nn.Linear(edge_feats, in_feats * in_feats)
        self.edge2 = nn.Linear(edge_feats, in_feats * in_feats)
        self.edge1a = nn.Linear(edge_feats, in_feats * in_feats)
        self.edge2a = nn.Linear(edge_feats, in_feats * in_feats)

        self.gvo1 = dgl.nn.pytorch.conv.NNConv(in_feats, in_feats, self.edge1, aggregator_type='sum')
        self.bn1 = nn.BatchNorm1d(in_feats)
        self.gvo1a = dgl.nn.pytorch.conv.NNConv(in_feats, in_feats, self.edge1a, aggregator_type='sum')
        self.bn1a = nn.BatchNorm1d(in_feats)


        self.gvo2 = dgl.nn.pytorch.conv.NNConv(in_feats, in_feats, self.edge2, aggregator_type='sum')
        self.bn2 = nn.BatchNorm1d(in_feats)

        self.gvo2a = dgl.nn.pytorch.conv.NNConv(in_feats, in_feats, self.edge2a, aggregator_type='sum')
        self.bn2a = nn.BatchNorm1d(in_feats)

        self.gvonc = dgl.nn.pytorch.GATConv(in_feats, out_feats, num_heads=8)

        self.final_layer = nn.Sequential(
            nn.Linear(out_feats * 8, 64),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(32, 1)
        )
        self.pooling2 = dgl.nn.pytorch.glob.MaxPooling()

        self.gate_nn = nn.Sequential(
            nn.Linear(out_feats * 8, 1)
        )
        self.pooling = dgl.nn.pytorch.glob.GlobalAttentionPooling(gate_nn=self.gate_nn)

    '''
    g: DglGraph
    n: node feature matrix
    e: edge feature matrix
    '''

    def forward(self, g, n, e, return_fp=True):
        h = self.gvo1(g, n, e)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.gvo1a(g, h, e)
        h = self.bn1a(h)
        h = F.relu(h)
        h = self.gvo2(g, h, e)
        h = self.bn2(h)
        h = F.relu(h)
        h = self.gvo2a(g, h, e)
        h = self.bn2a(h)
        h = F.relu(h)
        h = self.gvonc(g, h)  # returns [nodes, out_features, heads]
        h = F.elu(h)
        h = h.view(h.shape[0], -1)

        h1 = self.pooling(g, h)  # returns [batch, out_features]
        h_out = self.final_layer(h1)  # [batch, 1]

        return h_out, None
