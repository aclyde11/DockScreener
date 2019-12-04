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

        self.gvo1 = dgl.nn.pytorch.conv.NNConv(in_feats, in_feats, self.edge1, aggregator_type='sum')
        self.gvo2 = dgl.nn.pytorch.conv.NNConv(in_feats, in_feats, self.edge2, aggregator_type='sum')

        self.gvonc = dgl.nn.pytorch.GATConv(in_feats, out_feats, num_heads=8)

        self.final_layer = nn.Sequential(
            nn.Linear(out_feats * 2 * 8, 64),
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
        h = self.gvo1(g, n)
        h = self.gvo2(g, h)
        h = self.gvonc(g, h)  # returns [nodes, out_features, heads]
        h = F.elu(h)
        h = h.view(h.shape[0], -1)

        h1 = self.pooling(g, h)  # returns [batch, out_features]
        h2 = self.pooling2(g, h)
        h = torch.cat([h1, h2], dim=-1)

        p = F.elu(h)
        h_out = self.final_layer(p)  # [batch, 1]

        return h_out, p
