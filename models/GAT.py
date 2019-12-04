import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch

GAT_parameters = {
    # number of graph convolutions to perform
    # output feature size 1
    # aggregator type {4}
    # aggregator type {2}
    # output feature size 2
    # gcn activation functions
    # pooling function to use -> if using attention pooling then a NN search
    # linear network archtecture and activations
}

import torch
import torch.nn as nn
import torch.nn.functional as F


class GAT_small(nn.Module):
    def __init__(self, in_dim, edge_feats, prev_out=64):
        super(GAT_small, self).__init__()
        in_feats = in_dim
        out_feats = 16
        self.edge_layer = nn.Linear(edge_feats, out_feats * out_feats)

        self.conv1 = dgl.nn.pytorch.conv.SAGEConv(
            in_feats=in_feats,
            out_feats=out_feats,
            aggregator_type='mean')

        self.conv2 = dgl.nn.pytorch.conv.SAGEConv(
            in_feats=out_feats,
            out_feats=out_feats,
            aggregator_type='mean')

        # self.conv2 = dgl.nn.pytorch.conv.NNConv(
        #     in_feats=out_feats,
        #     out_feats=out_feats,
        #     edge_func=self.edge_layer,
        #     aggregator_type='sum')

        self.final_layer = nn.Sequential(
            nn.BatchNorm1d(out_feats * 2),
            nn.Linear(out_feats * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(32,1)
        )
        self.pooling2 = dgl.nn.pytorch.glob.MaxPooling()

        self.gate_nn = nn.Sequential(
            nn.BatchNorm1d(out_feats),
            nn.Linear(out_feats, 1)
        )
        self.pooling = dgl.nn.pytorch.glob.GlobalAttentionPooling(gate_nn=self.gate_nn)


    '''
    g: DglGraph
    n: node feature matrix
    e: edge feature matrix
    '''
    def forward(self, g, n, e, p):
        h = self.conv1(g,n)   # returns [nodes, out_features]
        h = F.elu(h)
        h = self.conv2(g,h)   # returns [nodes, out_features]
        h1 = self.pooling(g,h) # returns [batch, out_features]
        h2 = self.pooling2(g,h)
        h = torch.cat([h1,h2], dim=-1)
        h = F.elu(h)

        h = self.final_layer(h) #[batch, 1]
        return h

class GAT(nn.Module):
    def __init__(self, in_dim, edge_feats):
        super(GAT, self).__init__()
        in_feats = in_dim
        out_feats = 64


        self.gvo1 = dgl.nn.pytorch.conv.GraphConv(in_feats, out_feats, activation=F.relu)
        self.gvo2 = dgl.nn.pytorch.conv.GraphConv(in_feats, out_feats, activation=F.relu)

        self.gvonc = dgl.nn.pytorch.GATConv(in_feats, out_feats, num_heads=8)



        self.final_layer = nn.Sequential(
            nn.BatchNorm1d(out_feats * 2 * 8),
            nn.Linear(out_feats * 2 * 8, 64),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(32,1)
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
    def forward(self, g, n, e, return_fp = True):
        h = self.gvo1(g,n)
        h = self.gvo2(g, h)
        h = self.gvonc(g,h)   # returns [nodes, out_features, heads]
        h = F.elu(h)
        h = h.view(h.shape[0], -1)

        h1 = self.pooling(g,h) # returns [batch, out_features]
        h2 = self.pooling2(g,h)
        h = torch.cat([h1,h2], dim=-1)

        p = F.elu(h)
        h_out = self.final_layer(p) #[batch, 1]

        return h_out, p

