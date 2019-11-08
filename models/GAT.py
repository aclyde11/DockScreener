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

class GAT_small(nn.Module):
    def __init__(self, in_dim, edge_feats, prev_out=64):
        super(GAT_small, self).__init__()
        in_feats = in_dim
        out_feats = 16
        self.edge_layer = nn.Linear(edge_feats, out_feats * out_feats)

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
            nn.BatchNorm1d(out_feats * 2 + prev_out),
            nn.Linear(out_feats * 2 + prev_out, 32),
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
        h = self.conv2(g,h,e)   # returns [nodes, out_features]
        h1 = self.pooling(g,h) # returns [batch, out_features]
        h2 = self.pooling2(g,h)
        h = torch.cat([p, h1,h2], dim=-1)
        h = F.elu(h)

        h = self.final_layer(h) #[batch, 1]
        return h

class GAT(nn.Module):
    def __init__(self, in_dim, edge_feats, good_value=1e1):
        super(GAT, self).__init__()
        in_feats = in_dim
        out_feats = 64
        self.g = good_value
        self.edge_layer = nn.Linear(edge_feats, out_feats * out_feats)

        self.conv1 = dgl.nn.pytorch.conv.SAGEConv(
            in_feats=in_feats,
            out_feats=out_feats,
            aggregator_type='lstm')




        self.conv3 = dgl.nn.pytorch.conv.NNConv(
            in_feats=out_feats,
            out_feats=out_feats,
            edge_func=self.edge_layer,
            aggregator_type='sum')

        self.conv4 = self.conv2 = dgl.nn.pytorch.conv.SAGEConv(
            in_feats=out_feats,
            out_feats=out_feats,
            aggregator_type='lstm')

        self.final_layer = nn.Sequential(
            nn.BatchNorm1d(out_feats * 2),
            nn.Linear(out_feats * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(64,32),
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

        self.gat_small = GAT_small(in_dim, edge_feats, prev_out=out_feats * 2)


    '''
    g: DglGraph
    n: node feature matrix
    e: edge feature matrix
    '''
    def forward(self, g, n, e, return_fp = True):
        h = self.conv1(g,n)   # returns [nodes, out_features]
        h = F.elu(h)
        # h = self.conv2(g,h)   # returns [nodes, out_features]
        # h = F.elu(h)
        h = self.conv3(g,h,e) # returns [nodes, out_features]
        h = F.elu(h)
        h = self.conv4(g,h)   # returns [nodes, out_features]

        h1 = self.pooling(g,h) # returns [batch, out_features]
        h2 = self.pooling2(g,h)
        h = torch.cat([h1,h2], dim=-1)

        p = F.elu(h)
        h_out = self.final_layer(p) #[batch, 1]

        mask = h_out < self.g
        # h_small = h_out * mask

        h_small = self.gat_small(g, n , e, p)

        return h_out, h_small, mask

