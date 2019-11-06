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

class GAT(nn.Module):
    def __init__(self, in_dim, edge_feats):
        super(GAT, self).__init__()
        in_feats = in_dim
        out_feats = 32
        self.edge_layer = nn.Linear(edge_feats, out_feats * out_feats)

        self.conv1 = dgl.nn.pytorch.conv.SAGEConv(
            in_feats=in_feats,
            out_feats=out_feats,
            aggregator_type='lstm')


        self.conv2 = dgl.nn.pytorch.conv.SAGEConv(
            in_feats=out_feats,
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
            nn.Linear(out_feats, 32),
            nn.ReLU(),
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32,1)
        )
        # self.pooling = dgl.nn.pytorch.glob.MaxPooling()

        self.gate_nn = nn.Sequential(
            nn.Linear(out_feats, 16),
            nn.ReLU(),
            nn.Linear(16,1)
        )
        self.pooling = dgl.nn.pytorch.glob.GlobalAttentionPooling(gate_nn=self.gate_nn)


    '''
    g: DglGraph
    n: node feature matrix
    e: edge feature matrix
    '''
    def forward(self, g, n, e):
        h = self.conv1(g,n)   # returns [nodes, out_features]
        h = F.relu(h)
        h = self.conv2(g,h)   # returns [nodes, out_features]
        h = F.relu(h)
        h = self.conv3(g,h,e) # returns [nodes, out_features]
        h = F.relu(h)
        h = self.conv4(g,h)   # returns [nodes, out_features]
        h = self.pooling(g,h) # returns [batch, out_features]

        h = F.elu(h)
        h = self.final_layer(h) #[batch, 1]
        return h
