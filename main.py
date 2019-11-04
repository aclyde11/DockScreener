import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from features import datasets
from features import utils as featmaker
from models.GAT import GAT
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-i', type=str, required=True)
parser.add_argument('-b', type=int, default=100)
parser.add_argument('-n', type=int, default=1000)
args = parser.parse_args()

BATCH_SIZE=args.b

def load_cora_data():
    df = pd.read_csv(args.i, nrows=args.n)
    graphs = []
    for i in tqdm(range(df.shape[0])):
        graphs.append((featmaker.get_dgl_graph(df.iloc[i, 0]),
                       torch.FloatTensor([df.iloc[i, 1]]).view(1, 1)))
    return graphs


g = datasets.GraphDataset(load_cora_data())
train_loader = DataLoader(g, collate_fn=datasets.graph_collate, num_workers=3, batch_size=100)

net = GAT(133, 14)

# create optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

# main loop
dur = []
for epoch in range(30):
    for g, v in train_loader:
        if epoch >= 3:
            t0 = time.time()

        v_pred = net(g, g.ndata['atom_features'], g.edata['edge_features'])
        loss = F.mse_loss(v, v_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f}".format(
            epoch, loss.item(), np.mean(dur)))
