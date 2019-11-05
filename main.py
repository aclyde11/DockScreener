import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import multiprocessing
from features import datasets
from features import utils as featmaker
from models.GAT import GAT
from tqdm import tqdm
import argparse
from utils import Avg, MetricCollector
import pickle

def poolapply(i):
    try:
        x = i[0]
        y = i[1]
        g  = featmaker.get_dgl_graph(x)
        t = np.array([y]).reshape((1, 1))
        return g,t
    except:
        return None


def load_cora_data(f, size=None):
    print("Loading data")
    kwargs = {}
    if size is not None:
        kwargs['nrows'] = size
    df = pd.read_csv(f, **kwargs)
    pairs = map(lambda x : (str(x[0]), float(x[1])), df.itertuples(index=False))

    # with multiprocessing.Pool(processes=1) as pool:
    #     graphs = list(tqdm(pool.imap(poolapply, pairs)))
    graphs = [poolapply(i) for i in tqdm(pairs)]
    print("done")
    graphs = list(filter(lambda x: x is not None or x[0] is not None, graphs))
    return graphs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, required=True)
    parser.add_argument('-b', type=int, default=100)
    parser.add_argument('-n', type=int, default=1000)
    parser.add_argument('-e', type=str, required=True)
    args = parser.parse_args()
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(dev)
    BATCH_SIZE = args.b

    # g = datasets.GraphDataset(load_cora_data(args.i, size=250000))
    with open("train_data.pkl", 'rb') as f:
        # pickle.dump(g, f)
        g = pickle.load(f)
    train_loader = DataLoader(g, collate_fn=datasets.graph_collate, shuffle=True, num_workers=3, batch_size=BATCH_SIZE)

    # g = datasets.GraphDataset(load_cora_data(args.e))
    with open("test_data.pkl", 'rb') as f:
        # pickle.dump(g, f)
        g = pickle.load(f)

    test_loader = DataLoader(g, collate_fn=datasets.graph_collate, shuffle=True, num_workers=3, batch_size=BATCH_SIZE)

    net = GAT(133, 14).to(dev)

    # create optimizer
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4)

    # main loop
    dur = []
    for epoch in range(30):
        net.train()
        train_avg = Avg()
        for g, v in tqdm(train_loader):
            if epoch >= 3:
                t0 = time.time()
            v = v.to(dev)
            v_pred = net(g, g.ndata['atom_features'].to(dev), g.edata['edge_features'].to(dev))
            loss = F.mse_loss(v, v_pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_avg(loss.item())
        print("epoch", epoch, "train loss", train_avg.avg())

        net.eval()
        with torch.no_grad():
            test_avg = Avg()
            r2= MetricCollector()
            for g, v in test_loader:
                v = v.to(dev)
                v_pred = net(g, g.ndata['atom_features'].to(dev), g.edata['edge_features'].to(dev))
                loss = F.mse_loss(v, v_pred)
                test_avg(loss.item())
                r2(v, v_pred)
            print("epoch", epoch, "test loss", test_avg.avg(), r2.r2())
