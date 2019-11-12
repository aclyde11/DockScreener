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
from models.GAT import GAT, GAT_small
from tqdm import tqdm
import argparse
from utils import Avg, MetricCollector
import pickle
import math
import torch.backends.cudnn
torch.backends.cudnn.benchmark = True



def poolapply(i):
    try:
        x = i[0]
        y = i[1]
        g  = featmaker.get_dgl_graph(x)
        t = np.array([y]).reshape((1, 1))
        return g,t
    except:
        return None

def get_mom(epoch):
    if epoch < 15:
        return 0.005
    if epoch > 15 and epoch < 25:
        return  -0.005
    return 0

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

def find_lr(init_value = 1e-8, final_value=10., beta = 0.98):
    num = len(train_loader)-1
    mult = (final_value / init_value) ** (1/num)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []
    for data in train_loader:
        batch_num += 1
        #As before, get the loss for this mini-batch of inputs/outputs
        inputs,labels = data
        inputs, labels = torch.autograd.Variable(inputs), torch.autograd.Variable(labels)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = F.mse_loss(outputs, labels)
        #Compute the smoothed loss
        avg_loss = beta * avg_loss + (1-beta) *loss.data[0]
        smoothed_loss = avg_loss / (1 - beta**batch_num)
        #Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses
        #Record the best loss
        if smoothed_loss < best_loss or batch_num==1:
            best_loss = smoothed_loss
        #Store the values
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))
        #Do the SGD step
        loss.backward()
        optimizer.step()
        #Update the lr for the next step
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr
    return log_lrs, losses

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

    # # g = datasets.GraphDataset(load_cora_data(args.i, size=250000))
    with open("train_data.pkl", 'rb') as f:
    #     # pickle.dump(g, f)
        g = pickle.load(f)
    train_loader = DataLoader(g, collate_fn=datasets.graph_collate, shuffle=True, num_workers=3, batch_size=BATCH_SIZE)

    # print("getting top 10% values")
    # values = []
    # for i in range(len(g)):
    #     values.append(g[i][1])
    # values = np.array(values)
    # good_values = np.quantile(values, 0.05)
    # good_values_tensor = torch.FloatTensor([good_values]).float().flatten().to(dev)
    # print(good_values_tensor)
    #
    # print("1%", good_values)
    # good_values = np.where(values < good_values)
    # print(good_values)
    # values = []
    # for i in good_values[0]:
    #     values.append(g[i])
    #
    #
    # g_good = datasets.GraphDataset(values)
    # train_best_loader = DataLoader(g_good, collate_fn=datasets.graph_collate, shuffle=True, num_workers=3, batch_size=BATCH_SIZE)


    #
    # g = datasets.GraphDataset(load_cora_data(args.e))
    with open("test_data.pkl", 'rb') as f:
        # pickle.dump(g, f)
        g = pickle.load(f)

    test_loader = DataLoader(g, collate_fn=datasets.graph_collate, shuffle=True, num_workers=3, batch_size=BATCH_SIZE)

    net = GAT(133, 14).to(dev)
    net.load_state_dict(torch.load("model.pt"))
    net2 = GAT_small(133, 14, prev_out=64 * 2).to(dev)
    print("TOTAL PARMS", sum(p.numel() for p in net.parameters() if p.requires_grad))

    # create optimizer
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4)
    optimizer2 = torch.optim.AdamW(net2.parameters(), lr=1e-3)

    # net.load_state_dict(torch.load("model.pt"))
    # main loop
    dur = []

    lossf = F.mse_loss
    second_lossf = torch.nn.MSELoss(reduction='none')
    for epoch in range(50):
        # net.train()
        # train_avg = Avg()
        # # if epoch < 10:
        # for g, v in tqdm(train_loader):
        #     optimizer2.zero_grad()
        #     optimizer.zero_grad()
        #
        #     if epoch >= 3:
        #         t0 = time.time()
        #     v = v.to(dev)
        #     af = g.ndata['atom_features'].to(dev)
        #     ge = g.edata['edge_features'].to(dev)
        #     v_pred, p = net(g, af, ge)
        #     v_small = net2(g, af, ge, p.detach())
        #
        #     v_pred = v_pred.view(v.shape[0], -1)
        #     v_small = v_small.view(v.shape[0], -1)
        #
        #     loss_h = (second_lossf(v, v_small) * (v <= good_values_tensor)).mean()
        #     loss = lossf(v, v_pred).mean()
        #
        #     loss.backward()
        #     optimizer.step()
        #
        #     loss_h.backward()
        #     optimizer2.step()
        #     train_avg(loss.item())

        # else:
        #     if epoch == 10:
        #         optimizer = torch.optim.SGD(net.parameters(), lr=1e-5)
        #
        #     for g in optimizer.param_groups:
        #         g['lr'] += 5e-5
        #     for g, v in tqdm(train_best_loader):
        #         if epoch >= 3:
        #             t0 = time.time()
        #         v = v.to(dev)
        #         v_pred, v_small, _ = net(g, g.ndata['atom_features'].to(dev), g.edata['edge_features'].to(dev))
        #
        #         v = v.view(v.shape[0], -1)
        #         v_pred = v_pred.view(v.shape[0], -1)
        #         v_small = v_small.view(v.shape[0], -1)
        #         loss_h = second_lossf(v, v_small) * (v <= good_values_tensor)
        #         loss = lossf(v, v_pred).mean() + 0.5 * loss_h.mean()
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()
        #         train_avg(loss.item())

        # print("epoch", epoch, "train loss", train_avg.avg())
        # torch.save( net.state_dict(), 'model.pt')
        # net.eval()
        with torch.no_grad():
            test_avg = Avg()
            r2= MetricCollector()
            ps = []
            for g, v in tqdm(train_loader):
                v = v.to(dev)

                v = v.to(dev)
                af = g.ndata['atom_features'].to(dev)
                ge = g.edata['edge_features'].to(dev)
                v_pred, p = net(g, af, ge)
                v_small = net2(g, af, ge, p.detach())
                ps.append(p.detach().cpu().numpy())
                v = v.view(v.shape[0], -1)
                v_pred  = v_pred.view(v.shape[0], -1)
                # v_small =  v_small.view(v.shape[0], -1)

                # v_pred = v_small * (v_pred <= good_values_tensor).float() + v_pred * (v_pred > good_values_tensor).float()
                loss = lossf(v,v_pred).mean()
                test_avg(loss.item())
                r2(v, v_pred)

            x = np.concatenate([np.concatenate(ps, axis=0), np.array(r2.trues).reshape(-1,1)], axis=-1)
            np.concatenate([x, np.array(r2.preds).reshape(-1,1)], axis=-1)

            np.savez("outs.npz", x)
            print("Runnin guassian process")

            X_train = np.concatenate(ps, axis=0)
            from sklearn.gaussian_process import GaussianProcessRegressor
            gp = GaussianProcessRegressor()
            gp.fit(X_train, r2.trues)
            print("SCore gp", gp.score(X_train, r2.trues))

            r2.preds = gp.predict(X_train).flatten()


            print("epoch", epoch, "test loss", test_avg.avg(), r2.r2(), r2.nefr(0.05,0.05), r2.nefr(0.1,0.1), r2.nefr(0.01,0.01), r2.nefr(0.001,0.001))
            preds = np.array(r2.preds, dtype=np.float32)
            trues = np.array(r2.trues, dtype=np.float32)
            out = np.stack([preds, trues]).astype(np.float16)
            np.save("out_test.npy",  out)
