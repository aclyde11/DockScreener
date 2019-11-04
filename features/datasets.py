import dgl
import torch
from torch.utils.data import DataLoader, Dataset


class GraphDataset(Dataset):
    def __init__(self, g):
        self.graphs, self.values = zip(*g)
        assert (len(self.graphs) == len(self.values))

    def __getitem__(self, item):
        return self.graphs[item], self.values[item]

    def __len__(self):
        return len(self.graphs)

def graph_collate(x):
    g, v = zip(*x)
    batch_graph = dgl.batch(g)
    batch_values = torch.stack(v, dim=0)
    return batch_graph, batch_values.view(len(g), 1)