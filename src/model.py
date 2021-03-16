import numpy as np
from typing import List
# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
# torch_geometric
import torch_geometric
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.conv import GCNConv, CGConv, GATConv
# custom files
from utils import store_args
        

class CGConvNet(nn.Module):
    @store_args
    def __init__(self, in_feat, edge_feat, num_class):
        super(CGConvNet, self).__init__()
        # TODO modify this architecture
        self.cgconv1  = CGConv(channels=(in_feat, in_feat), dim=edge_feat)
        self.cgconv2  = CGConv(channels=(in_feat, in_feat), dim=edge_feat)
        self.gatconv1 = GATConv(in_channels=in_feat, out_channels=32, heads=3)
        self.gatconv2 = GATConv(in_channels=int(32*3), out_channels=16, heads=3)
        self.lin1 = nn.Linear(int(16*3), 16)
        self.lin2 = nn.Linear(16, num_class)
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.relu(self.cgconv1(x, edge_index, edge_attr))
        x = F.relu(self.cgconv2(x, edge_index, edge_attr))
        x = F.relu(self.gatconv1(x, edge_index))
        x = F.relu(self.gatconv2(x, edge_index))
        x = global_mean_pool(x, batch)  # aggregate from all nodes
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

if __name__ == "__main__":
    from dataloader import MNISTDataloader
    in_feat, edge_feat, num_nodes, num_classes, batch_size = 1, 2, 6, 10, 32
    loader = MNISTDataloader(num_nodes=num_nodes, batch_size=batch_size, train_val_split_ratio=0.2, seed=0, testing=False)
    trainloader = loader.train_dataloader
    for i, (data, target) in enumerate(trainloader):
        geom_loader = loader.process_torch_geometric(data, num_samples=10, k=5, polar=True)
        for batch in geom_loader:
            print(f'Torch Geometric Batch: {batch}')
        break
    net = CGConvNet(in_feat, edge_feat, num_classes)
    print(net)
    print(net(batch).shape)
