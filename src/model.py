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
from torch_geometric.nn.conv import GCNConv, CGConv, GATConv, SGConv
# custom files
from utils import store_args
        
class mySGConv(nn.Module):
    @store_args
    def __init__(self, in_channels:int, out_channels:int, num_edge_feat:int):
        super(mySGConv, self).__init__()
        self.conv_head1 = SGConv(in_channels=in_channels, out_channels=out_channels)
        self.conv_head2 = SGConv(in_channels=in_channels, out_channels=out_channels)
        if num_edge_feat==4:
            self.conv_head3 = SGConv(in_channels=in_channels, out_channels=out_channels)
            self.conv_head4 = SGConv(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x, edge_index, edge_feat):
        """
            x: torch.tensor; shape [batch, in_feat]
            edge_index: torch.tensor; shape [2, num_edges]
            edge_feat: torch.tensor; shape [num_edges, num_edge_feat]
        """
        edge_feat = torch.abs(edge_feat)    # since we need to take sqrt for degree normalisation it has to be positive
        x1 = self.conv_head1(x, edge_index, edge_feat[:,0]) # [batch, out_feat]
        x2 = self.conv_head2(x, edge_index, edge_feat[:,1]) # [batch, out_feat]
        x_out = torch.cat((x1,x2),1)
        if self.num_edge_feat == 4:
            x3 = self.conv_head3(x, edge_index, edge_feat[:,2])
            x4 = self.conv_head4(x, edge_index, edge_feat[:,3])
            x_ = torch.stack((x1,x2),1)
            x_out = torch.cat((x_out,x_),1)
        return x_out
        

class GNNConvNet(nn.Module):
    @store_args
    def __init__(self, in_feat:int, edge_feat:int, num_class:int):
        super(GNNConvNet, self).__init__()
        # TODO modify this architecture
        self.cgconv1  = CGConv(channels=(in_feat, in_feat), dim=edge_feat)
        self.cgconv2  = CGConv(channels=(in_feat, in_feat), dim=edge_feat)
        self.gatconv1 = GATConv(in_channels=in_feat, out_channels=32, heads=3)
        self.gatconv2 = GATConv(in_channels=int(32*3), out_channels=16, heads=3)
        self.sgconv1  = mySGConv(in_channels=int(16*3), out_channels=32, num_edge_feat=edge_feat)
        self.sgconv2  = mySGConv(in_channels=int(32*edge_feat), out_channels=32, num_edge_feat=edge_feat)
        self.lin1 = nn.Linear(int(32*edge_feat), 16)
        self.lin2 = nn.Linear(16, num_class)
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.relu(self.cgconv1(x, edge_index, edge_attr))
        x = F.relu(self.cgconv2(x, edge_index, edge_attr))
        x = F.relu(self.gatconv1(x, edge_index))
        x = F.relu(self.gatconv2(x, edge_index))
        x = F.relu(self.sgconv1(x, edge_index, edge_attr))
        x = F.relu(self.sgconv2(x, edge_index, edge_attr))
        x = global_mean_pool(x, batch)  # aggregate from all nodes
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

if __name__ == "__main__":
    from dataloader import MNISTDataloader
    in_feat, edge_feat, num_nodes, num_classes, batch_size = 1, 2, 6, 9, 32
    loader = MNISTDataloader(num_nodes=num_nodes, batch_size=batch_size, train_val_split_ratio=0.2, seed=0, remove9=True, testing=False)
    trainloader = loader.train_dataloader
    for i, (data, target) in enumerate(trainloader):
        geom_loader = loader.process_torch_geometric(data, num_samples=10, k=5, polar=True)
        for batch in geom_loader:
            print(f'Torch Geometric Batch: {batch}')
        break
    net = GNNConvNet(in_feat, edge_feat, num_classes)
    print(net)
    print(net(batch).shape)
    print(f'Number of parameters in network: {sum(p.numel() for p in net.parameters())}')
