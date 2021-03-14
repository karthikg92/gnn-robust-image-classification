import numpy as np
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.conv import GCNConv, CGConv

from layer import GraphConvolution
from utils import store_args

class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs
        
class Net(nn.Module):
    @store_args
    def __init__(self, in_feat:int, gcn_hid:List, lin_hid:List, num_nodes:int, num_class:int, dropout:float):
        """
            A network with two graph convolutional layers and linear layers
            Parameters:
            -----------
            in_feat: int
                The size of the node features
            gcn_hid: List
                # TODO: make this variable length by using mySequential
                A list of length 2 representing the hidden layer 
                dimensions of the graph convolutional layers
                Example: [10, 8]
            lin_hid: List
                A list of any length representing the hidden layer
                dimensions of the linear layers
            num_nodes: int
                The number of nodes in the adjacency matrix
                OR the number of pixels sampled in the image
            num_classes: int
                Number of classes for classification of input
            dropout: float
                The dropout probability in GCN layers
        """
        super(Net, self).__init__()

        gcn_layer_dims = [self.in_feat] + self.gcn_hid
        lin_layer_dims =  [int(self.num_nodes * self.gcn_hid[-1])] + self.lin_hid + [num_class]

        self.gc1 = GraphConvolution(in_feat, self.gcn_hid[0])
        self.gc2 = GraphConvolution(self.gcn_hid[0], self.gcn_hid[1])

        # self.gcn_layers = []
        # # GCN layers
        # for in_feat, out_feat in zip(gcn_layer_dims[:-1],gcn_layer_dims[1:]):
        #     self.gcn_layers.append(GraphConvolution(in_feat, out_feat))
        #     self.gcn_layers.append(nn.ReLU())
        #     self.gcn_layers.append(nn.Dropout(self.dropout))
        # self.gcn_layers = mySequential(*self.gcn_layers)
        
        # linear layers
        self.lin_layers = []
        for idx, (in_feat, out_feat) in enumerate(zip(lin_layer_dims[:-1], lin_layer_dims[1:])):
            self.lin_layers.append(nn.Linear(in_feat, out_feat))
            # if not final layer, then add relu since final layer is log_softmax
            if idx != len(lin_layer_dims)-2:
                self.lin_layers.append(nn.ReLU())
        self.lin_layers = nn.Sequential(*self.lin_layers)

    def forward(self, x:torch.tensor, adj:torch.tensor):
        """
            Performs adj * input * weight where (*) represents matmul
            Parameters:
            -----------
            input:torch.tensor
                Shape: [batch_size, num_nodes, in_feat]
            adj:torch.tensor
                Shape: [batch_size, num_nodes, num_nodes]
            Returns:
            --------
            output:torch.tensor
                Shape: [batch_size, num_nodes, out_feat]
        """
        # gcn_x = self.gcn_layers(x, adj) # shape (batch_size, num_nodes, out_feat)
        x = F.dropout(F.relu(self.gc1(x, adj)), self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = x.flatten(1)            # shape (batch_size, num_nodes*out_feat)   
        out = self.lin_layers(x)    # NOTE: use cross-entropy loss since no log_softmax here
        return out

class torchNet(nn.Module):
    @store_args
    def __init__(self, in_feat, gcn_hid, num_class):
        super(torchNet, self).__init__()
        self.conv1 = GCNConv(in_feat, gcn_hid[0])
        self.conv2 = GCNConv(gcn_hid[0], gcn_hid[1])
        self.lin = nn.Linear(gcn_hid[1], num_class)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x,batch)
        x = self.lin(x)
        return x

class CGConvNet(nn.Module):
    @store_args
    def __init__(self, in_feat, edge_feat, num_class):
        super(CGConvNet, self).__init__()
        self.conv1 = CGConv(channels=(in_feat, in_feat), dim=edge_feat)
        self.lin = nn.Linear(in_feat, num_class)
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = global_mean_pool(x,batch)
        x = self.lin(x)
        return x

if __name__ == "__main__":
    # from scipy import sparse
    # in_feat, edge_feat, num_nodes, num_classes, batch_size = 14, 2, 6, 10, 32
    # gcn_hid,lin_hid = [12,10], [5,3]
    # # lin_hid = [5,3]
    # dropout = 0.2
    # gcn = Net(in_feat, gcn_hid, lin_hid, num_nodes, num_classes, dropout)
    # # gcn = Net(14, [12,10],[5,3],6,10,0.2)
    # # gcn = Net(14,10,3,0.5)
    # print(gcn)
    # x = torch.rand(batch_size, num_nodes, in_feat)
    # adj = torch.rand(batch_size, num_nodes, num_nodes)
    # out = gcn(x, adj)
    # print(out.shape)

    # print('_'*50)
    # gcn = torchNet(in_feat,gcn_hid, num_classes)
    # x = torch.rand(num_nodes, in_feat)
    # adj = torch.rand(num_nodes, num_nodes)
    # sA = sparse.csr_matrix(adj.numpy())   # convert to sparse format
    # edge_index, edge_weight = from_scipy_sparse_matrix(sA)
    # datalist = []
    # data = torch_geometric.data.Data(x=x, edge_index=edge_index, edge_attr=edge_weight)
    # # make a dummy batch
    # for i in range(batch_size):
    #     datalist.append(data)
    # loader = torch_geometric.data.DataLoader(datalist, batch_size=batch_size)
    # for batch in loader:
    #     print('_')
    # print(gcn)
    # print(gcn(batch).shape)

    from dataloader import MNISTDataloader
    in_feat, edge_feat, num_nodes, num_classes, batch_size = 1, 2, 6, 10, 32
    loader = MNISTDataloader(num_nodes=num_nodes, batch_size=batch_size, train_val_split_ratio=0.2, seed=0, testing=False)
    trainloader = loader.train_dataloader
    for i, (data, target) in enumerate(trainloader):
        geom_loader = loader.process_torch_geometric(data, 10, True)
        for batch in geom_loader:
            print(f'Torch Geometric Batch: {batch}')
            batch.to('cpu')
        break
    net = CGConvNet(in_feat, edge_feat, num_classes)
    print(net)
    print(net(batch).shape)
