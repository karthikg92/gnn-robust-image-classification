import numpy as np
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

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


if __name__ == "__main__":
    in_feat, num_nodes, num_classes, batch_size = 14, 6, 10, 32
    gcn_hid,lin_hid = [12,10], [5,3]
    # lin_hid = [5,3]
    dropout = 0.2
    gcn = Net(in_feat, gcn_hid, lin_hid, num_nodes, num_classes, dropout)
    # gcn = Net(14, [12,10],[5,3],6,10,0.2)
    # gcn = Net(14,10,3,0.5)
    print(gcn)
    x = torch.rand(batch_size, num_nodes, in_feat)
    adj = torch.rand(batch_size, num_nodes, num_nodes)
    out = gcn(x, adj)
    print(out.shape)