from typing import List

import numpy as np
# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
# custom files
from utils import store_args

class ConvNet(nn.Module):
    @store_args
    def __init__(self, dataset_name:str, patch_type:str):
        super(ConvNet, self).__init__()
        if dataset_name == 'CIFAR' or 'SVHN':
            in_channels = 3
        elif dataset_name == 'MNIST':
            in_channels = 1
        if patch_type == 'mask':
            in_channels *= 2
        self.conv_layers = nn.Sequential(
                                        nn.Conv2d(in_channels, 64, 3),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(),
                                        nn.Conv2d(64, 64, 3),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(),
                                        nn.Conv2d(64, 64, 3),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(),
                                        nn.Conv2d(64, 64, 3),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(),
                                        nn.Conv2d(64, 64, 3),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(),
                                        nn.Conv2d(64, 64, 3),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(),
                                        nn.Conv2d(64, 64, 3),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(),
                                        nn.Conv2d(64, 64, 3),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(),
                                )
    def forward(self, x:torch.tensor):
        pass


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
    net = GNNConvNet(in_feat, edge_feat, num_classes,5)
    print(net)
    print(net(batch).shape)
    print(f'Number of parameters in network: {sum(p.numel() for p in net.parameters())}')
