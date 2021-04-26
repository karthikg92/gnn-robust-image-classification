# typing
from typing import Union, Tuple
from torch_geometric.typing import PairTensor, Adj, OptTensor, Size
# torch
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Linear, BatchNorm1d
# torch_geometric
import torch_geometric
from torch_geometric.nn.conv import MessagePassing


# referred https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html#implementing-the-edge-convolution
# for the custom torch_geometric layer
class ComboConv(MessagePassing):
    """
        Implementation of edge-node feature combo graph convolutional network
        For node pair (i,j):
            x_f_ij  = σ(W*[x_i, x_j] + b)
            x_e_ij  = σ(W*[e_ij] + b)
            x_fe_ij = σ(W*[x_i, x_j, e_ij] + b)
            x_d_ij  = σ(W*[e_ij] + b) ⊙ x_j
            x_c_ij  = σ(W*[x_f_ij, x_e_ij, x_fe_ij, x_d_ij] + b)
            x_i' = x_i + Σ x_c_ij
        Parameters:
        -----------
        • node_feat_dim (int): Size of node feature dimensions
        • edge_dim (int): Size of edge feature dimensions
        • aggr (string, optional): The aggregation operator to use
              (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
              (default: :obj:`"add"`)
        • batch_norm (bool, optional): If set to :obj:`True`, will make use of
              batch normalization. (default: :obj:`False`)
        • bias (bool, optional): If set to :obj:`False`, the layer will not learn
              an additive bias. (default: :obj:`True`)
        • **kwargs (optional): Additional arguments of
              :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, node_feat_dim: int, edge_dim: int, out_features: int,
                 add_self_loop: bool = True, aggr: str = 'add',
                 batch_norm: bool = False, bias: bool = True, **kwargs):
        super(ComboConv, self).__init__(aggr=aggr, **kwargs)
        self.edge_dim = edge_dim
        self.node_feat_dim = node_feat_dim
        self.add_self_loop = add_self_loop
        self.batch_norm = batch_norm

        ########################################
        # for x_f_ij = σ(W*[x_i, x_j] + b)
        self.lin_f = Linear(node_feat_dim * 2, node_feat_dim, bias=bias)
        # for x_e_ij = σ(W*[e_ij] + b)
        self.lin_e = Linear(edge_dim, node_feat_dim, bias=bias)
        # for x_fe_ij = σ(W*[x_i, x_j, e_ij] + b)
        self.lin_fe = Linear(edge_dim + node_feat_dim * 2, node_feat_dim, bias=bias)
        # for x_d_ij = σ(W*[e_ij] + b) ⊙ x_j
        self.lin_d = Linear(edge_dim, node_feat_dim, bias=bias)
        # for x_c = σ(W*[x_f_ij, x_e_ij, x_fe_ij, x_d_ij] + b)
        self.lin_c = Linear(4 * node_feat_dim, out_features, bias=bias)
        ########################################
        if self.batch_norm:
            self.bn = BatchNorm1d(out_features)
        ########################################
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_f.reset_parameters()
        self.lin_e.reset_parameters()
        self.lin_fe.reset_parameters()
        self.lin_d.reset_parameters()
        self.lin_c.reset_parameters()
        if self.batch_norm:
            self.bn.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr: Tensor = None, size: Size = None) -> Tensor:
        out = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr, size=size)
        out = self.bn(out) if self.batch_norm else out
        if self.add_self_loop:
            out += x[1]
        return out

    def message(self, x_i: Tensor, x_j:Tensor, edge_attr:Tensor) -> Tensor:
        x_f_ij = self.lin_f(torch.cat([x_i, x_j], dim=-1)).relu()
        x_e_ij = self.lin_e(edge_attr).relu()
        x_fe_ij = self.lin_fe(torch.cat([x_i, x_j, edge_attr], dim=-1)).relu()
        x_d_ij = self.lin_d(edge_attr).relu() * x_j
        x_c_ij = self.lin_c(torch.cat([x_f_ij, x_e_ij, x_fe_ij, x_d_ij], dim=-1)).relu()
        return x_c_ij

    def __repr__(self):
        return f'{self.__class__.__name__}(node_feat_dim={self.node_feat_dim}, edge_dim={self.edge_dim}, add_self_loop={self.add_self_loop})'

class comboConvFilter(MessagePassing):
    def __init__(self, node_feat_dim: int, edge_dim: int, out_features: int,
                 num_filters: int = 1, add_self_loop: bool = True, aggr: str = 'add',
                 batch_norm: bool = False, bias: bool = True, **kwargs):
        super(comboConvFilter, self).__init__(aggr=aggr, **kwargs)
        self.edge_dim = edge_dim
        self.node_feat_dim = node_feat_dim
        self.add_self_loop = add_self_loop
        self.num_filters = num_filters
        self.batch_norm = batch_norm

        self.layer_f = nn.ModuleList()
        self.layer_e = nn.ModuleList()
        self.layer_fe = nn.ModuleList()
        self.layer_d = nn.ModuleList()

        for _ in num_filters:
            self.layer_f.append(Linear(node_feat_dim * 2, node_feat_dim, bias=bias))
            self.layer_e.append(Linear(edge_dim, node_feat_dim, bias=bias))
            self.layer_fe.append(Linear(edge_dim + node_feat_dim * 2, node_feat_dim, bias=bias))
            self.layer_d.append(Linear(edge_dim, node_feat_dim, bias=bias))

        self.linear_final = Linear(node_feat_dim * 4 * num_filters, out_features, bias=bias)
        if self.batch_norm:
            self.bn = BatchNorm1d(out_features)

        self.reset_parameters()

    def reset_parameters(self):
        for _ in self.num_filters:
            self.layer_f[i].reset_parameters()
            self.layer_e[i].reset_parameters()
            self.layer_fe[i].reset_parameters()
            self.layer_d[i].reset_parameters()
        self.linear_final.reset_parameters()
        if self.batch_norm:
            self.bn.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr: Tensor = None, size: Size = None) -> Tensor:
        out = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr, size=size)
        out = self.bn(out) if self.batch_norm else out
        if self.add_self_loop:
            out += x[1]
        return out

    def forward_pass(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor, k: int):
        x_f_ij = self.layer_f[k](torch.cat([x_i, x_j], dim=-1)).relu()
        x_e_ij = self.layer_e[k](edge_attr).relu()
        x_fe_ij = self.layer_fe[k](torch.cat([x_i, x_j, edge_attr], dim=-1)).relu()
        x_d_ij = self.layer_d[k](edge_attr).relu() * x_j
        z = torch.cat([x_f_ij, x_e_ij, x_fe_ij, x_d_ij], dim=-1)
        return z

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr:Tensor) -> Tensor:
        # TODO add the message passing stuff here for all the
        #  filters by concatenating
        for k in range(self.num_filters):
            if k == 0:
                z = self.forward_pass(x_i=x_i, x_j=x_j, edge_attr=edge_attr, k=k)
            else:
                z_k = self.forward_pass(x_i=x_i, x_j=x_j, edge_attr=edge_attr, k=k)
                z = torch.cat([z, z_k], dim=-1)
        x_out = self.linear_final(z).relu()
        return x_out


    def __repr__(self):
        return f'{self.__class__.__name__}(node_feat_dim={self.node_feat_dim}, edge_dim={self.edge_dim}, num_filters={self.num_filters}, add_self_loop={self.add_self_loop})'

if __name__ == "__main__":
    from dataloader import MNISTDataloader

    loader = MNISTDataloader(num_nodes=10, batch_size=32, train_val_split_ratio=0.2, seed=0, remove9=False,
                             testing=False)
    trainloader = loader.train_dataloader
    for i, (data, target) in enumerate(trainloader):
        geom_loader = loader.process_torch_geometric(data=data, sampling_strategy='random', num_samples=10, k=2,
                                                     polar=False)
        for batch in geom_loader:
            print(f'Torch Geometric Batch: {batch}')
        break
    net = ComboConv(node_feat_dim=1, edge_dim=2, add_self_loop=True, out_features=5)
    x, edge_idx, edge_wt = batch.x, batch.edge_index, batch.edge_attr
    out = net(x=x, edge_index=edge_idx, edge_attr=edge_wt)
    print('_' * 50)
    print(net)
    print('_' * 50)
    print(f'Network output shape: {out.shape}')
    print('_' * 50)
    print(f'Number of parameters in network: {sum(p.numel() for p in net.parameters())}')
    print('_' * 50)
