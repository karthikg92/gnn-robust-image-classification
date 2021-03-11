import math
import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GraphConvolution(Module):
    """
        Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
        Parameters:
        -----------
        in_features: int
            The input feature dimension
        out_features: int
            The output feature dimension
        bias: bool
            If we want the bias term or not
    """
    def __init__(self, in_features:int, out_features:int, use_bias:bool=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if self.use_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """
            Initialise the weights of the networks uniformly 
            with stddev=1/sqrt(weight_size)
        """
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_parameters_xavier(self):
        """
            Xavier initialisation
        """
        nn.init.xavier_normal_(self.weight.data, gain=0.02) # Implement Xavier Uniform
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def reset_parameters_kaiming(self):
        """
            Kaiming initialisation
        """
        nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def forward(self, input:torch.tensor, adj:torch.tensor):
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
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        output = support
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + f'(in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias}'
        # return self.__class__.__name__ + ' (' \
        #        + str(self.in_features) + ' -> ' \
        #        + str(self.out_features) + ')'