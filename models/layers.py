import torch
import torch.nn as nn

from .initialization_utils import uniform, zeros,xavier_uniform,kaiming_uniform,xavier_normal
import ipdb

class GraphConv(nn.Module):
    def __init__(self, n_in, n_out, bias=True):
        super(GraphConv, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in, n_out, bias=bias)
        
        uniform(n_in, self.linear.weight)
        # kaiming_uniform(self.linear.weight,n_in,a=10)
        # xavier_uniform(self.linear.weight)
        # xavier_normal(self.linear.weight)
        # nn.init.normal(self.linear.weight, std=0.01)
        zeros(self.linear.bias)
        
    def forward(self, x, adj):
        x1 = torch.spmm(adj, x)
        x2 = self.linear(x1)
        return x2
    
class SGCLayer(nn.Module):
    def __init__(self, ):
        super(SGCLayer, self).__init__()
    def forward(self, x, adj):
        return torch.spmm(adj, x)
    
class APPNPLayer(nn.Module):
    def __init__(self, ):
        super(APPNPLayer, self).__init__()
    def forward(self, x, adj, x_0, alpha):  
        x = alpha*torch.spmm(adj, x)+(1-alpha)*x_0
        return x
    
class GCNIILayer(nn.Module):
    def __init__(self, n_in, n_out, bias=False):
        super(GCNIILayer, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in, n_out, bias=bias)
        
        uniform(n_in, self.linear.weight)
        zeros(self.linear.bias)
        
    def forward(self, x, adj, x_0, alpha, beta):  
        x = alpha*torch.spmm(adj, x)+(1-alpha)*x_0
        x = beta*self.linear(x)+(1-beta)*x
        return x
    
class DecoupleConv(nn.Module):
    def __init__(self, n_in, n_out, bias=False):
        super(DecoupleConv, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in, n_out, bias=bias)
        
        uniform(n_in, self.linear.weight)
        zeros(self.linear.bias)
        
    def forward(self, x, adj, identity_map_weight):
        x = torch.spmm(adj, x)
        x_w = identity_map_weight*self.linear(x)+(1-identity_map_weight)*x
        return x, x_w