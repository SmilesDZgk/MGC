import torch
import torch.nn as nn
import numpy as np
import math
from .activation import set_activation
from .initialization_utils import *
from torch_sparse import spspmm, spmm, SparseTensor, index_select
import ipdb

#########################################################
#########################################################
#########################################################    
def spspadd(indexA, valueA, indexB, valueB, M,N):
    rowA, colA = indexA
    rowB, colB = indexB
    row = torch.cat([rowA, rowB], dim=0)
    col = torch.cat([colA, colB], dim=0)

    if valueA is not None and valueB is not None:
        value = torch.cat([valueA, valueB], dim=0)

    sparse_sizes = (M, N)

    out = SparseTensor(row=row, col=col, value=value,
                        sparse_sizes=sparse_sizes)
    out = out.coalesce(reduce='sum')
    row, col, value = out.coo()
    index = torch.stack([row,col])
    return index, value

def spspmax(indexA, valueA, indexB, valueB, M,N):
    indexsub, valuesub = spspadd(indexA, valueA, indexB, -valueB, M,N)
    ids = torch.where(valuesub>0)[0]
    indexsub, valuesub 
    index, value = spspadd(indexsub, valuesub, indexB, valueB, M,N)

    return index, value



class MGCPreAppro(nn.Module):
    def __init__(self, n_layers,adj,x):
        super(MGCPreAppro, self).__init__()
        self.n_layers = n_layers 
           
        self.adj = adj
        self.x= x
    def process(self, inductive=False):
        sumout=self.create_f(mink=1)
        return [sumout]

    
    def create_f(self, mink=1):
        N = self.adj.size(0)
        x = self.x 
        out = torch.zeros_like(x)#+infA
        for ell in range(self.n_layers):
            # x = torch.mm(self.adj,x)
            x = torch.spmm(self.adj,x)
            if ell>=mink:
                out = torch.stack([out,x])
                out,inds = torch.max(out, dim=0) 
                # ipdb.set_trace()
                # out = out[inds]
                # x = out
        return out
        
        
    


class MGCPreSparse(nn.Module):
    def __init__(self, n_layers,adj,x):
        super(MGCPreSparse, self).__init__()
        self.n_layers = n_layers 
           
        self.adj = adj
        # ipdb.set_trace()
        # indexadj = adj.coalesce().indices()
        # valueadj = adj.coalesce().values()
        # self.adj2 = SparseTensor(row=indexadj[0], col=indexadj[1],value=valueadj,sparse_sizes=adj.size())
        self.adj2 = SparseTensor.from_torch_sparse_coo_tensor(self.adj)
        # self.coladj = self.adj.storage.col()
        # self.rowadj = self.adj.storage.row()
        # self.indiceadj = torch.stack([self.adj.storage.row(), self.adj.storage.col()], dim=0)
        # self.valueadj = self.adj.storage.value()
        self.x= x
        # SparseTensor().storage.col
        # self.D = (self.adj2>0).sum(dim=1)
        # self.M = self.D.sum()
    def process(self,ids):
        N = self.adj.size(0)
        batchA = index_select(self.adj2,0,ids)
        batchA = batchA.to_dense()
        M = batchA.size(0)
        Am = self.creta_A_sparse(self.n_layers, batchA)
        x2 = torch.mm(Am,self.x)
        xs = x2
        return xs
    
    def creta_A_sparse(self, n_layers, batchA):
        # if adj == None:
        N = self.adj.size(0)
        B = batchA.size(0)
        A = batchA.T
        DB = (batchA>0).sum(dim=1) #B
        # infA = (DB.unsqueeze(0)*self.D.unsqueeze(1))**(1/2)/(self.M)
        Am = torch.zeros_like(A)#+infA
        for ell in range(n_layers-1):
            A =torch.mm(self.adj, A)
            # A = spmm(self.indiceadj, self.valueadj,B,N,A)
            Am = torch.stack([Am,A])
            Am,_ = torch.max(Am, dim=0)
        Am= Am
        #/Am.sum(dim=0,keepdim=True)
        return Am.T

class MGCPre(nn.Module):
    def __init__(self, n_layers,adj,x):
        super(MGCPre, self).__init__()
        self.n_layers = n_layers        
        self.adj = adj
        # self.indexadj = adj.coalesce().indices()
        # self.valueadj = adj.coalesce().values()
        # self.adj2 = SparseTensor(row=self.indexadj[0], col=self.indexadj[1],value=self.valueadj,sparse_sizes=adj.size())
        self.x= x
        # self.D = SparseTensor(row=self.indexadj[0], col=self.indexadj[1],value=torch.ones_like(self.valueadj),sparse_sizes=adj.size()).sum(dim=1)
        # self.M = self.D.sum()
    def process(self,ids=None):
        N = self.adj.size(0)
        xs=[]
        if ids is None:
            Am = self.creta_A_dense(self.n_layers, self.adj,1)
            x2 = torch.mm(Am,self.x)
            # Am = self.creta_APPNP_dense(self.n_layers, self.adj,0)
            # x2 = torch.mm(Am,self.x)
            # x2 = x2/torch.norm(x2, dim=1, keepdim=True)*50
            xs.append(x2)

        return xs
    
    
    def creta_A_dense(self, n_layers, batchA, mink=0):
        # if adj == None:
        N = self.adj.size(0)
        B = batchA.size(0)
        A = batchA.to_dense()
        DB = (A>0).sum(dim=1) #B
        # infA = (DB.unsqueeze(0)*DB.unsqueeze(1))**(1/2)/(self.M)
        A = torch.eye(N).to(batchA.device)
        Am = torch.zeros_like(A)#+infA
        # Am = A
        # Am = A
        for ell in range(n_layers):
            A = torch.mm(self.adj,A)
            if ell>=mink:
                Am = torch.stack([Am,A])
                Am,_ = torch.max(Am, dim=0)
        # Am = DB.unsqueeze(1)**(-1/2)*Am*DB.unsqueeze(0)**(1/2)
        return Am*1.2


class MGC(nn.Module):
    def __init__(self, args, steps = 2):
        super(MGC, self).__init__()
        self.n_layers = args.n_layers
        self.n_hid = args.n_hid
        self.linears = nn.ModuleList()
        self.bns=nn.ModuleList()
        for _ in range(steps):
            self.linears.append(nn.Linear(args.n_feat, int(args.n_hid), bias=True))
            self.bns.append(nn.BatchNorm1d(args.n_hid)  )
           
        self.linears.append(nn.Linear(int(args.n_hid), args.n_classes, bias=False))
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout1)
        self.dropout2 = nn.Dropout(args.dropout2)

        #
        # coffs = torch.from_numpy(np.array([0.75, 0.25]))
        coffs = torch.from_numpy(np.array([1-args.alpha, args.alpha])) 
        self.coffs = nn.Parameter(coffs, requires_grad=False)

        self.norm = nn.LayerNorm(args.n_feat)  

        gain = nn.init.calculate_gain("relu")
        # print(gain)
        # nn.init.xavier_normal_(self.linears[0].weight, gain=gain)
        # nn.init.xavier_normal_(self.linears[1].weight, gain=gain)
        # nn.init.zeros_(self.linears[0].bias)
        # nn.init.xavier_normal_(self.linear_in2.weight, gain=gain)
        # nn.init.zeros_(self.linear_in2.bias)
        nn.init.xavier_normal_(self.linears[-1].weight, gain=gain)
        # nn.init.zeros_(self.linears[-1].bias)
            
    def forward(self, xs):
        sumx = 0
        for i,x in enumerate(xs):
            if i==0:
                x = self.dropout(x)
                # pass
            else:
                x = self.dropout(x)
                # pass
            x= self.linears[i](x)
            # x = self.bns[i](x)
            x = self.activation(x)
            # x = self.dropout2(x)
            sumx+=self.coffs[i]*x 

        x1 = self.dropout2(sumx)
        # x1 = sumx
        
        x = self.linears[-1](x1)
        return x

class MGCMLP(nn.Module):
    def __init__(self, args, steps = 2):
        super(MGCMLP, self).__init__()
        self.n_layers = args.n_layers
        self.n_hid = args.n_hid
        self.linears = nn.ModuleList()
        self.bns=nn.ModuleList()
        for _ in range(steps):
            self.linears.append(nn.Sequential(
            nn.Linear(args.n_feat, int(args.n_hid), bias=True),
            nn.BatchNorm1d(int(args.n_hid)),
            nn.ReLU(),
            nn.Dropout(args.dropout1),
            nn.Linear(int(args.n_hid), args.n_hid, bias=True)
            ))
            self.bns.append(nn.BatchNorm1d(args.n_hid))
        self.linears.append(nn.Linear(args.n_hid, int(args.n_hid), bias=True))
        self.bn =  nn.BatchNorm1d(int(args.n_hid))
        self.linears.append(nn.Linear(int(args.n_hid), args.n_classes, bias=True))
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout1)
        
        self.dropout2 = nn.Dropout(args.dropout2)

        #
        coffs = torch.from_numpy(np.array([0.95, 0.05])) 
        self.coffs = nn.Parameter(coffs, requires_grad=False)

        

        self.norm = nn.LayerNorm(args.n_feat)  

        gain = nn.init.calculate_gain("relu")
        # print(gain)
        # nn.init.xavier_normal_(self.linears[0].weight, gain=gain)
        # nn.init.xavier_normal_(self.linears[1].weight, gain=gain)
        # nn.init.zeros_(self.linears[0].bias)
        # nn.init.xavier_normal_(self.linear_in2.weight, gain=gain)
        # nn.init.zeros_(self.linear_in2.bias)
        nn.init.xavier_normal_(self.linears[-1].weight, gain=gain)
        nn.init.zeros_(self.linears[-1].bias)
            
    def forward(self, xs):
        sumx = 0
        for i,x in enumerate(xs):
            if i==0:
                x = self.dropout(x)
            else:
                x = self.dropout(x)
                # pass
            x= self.linears[i](x)
            # x = self.bns[i](x)
            x = self.activation(x)
            sumx+=self.coffs[i]*x 
        x1 = self.dropout2(sumx)
        
        x1 = self.linears[-2](x1)
        # x1 = self.bn(x1)
        x1 = self.activation(x1)
        x1 = self.dropout2(x1)
        x = self.linears[-1](x1)
        return x


#########################################################
#########################################################
#########################################################    

class SGCPre(nn.Module):
    def __init__(self, n_layers,adj,x):
        super(SGCPre, self).__init__()
        self.n_layers = n_layers        
        self.adj = adj
        self.x= x
    def process(self):
        x = self.x
        for _ in range(self.n_layers):
            x=torch.mm(self.adj,x)
        return x

class SSGCPre(nn.Module):
    def __init__(self, n_layers,adj,x):
        super(SSGCPre, self).__init__()
        self.n_layers = n_layers        
        self.adj = adj
        self.x= x
    def process(self):
        x = self.x
        xs = 0
        for _ in range(self.n_layers):
            x=torch.mm(self.adj,x)
            xs+=x
        return xs/self.n_layers

class SGC(nn.Module):
    def __init__(self, args):
        super(SGC, self).__init__()
        self.n_layers = args.n_layers
        self.n_hid = args.n_hid
        self.linear_out = nn.Linear(args.n_feat, args.n_classes)
        self.dropout = nn.Dropout(args.dropout1)

        gain = nn.init.calculate_gain("relu")
        # nn.init.xavier_normal_(self.linear_out.weight, gain=gain)
        # nn.init.zeros_(self.linear_out.bias)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear_out(x)
        return x


#########################################################
#########################################################
#########################################################    

class APPNP(nn.Module):
    def __init__(self, args):
        from .layers import APPNPLayer
        super(APPNP, self).__init__()
        self.n_layers = args.n_layers
        self.n_hid = args.n_hid
        
        self.gcs = nn.ModuleList()
        for _ in range(self.n_layers):
            self.gcs.append(APPNPLayer())
        self.linear_in = nn.Linear(args.n_feat, args.n_hid)
        self.linear_out = nn.Linear(args.n_hid, args.n_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout1)
        self.adj=None
    
    def set_adj(self,adj):
        self.adj = adj

    def forward(self, x):
        x = self.linear_in(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.linear_out(x)
        x_0 = x.clone()
        for ell in range(len(self.gcs)):
            alpha = 0.9
            x = self.gcs[ell](x, self.adj, x_0, alpha)
        # x = self.linear_out(x)
        return x
#########################################################
#########################################################
#########################################################    


class GCNFA(nn.Module):
    def __init__(self, args):
        from .layers import GraphConv
        super(GCNFA, self).__init__()
        self.n_layers = args.n_layers
        self.n_hid = args.n_hid
        
        self.gcs = nn.ModuleList()
        
        for i in range(args.n_layers-1):
            if i==0:
                self.gcs.append(GraphConv(args.n_feat,  args.n_hid, bias=False))
            else:
                self.gcs.append(GraphConv(args.n_hid,  args.n_hid, bias=False))
        self.linear = GraphConv(args.n_hid, args.n_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout1)
        self.adj=None
    
    def set_adj(self,adj):
        self.adj = adj

    def forward(self, x):
        for ell in range(len(self.gcs)):
            x = self.gcs[ell](x, self.adj)
            x = self.relu(x)
            x = self.dropout(x)
        x2 = x.mean(dim=0,keepdim=True)
        # ipdb.set_trace()
        x2 = x2.expand(x.size(0), -1)
        x = x+x2
        x = self.linear(x, self.adj)
        return x


class GCN(nn.Module):
    def __init__(self, args):
        from .layers import GraphConv
        super(GCN, self).__init__()
        self.n_layers = args.n_layers
        self.n_hid = args.n_hid
        
        self.gcs = nn.ModuleList()
        
        for i in range(1):
            if i==0:
                self.gcs.append(GraphConv(args.n_feat,  args.n_hid, bias=True))
            else:
                self.gcs.append(GraphConv(args.n_hid,  args.n_hid, bias=True))
        # self.linear = GraphConv(args.n_hid, args.n_classes)
        self.linear = nn.Linear(args.n_hid, args.n_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout2)
        self.adj=None
    
    def set_adj(self,adj):
        self.adj = adj

    def forward(self, x):
        x = self.dropout(x)
        for ell in range(len(self.gcs)):
            x1 = self.gcs[ell](x, self.adj)
            x1 = self.relu(x1)
            if ell>0:
                x = x+self.dropout(x1)
            else:
                x = self.dropout(x1)
        
        # x = self.linear(x, self.adj)
        x = self.linear(x)
        return x