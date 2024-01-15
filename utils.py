import os
from functools import namedtuple
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from sklearn.metrics import f1_score
import numpy as np
import argparse
import scipy.sparse as sp
import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset, CoauthorCSDataset, RedditDataset, PPIDataset
import copy
import pickle
from settings import *
from models import  MGC, MGCMLP, SGC, APPNP, GCNFA,GCN# GCN, SGC, SGC2, SGC3,
from utils_data import load_data
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from sklearn.metrics import accuracy_score, f1_score
import settings
from torch_sparse import SparseTensor
def init_config():
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--dataset", type=str, default="cora",
                        help="Dataset name ('cora', 'citeseer', 'pubmed', 'coathour').")
    parser.add_argument("--model", type=str, default="GCN",
                        help="Dataset name ('GCN', 'SGC').")
    parser.add_argument("--dropout1", type=float, default=0.0,
                        help="dropout probability")
    parser.add_argument("--dropout2", type=float, default=0.4,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--n-hid", type=int, default=16,
                        help="number of hidden gcn units")
    parser.add_argument("--act", type=str, default='relu')
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--eps", type=float, default=0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    parser.add_argument("--inductive", action='store_true',
                        help="default=False")
    parser.set_defaults(self_loop=True)
    args = parser.parse_args()
    set_seed(-1)

    if args.gpu < 0:
        args.device = 'cpu'
    else:
        args.device = 'cuda:%d'%args.gpu

    return args


def set_seed(id):
    # seeds = [104,202,303,405,505,610,770,870,909,42,7,9]
    seeds = [17,993,  199, 191, 61, 101,202,303,404,505,606,770,870,909,42,7,9]
    if id <12:
        seed = seeds[id]
    else:
        seed = 5*id+3
    
    # seed = id+1
    # print('===============', seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def adj_rw_norm(adj):
    """
    Normalize adj according to the method of rw normalization.
    """
    diag_shape = (adj.shape[0],adj.shape[1])
    D = adj.sum(1).flatten()
    norm_diag = sp.dia_matrix((1/D,0),shape=diag_shape)
    adj_norm = norm_diag.dot(adj)
    return adj_norm

def infAX(adj, features,selfloop=False):
    if selfloop:
        adj += 1*sp.diags(np.ones(adj.shape[0]), 0)
    D = np.array(adj.sum(1))
    M = D.sum()
    vec =  np.sqrt(D/M)
    # import ipdb
    # ipdb.set_trace()
    F = (vec*features).sum(axis=0)[np.newaxis, :]
    return vec*F


def sym_normalize(adj, selfloop=False):
    """Normalization by D^{-1/2} (A+I) D^{-1/2}."""
    if selfloop:
        adj += 1*sp.diags(np.ones(adj.shape[0]), 0)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt, 0)

    
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    M = rowsum.sum()
    U = np.sqrt(rowsum/M).squeeze()
    
    
    return adj
def sym_U(adj, selfloop=False):
    """Normalization by D^{-1/2} (A+I) D^{-1/2}."""
    if selfloop:
        adj += 1*sp.diags(np.ones(adj.shape[0]), 0)
    rowsum = np.array(adj.sum(1))
    M = rowsum.sum()
    U = np.sqrt(rowsum/M).squeeze()
    
    return U
    

def normalization(features):
    MAX = features.max()
    MIN = features.min()
    features = (features-MIN)#/(MAX-MIN)
    return features, MIN

def signSplit(features):
    pids = np.where(features>0)
    nfeat = features.clone()
    nfeat[pids] = 0
    pfeat = features - nfeat
    return pfeat, -nfeat

def signSplitTensor(features):
    pids = torch.where(features>0)
    nfeat = features.clone()
    nfeat[pids] = 0
    pfeat = features - nfeat
    return pfeat, -nfeat


def set_dataset(dataname, splits_id=0):

    if dataname == 'cora':
        data = CoraGraphDataset()
        g = data[0]
    elif dataname == 'citeseer':
        data = CiteseerGraphDataset()
        g = data[0]
    elif dataname == 'pubmed':
        data = PubmedGraphDataset()
        g = data[0]
    elif dataname == 'coauthor':
        data = PubmedGraphDataset()
        g = data[0]
    
    elif dataname in ['film', 'texas', 'wisconsin', 'cornell','chameleon']:
        g = load_data(dataname,splits_id=splits_id)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataname))
    return g

def full_load_data(dataset_name, splits_id=0):
    if dataset_name in {'cora', 'citeseer', 'pubmed'}:
        g, features, labels = full_load_citation(dataset_name)
        labels = np.argmax(labels, axis=-1)
        features = features.todense()
        splits_file_path = DATAPATH+'/fulldata/splits/'+'%s_split_0.6_0.2_%d.npz'%(dataset_name,splits_id)
  
        with np.load(splits_file_path) as splits_file:
            train_mask = splits_file['train_mask']
            val_mask = splits_file['val_mask']
            test_mask = splits_file['test_mask']
        
        num_features = features.shape[1]
        num_labels = len(np.unique(labels))
        assert (np.array_equal(np.unique(labels), np.arange(len(np.unique(labels)))))

        features = torch.FloatTensor(features)
        labels = torch.LongTensor(labels)
        train_mask = torch.BoolTensor(train_mask)
        val_mask = torch.BoolTensor(val_mask)
        test_mask = torch.BoolTensor(test_mask)

        g.ndata['feat'] = features
        g.ndata['label'] = labels
        g.ndata['train_mask'] = train_mask
        g.ndata['val_mask'] = val_mask
        g.ndata['test_mask'] = test_mask
        train_idx = torch.where(g.ndata['train_mask']>0)[0]
        val_idx = torch.where(g.ndata['val_mask']>0)[0]
        test_idx = torch.where(g.ndata['test_mask']>0)[0]
        split_idx = {'train':train_idx, 'valid':val_idx, 'test':test_idx}
        return g, split_idx

def set_dataset_batch(dataname, splits_id=0):

    if dataname == 'pubmed':
        data = PubmedGraphDataset()
        g = data[0]
        train_idx = torch.where(g.ndata['train_mask']>0)[0]
        val_idx = torch.where(g.ndata['val_mask']>0)[0]
        test_idx = torch.where(g.ndata['test_mask']>0)[0]
        split_idx = {'train':train_idx, 'valid':val_idx, 'test':test_idx}
    elif dataname == 'coauthor':
        data = PubmedGraphDataset()
        g = data[0]
        train_idx = torch.where(g.ndata['train_mask']>0)[0]
        val_idx = torch.where(g.ndata['val_mask']>0)[0]
        test_idx = torch.where(g.ndata['test_mask']>0)[0]
        split_idx = {'train':train_idx, 'valid':val_idx, 'test':test_idx}
    elif dataname in ['ogbn-products', 'ogbn-arxiv', 'ogbn-mag', 'ogbn-papers100M']:
        dataset = DglNodePropPredDataset(name=dataname, root=settings.DATAPATH)
        split_idx = dataset.get_idx_split()
        g, labels = dataset[0]
        if dataname == "ogbn-arxiv":
            g = dgl.add_reverse_edges(g, copy_ndata=True)
        elif dataname =='ogbn-papers100M':
            g = dgl.add_reverse_edges(g, copy_ndata=True)
            labels = labels.long()
        elif dataname == 'ogbn-mag':
            # path = os.path.join(args.emb_path, f"{args.pretrain_model}_mag")
            
            labels = labels["paper"]
            split_idx = {k: split_idx[k]['paper'] for k in split_idx}
            g = dgl.edge_type_subgraph(g, [('paper', 'cites', 'paper')])
            # train_nid = train_nid["paper"]
            # val_nid = val_nid["paper"]
            # test_nid = test_nid["paper"]
            # features = g.nodes['paper'].data['feat']
            # g.ndata['feat'] = features

            # author_emb = torch.load(os.path.join(path, "author.pt"), map_location=torch.device("cpu")).float()
            # topic_emb = torch.load(os.path.join(path, "field_of_study.pt"), map_location=torch.device("cpu")).float()
            # institution_emb = torch.load(os.path.join(path, "institution.pt"), map_location=torch.device("cpu")).float()

            # g.nodes["author"].data["feat"] = author_emb.to(device)
            # g.nodes["institution"].data["feat"] = institution_emb.to(device)
            # g.nodes["field_of_study"].data["feat"] = topic_emb.to(device)
            # g.nodes["paper"].data["feat"] = features.to(device)
            # paper_dim = g.nodes["paper"].data["feat"].shape[1]
            # author_dim = g.nodes["author"].data["feat"].shape[1]
            # if paper_dim != author_dim:
            #     paper_feat = g.nodes["paper"].data.pop("feat")
            #     rand_weight = torch.Tensor(paper_dim, author_dim).uniform_(-0.5, 0.5)
            #     g.nodes["paper"].data["feat"] = torch.matmul(paper_feat, rand_weight.to(device))
            #     print(f"Randomly project paper feature from dimension {paper_dim} to {author_dim}")

            
        g.ndata['train_mask']= torch.zeros((labels.size(0)), dtype=torch.bool)
        g.ndata['val_mask']=torch.zeros((labels.size(0)), dtype=torch.bool)
        g.ndata['test_mask']=torch.zeros((labels.size(0)), dtype=torch.bool)
        g.ndata["label"] = labels.squeeze()
        g.ndata['feat'] = g.ndata['feat'].float()
        g.ndata['train_mask'][split_idx['train']]=True
        g.ndata['val_mask'][split_idx['valid']] =True
        g.ndata['test_mask'][split_idx['test']]=True
        # g.ndata['feat'] = g.ndata['feat']/g.ndata['feat'].sum(dim=1,keepdim=True)
    elif dataname == 'reddit':
        data = RedditDataset()
        # import ipdb
        # ipdb.set_trace()
        g = data[0]
        train_idx = torch.where(g.ndata['train_mask']>0)[0]
        val_idx = torch.where(g.ndata['val_mask']>0)[0]
        test_idx = torch.where(g.ndata['test_mask']>0)[0]
        split_idx = {'train':train_idx, 'valid':val_idx, 'test':test_idx}
    
    elif dataname  == 'flickr':
        from torch_geometric.datasets import Flickr
        pyg_data = Flickr(os.path.join(DATAPATH, "flickr"))[0]
        # import ipdb
        # ipdb.set_trace()
        feat = pyg_data.x
        labels = pyg_data.y
        # labels = torch.argmax(labels, dim=1)
        u, v = pyg_data.edge_index
        g = dgl.graph((u, v))
        g.ndata['feat'] = feat
        g.ndata['label'] = labels
        g.ndata['train_mask'] = pyg_data.train_mask
        g.ndata['val_mask'] = pyg_data.val_mask
        g.ndata['test_mask'] = pyg_data.test_mask
        train_idx = torch.where(g.ndata['train_mask']>0)[0]
        val_idx = torch.where(g.ndata['val_mask']>0)[0]
        test_idx = torch.where(g.ndata['test_mask']>0)[0]
        split_idx = {'train':train_idx, 'valid':val_idx, 'test':test_idx}
        
    elif dataname == 'ppi':
        data = load_ppi_data(DATAPATH)
        g = data.g
        train_idx = torch.where(g.ndata['train_mask']>0)[0]
        val_idx = torch.where(g.ndata['val_mask']>0)[0]
        test_idx = torch.where(g.ndata['test_mask']>0)[0]
        split_idx = {'train':train_idx, 'valid':val_idx, 'test':test_idx}

    else:
        raise ValueError('Unknown dataset: {}'.format(dataname))
    return g, split_idx

def set_criteron(dataname):
    if dataname in ['flickr', 'reddit', 'cora', 'citeseer', 'pubmed', 'coauthor','film', 'texas', 'wisconsin', 'corenell']:
        criterion = nn.CrossEntropyLoss()
        multi_class=False
    elif dataname in ['ppi', 'ppi-large', 'amazon', 'yelp']:
        criterion = nn.BCEWithLogitsLoss()
        multi_class=True
    else:
        criterion = nn.CrossEntropyLoss()
        multi_class=False
        # raise ValueError('Unknown dataset: {}'.format(dataname))
    return criterion, multi_class

def set_model(args):
    if args.model in ['MGC', 'mgc']:
        return MGC(args).to(args.device)
    if args.model in ['MGCMLP', 'mgcmlp']:
        return MGCMLP(args).to(args.device)
        # return GCN(args).to(args.device)
    # if args.model in ['GCN', 'gcn']:
    #     return GCN(args).to(args.device)
    elif args.model in ['SGC', 'sgc', 'SSGC', 'ssgc']:
        return SGC(args).to(args.device)
    elif args.model in ['APPNP', 'appnp']:
        return APPNP(args).to(args.device)
    elif args.model in ['GCNFA', 'gcnfa']:
        return GCNFA(args).to(args.device)
    # elif args.model in ['SGC3', 'sgc3']:
    #     return SGC3(args).to(args.device)
    else:
        raise ValueError('Unknown model: {}'.format(args.model))

def degree_eps(adj):
    D = adj.sum(1).flatten()
    # std = np.std(D)
    # mu = np.mean(D)
    # D = D/D.max()
    # # D = (D-mu)/std*0.05+0.005
    D = 1.0/D

    return D


def curvature_eps(dataname):
    curva_path = DATAPATH+'/%s/curvature_%s.pkl'%(dataname,dataname)
    eps = pickle.load(open(curva_path, 'rb'))
    mu = np.mean(eps)
    std = np.std(eps)
    eps = (np.array(eps)-mu)/std
    return np.array(eps)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    if len(sparse_mx.row) == 0 and len(sparse_mx.col) == 0:
        indices = torch.LongTensor([[], []])
    else:
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    mx_tensor = torch.sparse.FloatTensor(indices, values, shape)

    return mx_tensor

def sparse_mx_to_torch_sparse_tensor2(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    if len(sparse_mx.row) == 0 and len(sparse_mx.col) == 0:
        indices = torch.LongTensor([[], []])
    else:
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)

    mx_tensor = SparseTensor(row=indices[0], col=indices[1],value=values,sparse_sizes=shape)
    # mx_tensor = torch.sparse.FloatTensor(indices, values, shape)

    return mx_tensor

class ResultRecorder:
    def __init__(self, note):
        self.train_loss_record = []
        self.train_acc_record = []
        self.loss_record = []
        self.acc_record = []
        self.best_acc = -1.0
        self.best_loss = 10**3
        self.best_model = None
        self.note = note
        self.sample_time = []
        self.compute_time = []
        self.state_dicts = []
        self.grad_norms = []
        self.not_imporved = 0
        
    def update(self, train_loss, train_acc, loss, acc, model, sample_time=0, compute_time=0):
        self.sample_time += [sample_time]
        self.compute_time += [compute_time]

        self.train_loss_record += [train_loss]
        self.train_acc_record += [train_acc]
        
        self.loss_record += [loss]
        self.acc_record += [acc]
            
        # if self.best_acc is None:
        #     self.best_acc = acc
        # elif self.best_acc < acc:
        #     self.best_acc = acc
        #     self.best_model = copy.deepcopy(model)
        if self.best_acc<acc:
            self.best_acc = acc
            self.best_model = copy.deepcopy(model)
        else:
            self.not_imporved+=1
            if self.not_imporved>=30:
                return False
        return True

        # if self.best_loss> loss and self.best_acc<acc:
        #     self.best_loss = loss
        #     self.best_acc = acc
        #     self.best_model = copy.deepcopy(model)


def load_ppi_data(root):
    DataType = namedtuple('Dataset', ['num_classes', 'g'])
    adj_full = sp.load_npz(os.path.join(root, 'ppi', 'adj_full.npz'))
    G = dgl.from_scipy(adj_full)
    nodes_num = G.num_nodes()
    role = json.load(open(os.path.join(root, 'ppi','role.json'),'r'))
    tr = list(role['tr'])
    te = list(role['te'])
    va = list(role['va'])
    mask = np.zeros((nodes_num,), dtype=bool)
    train_mask = mask.copy()
    train_mask[tr] = True
    val_mask = mask.copy()
    val_mask[va] = True
    test_mask = mask.copy()
    test_mask[te] = True
    
    G.ndata['train_mask'] = torch.tensor(train_mask, dtype=torch.bool)
    G.ndata['val_mask'] = torch.tensor(val_mask, dtype=torch.bool)
    G.ndata['test_mask'] = torch.tensor(test_mask, dtype=torch.bool)

    feats=np.load(os.path.join(root, 'ppi', 'feats.npy'))
    G.ndata['feat'] = torch.tensor(feats, dtype=torch.float)

    class_map = json.load(open(os.path.join(root, 'ppi', 'class_map.json'), 'r'))
    labels = np.array([class_map[str(i)] for i in range(nodes_num)])
    G.ndata['label'] = torch.tensor(labels, dtype=torch.float)
    data = DataType(g=G, num_classes=labels.shape[1])
    return data

import pickle as  pkl
import networkx as nx
import sys
def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def full_load_citation(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(DATAPATH+'/fulldata/'+"/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(DATAPATH+'/fulldata/'+"/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    # adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    graph = dgl.from_networkx(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]


    return graph, features, labels 

def ACCEvaluator(y_pred, y_true):

        return accuracy_score(y_true.cpu(), y_pred.cpu())

def F1Evaluator( y_pred, y_true):

        return f1_score(y_true.cpu(), y_pred.cpu(), average='micro')