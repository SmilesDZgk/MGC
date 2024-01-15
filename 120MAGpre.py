import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor, coalesce
import settings
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from ogb.lsc import MAG240MDataset
import ipdb
import scipy.sparse as sp
import os.path as osp
import time
from dgl import GCNNorm
import os
tranform = GCNNorm()

def signPos(features):
    nids = torch.where(features<0)
    features[nids] = 0
    return features

def sym_normalize2(adj, selfloop=False):
    """Normalization by D^{-1/2} (A+I) D^{-1/2}."""
    if selfloop:
        adj += 1*sp.diags(np.ones(adj.shape[0]), 0)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    ipdb.set_trace()
    adj = d_inv_sqrt[:,np.newaxis]*adj*d_inv_sqrt[np.newaxis,:]
    
    
    return adj
def sym_normalize(adj, selfloop=False):
    """Normalization by D^{-1/2} (A+I) D^{-1/2}."""
    if selfloop:
        adj += 1*sp.diags(np.ones(adj.shape[0]), 0)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt, 0)
    adj = (d_mat_inv_sqrt.dot(adj)).dot(d_mat_inv_sqrt)
    
    # adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    
    return adj

def main():
    parser = argparse.ArgumentParser(description='LSC-papers120M (MLP)')
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--dropedge_rate', type=float, default=0.0)
    parser.add_argument('--model', type=str, default='sgc') # sgc, mgc, ssgc
    parser.add_argument("--pos", action='store_true',
                        help="positive or negative")
    args = parser.parse_args()

    dataset = MAG240MDataset(settings.DATAPATH+'/MAG240M/')

    t = time.perf_counter()
    print('Reading adjacency matrix...', end=' ', flush=True)
    path = f'{dataset.dir}/paper_to_paper_symmetric_gcn.pt'
    if osp.exists(path):
        adj_t = torch.load(path)
    else:
        path_sym = f'{dataset.dir}/paper_to_paper_symmetric.pt'
        if osp.exists(path_sym):
            adj_t = torch.load(path_sym)
        else:
            edge_index = dataset.edge_index('paper', 'cites', 'paper')
            edge_index = torch.from_numpy(edge_index)
            adj_t = SparseTensor(
                row=edge_index[0], col=edge_index[1],
                sparse_sizes=(dataset.num_papers, dataset.num_papers),
                is_sorted=True)
            adj_t = adj_t.to_symmetric()
            torch.save(adj_t, path_sym)
        adj_t = gcn_norm(adj_t, add_self_loops=True)
        torch.save(adj_t, path)
    print(f'Done! [{time.perf_counter() - t:.2f}s]')




    train_idx = dataset.get_idx_split('train')
    valid_idx = dataset.get_idx_split('valid')
    test_idx = dataset.get_idx_split('test-dev')
    num_features = dataset.num_paper_features

    pbar = tqdm(total=args.num_layers * (num_features // 128))
    pbar.set_description('Pre-processing node features')

    dataset.paper_label[train_idx]
    np.save(f'{dataset.dir}/label_train.npy', dataset.paper_label[train_idx])
    np.save(f'{dataset.dir}/label_valid.npy', dataset.paper_label[valid_idx])
    np.save(f'{dataset.dir}/label_test.npy', dataset.paper_label[test_idx])
    # return 


    x = dataset.paper_feat
    x = torch.from_numpy(x.astype(np.float32))
    np.save(f'{dataset.dir}/x_train.npy', x[train_idx].numpy())
    np.save(f'{dataset.dir}/x_valid.npy', x[valid_idx].numpy())
    np.save(f'{dataset.dir}/x_test.npy', x[test_idx].numpy())

    
    for j in range(0, num_features, 128):  # Run spmm in column-wise chunks...
        x = dataset.paper_feat[:, j:min(j + 128, num_features)]
        x = torch.from_numpy(x.astype(np.float32))
        if args.pos:
            x= signPos(x)
            post = 'pos'
        else:
            x = signPos(-x) # positive/ negative
            post = 'neg'


        for i in range(1, args.num_layers + 1):
            x = adj_t @ x


            np.save(f'{dataset.dir}/x_train_{i}_{j}.npy', x[train_idx].numpy())
            np.save(f'{dataset.dir}/x_valid_{i}_{j}.npy', x[valid_idx].numpy())
            np.save(f'{dataset.dir}/x_test_{i}_{j}.npy', x[test_idx].numpy())


            pbar.update(1)
    pbar.close()

    t = time.perf_counter()
    print('Merging node features...', end=' ', flush=True)
    mpx_train = np.zeros((len(train_idx), num_features))
    mpx_vaild = np.zeros((len(valid_idx), num_features))
    mpx_test = np.zeros((len(test_idx), num_features))

    avgx_train = 0
    avgx_vaild = 0
    avgx_test = 0

    for i in range(1, args.num_layers + 1):
        x_train, x_valid, x_test = [], [], []
        for j in range(0, num_features, 128):
            x_train += [np.load(f'{dataset.dir}/x_train_{i}_{j}.npy')]
            x_valid += [np.load(f'{dataset.dir}/x_valid_{i}_{j}.npy')]
            x_test += [np.load(f'{dataset.dir}/x_test_{i}_{j}.npy')]
        x_train = np.concatenate(x_train, axis=-1)
        x_valid = np.concatenate(x_valid, axis=-1)
        x_test = np.concatenate(x_test, axis=-1)
        np.save(f'{dataset.dir}/x_train_{i}_{post}.npy', x_train)
        np.save(f'{dataset.dir}/x_valid_{i}_{post}.npy', x_valid)
        np.save(f'{dataset.dir}/x_test_{i}_{post}.npy', x_test)

        if i>=2:
            mpx_train = np.max(np.stack([mpx_train,x_train], axis= 0),axis=0)
            mpx_vaild = np.max(np.stack([mpx_vaild,x_valid], axis= 0),axis=0)
            mpx_test = np.max(np.stack([mpx_test,x_test], axis= 0),axis=0)

        avgx_train = avgx_train+1.0/args.num_layers*((x_train))
        avgx_vaild = avgx_vaild+1.0/args.num_layers*((x_valid))
        avgx_test = avgx_test+1.0/args.num_layers*((x_test))
    
    np.save(f'{dataset.dir}/mpx_train_{i}_{post}.npy', mpx_train)
    np.save(f'{dataset.dir}/mpx_valid_{i}_{post}.npy', mpx_vaild)
    np.save(f'{dataset.dir}/mpx_test_{i}_{post}.npy', mpx_test)

    np.save(f'{dataset.dir}/avgx_train_{i}_{post}.npy', avgx_train)
    np.save(f'{dataset.dir}/avgx_valid_{i}_{post}.npy', avgx_vaild)
    np.save(f'{dataset.dir}/avgx_test_{i}_{post}.npy', avgx_test)



    print(f'Done! [{time.perf_counter() - t:.2f}s]')

    t = time.perf_counter()
    print('Cleaning up...', end=' ', flush=True)
    for i in range(1, args.num_layers + 1):
        for j in range(0, num_features, 128):
            os.remove(f'{dataset.dir}/x_train_{i}_{j}.npy')
            os.remove(f'{dataset.dir}/x_valid_{i}_{j}.npy')
            os.remove(f'{dataset.dir}/x_test_{i}_{j}.npy')

    print(f'Done! [{time.perf_counter() - t:.2f}s]')




if __name__ == "__main__":
    main()