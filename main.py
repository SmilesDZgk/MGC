import sys
import torch
import torch.nn as nn
import torch.optim as optim
from utils import *
import dgl
from sklearn.metrics import accuracy_score
from tqdm import trange
import numpy as np
import copy 
import time
import pickle
import os
from models import MGCPre, SGCPre, SSGCPre, MGCPreAppro
import pdb


def evaluate(model, features, labels, mask, loss_fcn):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        loss = loss_fcn(logits[mask], labels[mask])
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels),loss

def main(args):
        
    g = set_dataset(args.dataset, args.split)
    # g,_ = set_dataset_batch(args.dataset, args.split)
    # g = full_load_data(args.dataset, args.split)
    # g = g.to(args.device)
    # ipdb.set_trace()
    loss_fcn, multi_class = set_criteron(args.dataset)
    # g = data[0].int().to(args.device)

    features = g.ndata['feat'].to(args.device)
    # features,MIN = normalization(features)
    labels = g.ndata['label'].to(args.device)
    train_mask = g.ndata['train_mask'].to(args.device)
    val_mask = g.ndata['val_mask'].to(args.device)
    test_mask = g.ndata['test_mask'].to(args.device)
    args.n_feat = features.shape[1]
    args.n_classes = torch.max(labels).item()+1
    # ipdb.set_trace()

    g = dgl.remove_self_loop(g)
    if args.self_loop:
        g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()
    adj = g.adj()


    adj = sp.csr_matrix((adj.val.cpu().numpy(), (adj.row.cpu().numpy(), adj.col.cpu().numpy())), adj.shape)

    adj = sym_normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj).to(args.device)
    if args.model in ['MGC', 'mgc','MGCMLP', 'mgcmlp']:
        premodel = MGCPre(args.n_layers,adj, features)
        x2 = premodel.process()
        if args.inductive:
            idstrain = train_mask+ val_mask
            adjt = g.subgraph(idstrain).adj(scipy_fmt='csr')
            adjt = sym_normalize(adjt)
            adjt = sparse_mx_to_torch_sparse_tensor(adjt).to(args.device)
            featuret = features[idstrain]
            premodel = MGCPreAppro(args.n_layers,adjt, featuret)
            x2[0][idstrain] = premodel.process()[0]
    elif args.model in ['SGC', 'sgc']:
        premodel = SGCPre(args.n_layers,adj, features)
        x2 = premodel.process()
    elif args.model in ['SSGC', 'ssgc']:
        path = settings.DATAPATH +'/'+args.dataset+'_%d_mainssgc_pre.pkl'%(args.n_layers)
        if os.path.exists(path):
            x2 = pickle.load(open(path, 'rb'))
            x2 = torch.from_numpy(x2).to(args.device)
        else:
            premodel = SSGCPre(args.n_layers,adj.to(args.device), features)
            x2 = premodel.process()
            if args.inductive:
                idstrain = train_mask+ val_mask
                adjt = g.subgraph(idstrain).adj(scipy_fmt='csr')
                adjt = sym_normalize(adjt)
                adjt = sparse_mx_to_torch_sparse_tensor(adjt).to(args.device)
                featuret = features[idstrain]
                premodel = SSGCPre(args.n_layers,adjt, featuret)
                x2[idstrain] = premodel.process()
            pickle.dump(x2.cpu().numpy(), open(path, 'wb'))

    
    features = [x2[0]]+[features]

    # initialize graph
    def training():
        model = set_model(args)
        # model.set_adj(adj)
        # use optimizer
        optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)

        results = ResultRecorder(note="GCN (L=%d)"%args.n_layers)
        tbar = trange(args.n_epochs, desc='Training Epochs')
        dur = []
        for epoch in tbar:
            model.train()
            if epoch >= 3:
                t0 = time.time()
            # forward
            logits = model.forward(features)
            loss = loss_fcn(logits[train_mask], labels[train_mask])

            logits = logits[train_mask]
            _, indices = torch.max(logits, dim=1)
            correct = torch.sum(indices == labels[train_mask])
            train_acc = correct.item() * 1.0 / len(labels),loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch >= 3:
                dur.append(time.time() - t0)
            if epoch == 100:
                # model.coffs.requires_grad = True
                optimizer = torch.optim.Adam(model.parameters(),lr=args.lr/2,weight_decay=args.weight_decay)

            acc, val_loss = evaluate(model, features, labels, val_mask, loss_fcn)

            tbar.set_postfix(loss=loss.item(),
                            val_loss=val_loss.item(),
                            val_score=acc)

            results.update(loss, train_acc,val_loss, acc, model)
        acc, test_loss = evaluate(results.best_model.to(features[0].device), features, labels, test_mask, loss_fcn)
        results.test_loss = test_loss
        results.test_acc = acc
        print('Test_loss: %.4f | test_acc: %.4f' % (test_loss, acc))
        print('---------------------------------------------------')
        return results
    

    set_seed(0)
    result = training()

   

if __name__ == '__main__':
    args = init_config()
    print(args)
    main(args)

