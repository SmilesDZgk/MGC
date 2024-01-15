import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy 
import time
import pickle
import os
import ipdb
import settings
import random
import argparse


def init_config():
    parser = argparse.ArgumentParser(description='GCN')

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
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="alpha")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    args = parser.parse_args()
    set_seed(-1)
    if args.gpu < 0:
        args.device = 'cpu'
    else:
        args.device = 'cuda:%d'%args.gpu
    return args


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
            
        if self.best_acc<acc:
            self.best_acc = acc
            self.best_model = copy.deepcopy(model)
        else:
            self.not_imporved+=1
            if self.not_imporved>=30:
                return False
        return True

def set_seed(id):
    seeds = [104,202,303,405,505,610,770,870,909,42,7,9]
    # seeds = [101,202,303,404,505,606,770,870,909,42,7,9]
    if id <12:
        seed = seeds[id]
    else:
        seed = 5*id+3
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MGC(nn.Module):
    def __init__(self, args, steps = 2):
        super(MGC, self).__init__()
        self.n_hid = args.n_hid
        self.linears = nn.ModuleList()
        self.bns=nn.ModuleList()
        # import pdb;pdb.set_trace()
        for _ in range(steps):
            self.linears.append(nn.Linear(args.n_feat, int(args.n_hid), bias=True))
            self.bns.append(nn.BatchNorm1d(args.n_hid)  )
           
        self.linears.append(nn.Linear(int(args.n_hid), args.n_classes, bias=True))
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout1)
        self.dropout2 = nn.Dropout(args.dropout2)
        coffs = torch.from_numpy(np.array([1-args.alpha, args.alpha])) 
        self.coffs = nn.Parameter(coffs, requires_grad=False)

        self.norm = nn.LayerNorm(args.n_feat)  

        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.linears[-1].weight, gain=gain)
        nn.init.zeros_(self.linears[-1].bias)
            
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
            x = self.bns[i](x)
            x = self.activation(x)
            sumx+=self.coffs[i]*x 

        x1 = self.dropout2(sumx)
        
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
        coffs = torch.from_numpy(np.array([1-args.alpha, args.alpha])) 
        self.coffs = nn.Parameter(coffs, requires_grad=False)

        

        self.norm = nn.LayerNorm(args.n_feat)  

        gain = nn.init.calculate_gain("relu")
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


def evaluate(model, features, labels, loss_fcn):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        loss = loss_fcn(logits, labels)
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels),loss

def testmodel(model, features, labels, loss_fcn, device):
    model = model.to(device)
    model.eval()
    # ipdb.set_trace()
    with torch.no_grad():
        B = 1024
        N = labels.size(0)
        S = int(N/B)+1
        correct=0
        y_pred = []
        for i in range(S):
            featureB = features[:,B*i:B*(i+1),:].to(device)
            labelB = labels[B*i:B*(i+1)].to(device)
            # import pdb;
            # pdb.set_trace()
            logits = model(featureB)
            # loss = loss_fcn(logits, labelB)
            _, indicates = torch.max(logits, dim=1)
            y_pred.append(indicates)
        y_pred = torch.cat(y_pred)
        from ogb.lsc import MAG240MEvaluator
        evaluator = MAG240MEvaluator()
        input_dict = {'y_pred': y_pred}
        # result_dict = evaluator.eval(input_dict)
        dir_path = 'tmp'
        evaluator.save_test_submission(input_dict = input_dict, dir_path = dir_path, mode = 'test-dev')
        # result = np.load('tmp/y_pred_mag240m_test-dev.npz')

        # import pdb; pdb.set_trace()
        # return result_dict['acc']

            # correct += torch.sum(indices == labelB)
        # return correct.item() * 1.0 / len(labels),loss

def main(args):
        
    # g = set_dataset(args.dataset, args.split)
    rootpath = settings.DATAPATH+'/MAG240M/mag240m_kddcup2021/'
    train = np.load(rootpath+'x_train.npy')
    test = np.load(rootpath+'x_test.npy')
    vaild = np.load(rootpath+'x_valid.npy')

    train_idx = torch.from_numpy(np.array(range(len(train))))
    valid_idx = torch.from_numpy(np.array(range(len(train), len(train)+len(vaild))))
    test_idx = torch.from_numpy(np.array(range(len(train)+len(vaild), len(train)+len(vaild)+len(test))))
    emb = np.concatenate([train, vaild,test],axis=0)
    emb = torch.from_numpy(emb).float()

    train = np.load(rootpath+'mpx_train_8_pos.npy')-np.load(rootpath+'mpx_train_8_neg.npy')
    test = np.load(rootpath+'mpx_test_8_pos.npy')-np.load(rootpath+'mpx_test_8_neg.npy')
    vaild = np.load(rootpath+'mpx_valid_8_pos.npy')-np.load(rootpath+'mpx_valid_8_neg.npy')

    mgcfeature = np.concatenate([train, vaild,test],axis=0)
    mgcfeature = torch.from_numpy(mgcfeature)

    train = np.load(rootpath+'avgx_train_8_pos.npy')-np.load(rootpath+'avgx_train_8_neg.npy')
    test = np.load(rootpath+'avgx_test_8_pos.npy')-np.load(rootpath+'avgx_test_8_neg.npy')
    vaild = np.load(rootpath+'avgx_valid_8_pos.npy')-np.load(rootpath+'avgx_valid_8_neg.npy')

    ssgcfeature = np.concatenate([train, vaild,test],axis=0)
    ssgcfeature = torch.from_numpy(ssgcfeature)

    # features = torch.stack([mgcfeature,emb], dim=0).float()
    features = torch.stack([ssgcfeature,emb], dim=0).float()


    train = np.load(rootpath+'label_train.npy')
    test = np.load(rootpath+'label_test.npy')
    vaild = np.load(rootpath+'label_valid.npy')
    labels = np.concatenate([train, vaild,test],axis=0)
    labels = torch.from_numpy(labels).squeeze().long()





    args.n_feat = features[0].shape[1]
    args.n_classes = torch.max(labels).item()+1

    print(len(train_idx),len(valid_idx),len(test_idx),args.n_classes )
    





    
    
    
    # pos_dct = torch.load(rootpath+'sgc_drop0.00_100M_pos_dict.pt')
    # neg_dct = torch.load(rootpath+'sgc_drop0.00_100M_neg_dict.pt')
    # emb = pos_dct['embedding']-neg_dct['embedding']
    # mgcfeature = pos_dct['embedding_MGC_pos'] - neg_dct['embedding_MGC_neg']
    # ssgcfeature = pos_dct['embedding_SSGC_pos'] - neg_dct['embedding_SSGC_neg']
    # features = torch.stack([mgcfeature,emb], dim=0)
    # split_idx = pos_dct['split_idx']
    # train_idx = split_idx['train']
    # val_idx = split_idx['valid']
    # test_idx = split_idx['test']
    # labels = pos_dct['label'].squeeze()
    # args.n_feat = features[0].shape[1]
    # args.n_classes = torch.max(labels).item()+1


    loss_fcn = nn.CrossEntropyLoss()
    def training():
        model = MGC(args).to(args.device)
        # import pdb;pdb.set_trace()
        # model.set_adj(adj)
        # use optimizer
        optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)


        results = ResultRecorder(note="MGC")
        dur = []
        lr = args.lr
        # for epoch in tbar:
        for epoch in range(args.n_epochs):
            model.train()
            if epoch >= 3:
                t0 = time.time()

            
            # B= 1024 #train_idx.size(0)+1
            B = 512
            tmp_idx = torch.randperm(train_idx.size(0))
            curr_train_idx = train_idx[tmp_idx]
            N = int(curr_train_idx.size(0)/B)+1
            train_score = 0
            for i in range(N):
                nidxs = curr_train_idx[B*i:B*(i+1)]
                featuresB = features[:,nidxs,:].to(args.device)
                labelsB = labels[nidxs].to(args.device)
            
                # forward
                # ipdb.set_trace()
                logits = model.forward(featuresB)
                # ipdb.set_trace()
                loss = loss_fcn(logits, labelsB)

                # logits = logits[train_mask]
                _, indices = torch.max(logits, dim=1)
                correct = torch.sum(indices == labelsB)
                train_acc = correct.item() * 1.0 / len(labelsB),loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch >= 3:
                dur.append(time.time() - t0)
            # if (epoch+1)%100 == 0:
            if epoch==100:
                # model.coffs.requires_grad = True
                lr = lr/2
                optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=args.weight_decay)

            acc, val_loss = evaluate(model, features[:,valid_idx,:].to(args.device), labels[valid_idx].to(args.device), loss_fcn)

            print('epoch:{:d} loss: {:.4f}, val_loss: {:.4f}, val_score:{:.4f}'.format(epoch, loss.item(),val_loss.item(), acc))

            results.update(loss, train_acc,val_loss, acc, model)
        acc, test_loss = testmodel(results.best_model, features[:,test_idx,:].to(args.device), labels[test_idx].to(args.device), loss_fcn,args.device)
        results.test_loss = test_loss
        results.test_acc = acc
        print('Test_loss: %.4f | test_acc: %.4f' % (test_loss, acc))
        print('---------------------------------------------------')
        return results
    

    accs = []
    for id in range(20):  # id=4
        set_seed(id)
        print('==================%d================'%(id))
        result = training()
        acc = result.test_acc
        accs.append(acc)
    accs = sorted(accs)[-10:]
    return np.mean(accs), np.std(accs)
    

if __name__ == '__main__':
    args = init_config()
    print(args)
    # main(args)
    mu,std = main(args)
    print('mean acc: {:.4f}'.format(mu), 'std acc: {:.4f}'.format(std))



