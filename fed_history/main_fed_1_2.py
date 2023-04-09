#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import math  
from torch import nn, autograd
from scipy.special import comb, perm
import math

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate, DatasetSplit
from models.Nets import MLP, CNNMnist, CNNCifar, SampleConvNet
from models.Fed import FedAvg
from models.test import test_img
from torch.utils.data import DataLoader, Dataset


import models.Update

def B(l, sigma_star):
    B = 0
    for i in range(l+1):        
        try:
            B = B+pow(-1, i)*comb(l, i)*math.exp(i*(i-1)/(2*sigma_star**2))
        except OverflowError:
            B = float('inf')

    return B


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    
    print(args.an)
    
    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST(root="./datasets", train=True, transform=trans_mnist)
        dataset_test = datasets.MNIST(root="./datasets", train=False, transform=trans_mnist)
 
       # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'dppca' and args.dataset == 'mnist':
        net_glob = SampleConvNet().to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    theta_glob = net_glob.state_dict()
    w_glob = dict()

    for name, parms in net_glob.named_parameters():	
        w_glob[name] = torch.zeros_like(parms.data).to(args.device)

    pow_w = dict() # 计算梯度的平方
    for k, v in w_glob.items():
        pow_w[k] = v.pow(2)
      
    expe_w_star = dict()
    for k, v in pow_w.items():
        expe_w_star[k] = 0.1 * v
    
    expe_w_star_sqrt = dict()    
    for k, v in expe_w_star.items():
        expe_w_star_sqrt[k] = v.sqrt()

# os.exit() 

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    


    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)] # 初始化
        theta_locals = [theta_glob for i in range(args.num_users)]

    lr_train = 0.001

    alpha = np.arange(2, 65)
    epsilon = args.epsilon

    C = args.lcf
    beta = 1.2
    sigma_star = args.nl
    G = 1e-6

    T = 1

    delta_repro = []
    acc_test_repro=[]
    acc_train_repro=[]

    # net_

    for iter in range(args.epochs):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
            theta_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False) 
        data_num = 0
        for idx in idxs_users:
            data_num += len(dict_users[idx])
        p=[] # 权重向量        
        delta_max = 0
        for idx in range(args.num_users): # 一次迭代中选取的客户数目
            net_local = copy.deepcopy(net_glob)
            net_local.train()
            lot_size = args.ls
            batch_size = int(lot_size*len(dict_users[idx])/data_num)            
            q = batch_size/len(dict_users[idx])
            p.append(len(dict_users[idx])/data_num)
            dataset_local = DatasetSplit(dataset_train, dict_users[idx])
            ldr_train = DataLoader(dataset_local, batch_size=batch_size, shuffle=True)
            loss_func = nn.CrossEntropyLoss()
            w = {}
            params = dict(net_local.named_parameters())
            for name in params:
                w[name] = torch.zeros(params[name].shape).to(args.device)
            optimizer = torch.optim.RMSprop(net_local.parameters(), lr=args.lr)
            for iter in range(args.local_ep):
                optimizer.zero_grad()
                for batch_idx, (images, labels) in enumerate(ldr_train):
                    images, labels = images.to(args.device), labels.to(args.device) # labels的长度为60
                    log_probs = net_local(images)
                    loss = loss_func(log_probs, labels)
                    net_local.zero_grad()
                    loss.backward()                
                if iter == args.local_ep-1:
                    w = dict()
                    for name, parms in net_local.named_parameters():                	
                        w[name] = parms.grad
                else:
                    w = dict()
                    for name, parms in net_local.named_parameters():                	
                        w[name] = parms.grad
                    optimizer.step()

                
                pow_w = dict() # 计算梯度的平方
                for k, v in w.items():
                    pow_w[k] = v.pow(2)
      
                # expe_w_star = dict()
                for k, v in pow_w.items():
                    expe_w_star[k] = 0.1 * v + 0.9 * expe_w_star[k]
    
                expe_w_star_sqrt = dict()    
                for k, v in expe_w_star.items():
                    expe_w_star_sqrt[k] = v.sqrt()
                # 最后一次迭代不要更新参数，只复制梯度出来
                
                optimizer.zero_grad()


            var_expe_w = []
            for k, v in w.items():
                var_expe_w.append(torch.var(expe_w_star_sqrt[k]))

            w_clip = copy.deepcopy(w)
            s = copy.deepcopy(w)
            sigma = copy.deepcopy(w)
            key = list(w.keys())
            sub_client_num = len(idxs_users) # 一次迭代中选取的客户数

            ## 梯度扰动
            for i in range(len(var_expe_w)): # 对每一层进行运算

            
                idx_key = key[i]

                if var_expe_w[i] <= G or not(args.an):

                    
                    # 截断操作
                    w_shape = w[idx_key].shape
                    w[idx_key].flatten()                
                    w[idx_key] = w[idx_key]/np.max((1, float(torch.norm(w[idx_key], p=2)) / C))
                    w[idx_key].view(w_shape)  
                    w[idx_key] = w[idx_key] + torch.normal(0, C*sigma_star/(p[-1]*math.sqrt(sub_client_num)), w[idx_key].size())


                else:

                    s[idx_key] = beta * torch.sqrt(expe_w_star[idx_key])
                    w_clip[idx_key] = torch.min(torch.max(w[idx_key],-s[idx_key]),s[idx_key])
                    q = w_clip[idx_key].size()[-1]

                    sigma[idx_key] = beta*sigma_star*torch.sqrt(q*expe_w_star[idx_key])
                    w[idx_key] = w_clip[idx_key] + torch.normal(0, sigma[idx_key]/(p[-1]*math.sqrt(sub_client_num)))                


            theta = dict()
            for name, parms in net_local.named_parameters():	
                parms.grad = w[name]


            optimizer.step()
            optimizer.zero_grad()

            theta = net_local.state_dict() # 得到参数

            if idx in idxs_users:

                theta_locals.append(theta)   





        
        # 计算隐私预算 
        x1 = 4*(math.exp(1/sigma_star**2)-1)
        x2 = 2*math.exp(1/sigma_star**2)
        x = min(x1, x2)

        log_delta = np.zeros(np.shape(alpha))
        delta_np = np.zeros(np.shape(alpha))
        q = 0.01
        for i in range(63):
            calc_sum=0
            if alpha[i]==2:
                calc_sum=0
            else:
                for j in range(3, alpha[i]+1):
                    calc_sum = calc_sum+pow(q, j)*comb(alpha[i], j)*math.sqrt(B(2*int(math.ceil(j/2)), sigma_star)*B(2*int(math.floor(j/2)), sigma_star))
            log_delta[i] = -(epsilon-(T/(alpha[i]-1))*math.log(1+comb(alpha[i],2)*q**2*x+4*calc_sum))*(alpha[i]-1)

            try:
                delta_np[i] = math.exp(log_delta[i])
            except OverflowError:
                delta_np[i] = float('inf')
                       
        delta = np.min(delta_np)

        theta_glob = FedAvg(theta_locals, p)

        # copy weight to net_glob
        net_glob.load_state_dict(theta_glob)

        print(
            f"Train Round: {T} \t"
            # f"Loss: {np.mean(losses):.6f} "
            f"(ε = {epsilon:.2f}, δ = {delta})"
        )
        T = T+1

        for k, v in w_glob.items():
            
            pow_w[k] = v.pow(2)
        
        gamma_star = 0.9
        for k, v in expe_w_star.items():
            expe_w_star[k] = gamma_star * v + (1-gamma_star) * pow_w[k]

            
        if delta>args.delta:
            break 

        if (iter+1)%1 == 0:
            delta_repro.append(delta) 

            net_glob.eval()
            final_acc_train, final_loss_train = test_img(net_glob, dataset_train, args)
            final_acc_test, final_loss_test = test_img(net_glob, dataset_test, args)
            
            acc_train_repro.append(final_acc_train)
            acc_test_repro.append(final_acc_test)

            print("Training accuracy: {:.2f}".format(final_acc_train))
            print("Testing accuracy: {:.2f}".format(final_acc_test))

 

    plt.figure()
    plt.plot(range(len(delta_repro)), delta_repro)
    plt.ylabel('Delta value')
    plt.savefig('./save/fed_{}_lr{}_K{}_round{}_an{}_lepoch{}_C{}_iid{}_ε{}_δ{}_clip{}_nl{}_ls{}_delta.png'.format(
        args.dataset, args.lr, args.num_users, args.epochs, args.an, args.local_ep, args.frac, args.iid, args.epsilon, args.delta, args.lcf, args.nl, args.ls))

    plt.figure()
    plt.plot(range(len(delta_repro)), acc_test_repro)
    plt.plot(range(len(delta_repro)), acc_train_repro)
    plt.legend([f'Testing accuracy', 'training accuracy'], loc='upper left')
    plt.savefig('./save/fed_{}_lr{}_K{}_round{}_an{}_lepoch{}_C{}_iid{}_ε{}_δ{}_clip{}_nl{}_ls{}_accuracy.png'.format(
        args.dataset, args.lr, args.num_users, args.epochs, args.an, args.local_ep, args.frac, args.iid, args.epsilon, args.delta, args.lcf, args.nl, args.ls))
   
    
    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

