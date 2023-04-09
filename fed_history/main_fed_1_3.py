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
from functools import reduce

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


    expe_w = dict()
    for k, v in w_glob.items():
        expe_w[k] = 0.1 * v * v
    
    expe_w_sqrt = dict()    
    for k, v in expe_w.items():
        expe_w_sqrt[k] = v.sqrt()


    L = np.zeros(args.num_users)
    delta_locals = np.zeros(args.num_users)

    expe_w_locals = [expe_w for i in range(args.num_users)]
    expe_w_sqrt_locals = [expe_w_sqrt for i in range(args.num_users)]

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    

    lr_train = 0.001

    alpha = np.arange(2, 65)
    epsilon = args.epsilon

    C = args.lcf
    beta = args.beta
    sigma_star = args.nl
    G = 1e-6

    T = 1
    K = 1
    gamma_star=args.gamma_star
    

    delta_repro = []
    delta_local_repro = []
    acc_test_repro=[]
    acc_train_repro=[]

    data_num_all = 0
    for idx in range(args.num_users):
        data_num_all += len(dict_users[idx])



    for iter in range(args.epochs):

        loss_locals = []

        delta_w_locals = []

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False) #一轮交互中选取的客户

        data_num = 0
        for idx in idxs_users:
            data_num += len(dict_users[idx])

        p=dict() # 权重向量     
        for idx in idxs_users:
            p[idx] = len(dict_users[idx])/data_num

        delta_max = 0

        theta_locals = dict()

        for idx in range(args.num_users): # 所有客户并行训练

            net_local = copy.deepcopy(net_glob)
            net_local.train()
            lot_size = args.ls
            batch_size = int(lot_size*len(dict_users[idx])/data_num) 

            dataset_local = DatasetSplit(dataset_train, dict_users[idx])
            ldr_train = DataLoader(dataset_local, batch_size=batch_size, shuffle=True)
            loss_func = nn.CrossEntropyLoss()
            w = {}
            params = dict(net_local.named_parameters())
            for name in params:
                w[name] = torch.zeros(params[name].shape).to(args.device)
            optimizer = torch.optim.RMSprop(net_local.parameters(), lr=args.lr)
            if args.local_ep > 1:
                aa = 0
                local_ep = int(args.local_ep)
                for iterr in range(local_ep):            
                    if iterr == local_ep-1:
                    # 最后一个epoch不要更新参数，只复制梯度出来
                   
                        w = dict()
                        for name, parms in net_local.named_parameters():                	
                            w[name] = copy.deepcopy(parms.grad)
                       
                        for k,v in w.items():
                            expe_w_locals[idx][k] = 0.1 * v * v + 0.9 * expe_w_locals[idx][k]
                        for k, v in expe_w_locals[idx].items():
                            expe_w_sqrt_locals[idx][k] = v.sqrt()
                        optimizer.zero_grad()
                        theta_local = net_local.state_dict() #保存当前参数方便更新

                    elif iterr < local_ep-1:
                        
                        for batch_idx, (images, labels) in enumerate(ldr_train):
                            aa += 1
                            images, labels = images.to(args.device), labels.to(args.device) # labels的长度为60
                            optimizer.zero_grad()
                            log_probs = net_local(images)
                            theta_local = net_local.state_dict()
                            R2=0
                            for k,v in theta_local.items():
                                R2 = R2+0.05*np.sum(np.square((theta_local[k]-theta_glob[k]).numpy()))/2
                        
                            loss = loss_func(log_probs, labels)+R2
                        
                            loss.backward() #计算梯度

                            w=dict()
                            for name,parms in net_local.named_parameters():
                                w[name] = parms.grad
                            for k,v in w.items():
                                expe_w_locals[idx][k] = 0.1 * v * v + 0.9 * expe_w_locals[idx][k]
                            if aa == int(len(dict_users[idx])/batch_size) and iterr == local_ep-2:
                                break
                            optimizer.step() #更新参数
                            
            else:
                aa = 0
                for batch_idx, (images, labels) in enumerate(ldr_train):
                    images, labels = images.to(args.device), labels.to(args.device) # labels的长度为60
                    optimizer.zero_grad()
                    log_probs = net_local(images)
                    
                    R2=0
                    for k,v in theta_local.items():
                        R2 = R2+0.05*np.sum(np.square((theta_local[k]-theta_glob[k]).numpy()))/2
                        
                    loss = loss_func(log_probs, labels)+R2
                        
                    loss.backward() #计算梯度

                    w=dict()
                    for name,parms in net_local.named_parameters():
                        w[name] = parms.grad
                    for k,v in w.items():
                        expe_w_locals[idx][k] = 0.1 * v * v + 0.9 * expe_w_locals[idx][k]
                    for k, v in expe_w_locals[idx].items():
                        expe_w_sqrt_locals[idx][k] = v.sqrt()
                    aa += 1            
                    if aa == int(args.local_ep*len(dict_users[idx])/batch_size):
                        theta_local = net_local.state_dict()
                        break     
                    optimizer.step() #更新参数


            
            delta_w=dict()
            for k,v in expe_w_locals[idx].items():
                delta_w[k] = w[k]/torch.sqrt(v+1e-8)

            

        

            var_expe_w = dict()
            for k, v in expe_w_sqrt_locals[idx].items():
                var_expe_w[k]=torch.var(v)

            s = dict()
            sigma = dict()
            key = list(w.keys())

            expe_w_star = expe_w_locals[idx]

            sigmaa = []
            ww = []
            for idx_key in key: # 对每一层进行运算

                if var_expe_w[idx_key] <= G or not(args.an):

                    
                    # 截断操作
                    delta_w_shape = delta_w[idx_key].shape
                    delta_w[idx_key].flatten()                
                    delta_w[idx_key] = delta_w[idx_key]/np.max((1, float(torch.norm(delta_w[idx_key], p=2)) / C))
                    delta_w[idx_key].view(delta_w_shape)  
                    delta_w[idx_key] = delta_w[idx_key] + torch.normal(0, C*sigma_star, delta_w[idx_key].size())


                else:
                    s[idx_key] = beta * torch.sqrt(expe_w_star[idx_key]/(gamma_star*expe_w_star[idx_key]+1e-8))
                    delta_w[idx_key] = torch.min(torch.max(delta_w[idx_key],-s[idx_key]),s[idx_key])
                    m = delta_w[idx_key].size()[0]
                    sigma[idx_key] = s[idx_key]*sigma_star*(m**0.5)
                    print('####')
                    # 提取sigma值
                    if iter==2 and idx==idxs_users[0]:
                        print('####')
                        sigmaa.append((sigma[idx_key].numpy()).flatten())
                    
                    delta_w[idx_key] = delta_w[idx_key] + torch.normal(0, sigma[idx_key])   

            if iter==2 and idx==idxs_users[0]:
                sigmaa = (np.array(sigmaa)).flatten()
                # sigmaa = sigmaa.numpy()
                np.savetxt("./savedata/sigma_round{}.txt".format(iter),sigmaa)
                sys.exit()
            
            
            for k,v in theta_local.items():	
                v = v - args.lr*delta_w[k] # 更新扰动过后的局部参数
                theta_local[k] = v

            if args.local_ep<=1:
                local_ep=2
            else:
                local_ep = args.local_ep

            if idx in idxs_users:
                theta_locals[idx] = theta_local
                L[idx] += 1
                x1 = 4*(math.exp(1/sigma_star**2)-1)
                x2 = 2*math.exp(1/sigma_star**2)
                x = min(x1, x2)

                log_delta = np.zeros(np.shape(alpha))
                delta_np = np.zeros(np.shape(alpha))
                q = lot_size/data_num
                for i in range(63):
                    calc_sum=0
                    if alpha[i]==2:
                        calc_sum=0
                    else:
                        for j in range(3, alpha[i]+1):
                            try:
                                try:
                                    xx = pow(q, j)*comb(alpha[i], j)*math.sqrt(abs(B(2*int(math.ceil(j/2)), sigma_star)*B(2*int(math.floor(j/2)), sigma_star)))
                                except OverflowError:
                                    xx = float('inf')
                                calc_sum = calc_sum+xx
                            except OverflowError:
                                calc_sum = float('inf')
                    log_delta[i] = -(epsilon-(L[idx]*(aa)/(alpha[i]-1))*math.log(1+comb(alpha[i],2)*q**2*x+4*calc_sum))*(alpha[i]-1)

                    try:
                        delta_np[i] = math.exp(log_delta[i])
                    except OverflowError:
                        delta_np[i] = float('inf')
                       
                delta_locals[idx] = np.min(delta_np)

        # if iter == 2:
        #     break


        T = T+aa
        # 计算隐私预算 
        x1 = 4*(math.exp(1/sigma_star**2)-1)
        x2 = 2*math.exp(1/sigma_star**2)
        x = min(x1, x2)

        log_delta = np.zeros(np.shape(alpha))
        delta_np = np.zeros(np.shape(alpha))
        q = lot_size/data_num_all
        print(q)
        for i in range(63):
            calc_sum=0
            if alpha[i]==2:
                calc_sum=0
            else:
                for j in range(3, alpha[i]+1):
                    calc_sum = calc_sum+pow(q, j)*comb(alpha[i], j)*math.sqrt(abs(B(2*int(math.ceil(j/2)), sigma_star)*B(2*int(math.floor(j/2)), sigma_star)))
            log_delta[i] = -(epsilon-(T/(alpha[i]-1))*math.log(1+comb(alpha[i],2)*q**2*x+4*calc_sum))*(alpha[i]-1)

            try:
                delta_np[i] = math.exp(log_delta[i])
            except OverflowError:
                delta_np[i] = float('inf')
                       
        delta = np.min(delta_np)
        delta_local = np.max(delta_locals)

        theta_glob = FedAvg(theta_locals, p)


        net_glob.load_state_dict(theta_glob)

        print(
            f"Train Round: {K} \t"
            f"(ε = {epsilon:.2f}, δ = {delta}, δ.Client = {delta_local})"
        )
        K = K+1

            
        if delta>args.delta or delta_local>args.delta:
            break 

        if (iter+1)%1 == 0:
            delta_repro.append(delta) 
            delta_local_repro.append(delta_local)
            net_glob.eval()
            final_acc_train, final_loss_train = test_img(net_glob, dataset_train, args)
            final_acc_test, final_loss_test = test_img(net_glob, dataset_test, args)
            
            acc_train_repro.append(final_acc_train)
            acc_test_repro.append(final_acc_test)

            print("Training accuracy: {:.2f}".format(final_acc_train))
            print("Testing accuracy: {:.2f}".format(final_acc_test))

 

    # plt.figure()
    # plt.plot(range(len(delta_repro)), delta_repro)
    # plt.ylabel('Delta value')
    # plt.savefig('./save/fed_{}_lr{}_K{}_round{}_an{}_lepoch{}_C{}_iid{}_ε{}_δ{}_clip{}_nl{}_ls{}_delta.png'.format(
    #     args.dataset, args.lr, args.num_users, args.epochs, args.an, args.local_ep, args.frac, args.iid, args.epsilon, args.delta, args.lcf, args.nl, args.ls))

    # plt.figure()
    # plt.plot(range(len(delta_repro)), acc_test_repro)
    # plt.plot(range(len(delta_repro)), acc_train_repro)
    # plt.legend([f'Testing accuracy', 'training accuracy'], loc='upper left')
    # plt.savefig('./save/fed_{}_lr{}_K{}_round{}_an{}_lepoch{}_C{}_iid{}_ε{}_δ{}_clip{}_nl{}_ls{}_accuracy.png'.format(
    #     args.dataset, args.lr, args.num_users, args.epochs, args.an, args.local_ep, args.frac, args.iid, args.epsilon, args.delta, args.lcf, args.nl, args.ls))
   
    
    x = np.array(range(len(delta_repro)))
    np.savetxt("./savedata/round_epsilon{}_frac{}_lr{}_nl{}_beta{}_gamma{}_ls{}_lepoch{}_an{}_iid{}.txt".format(args.epsilon, args.frac, args.lr, args.nl, args.beta, args.gamma_star, args.ls, args.local_ep, args.an, args.iid),x)
    y3 = np.array([i * 10000 for i in delta_repro])
    np.savetxt("./savedata/downlink_epsilon{}_frac{}_lr{}_nl{}_beta{}_gamma{}_ls{}_lepoch{}_an{}_iid{}.txt".format(args.epsilon, args.frac, args.lr, args.nl, args.beta, args.gamma_star, args.ls, args.local_ep, args.an, args.iid),y3)

    y4 = np.array([i * 10000 for i in delta_local_repro])
    np.savetxt("./savedata/upload_epsilon{}_frac{}_lr{}_nl{}_beta{}_gamma{}_ls{}_lepoch{}_an{}_iid{}.txt".format(args.epsilon, args.frac, args.lr, args.nl, args.beta, args.gamma_star, args.ls, args.local_ep, args.an, args.iid),y4)

    y1 = np.array(acc_test_repro)
    np.savetxt("./savedata/testing_epsilon{}_frac{}_lr{}_nl{}_beta{}_gamma{}_ls{}_lepoch{}_an{}_iid{}.txt".format(args.epsilon, args.frac, args.lr, args.nl, args.beta, args.gamma_star, args.ls, args.local_ep, args.an, args.iid),y1)
    
    y2 = np.array(acc_train_repro)
    np.savetxt("./savedata/training_epsilon{}_frac{}_lr{}_nl{}_beta{}_gamma{}_ls{}_lepoch{}_an{}_iid{}.txt".format(args.epsilon, args.frac, args.lr, args.nl, args.beta, args.gamma_star, args.ls, args.local_ep, args.an, args.iid),y2)

    # fig = plt.figure()
    # plt.grid(linestyle='-.')

    # ax1 = fig.add_subplot(1111)
    # fig,ax1 = plt.subplots()
    # plt.grid(linestyle='-.')
    # ax2 = ax1.twinx()    




    # ax1.plot(x, y1, 'r', marker='o', markerfacecolor='red', label = 'Testing Accuracy')
    # ax1.plot(x, y2, 'b', marker='o', markerfacecolor='blue', label = 'Training Accuracy') 
    # plt.xticks(np.arange(0,len(delta_repro),1))
    # plt.yticks(np.arange(60,100, 5)) 
    # ax1.set_ylabel('Accuracy')
    # ax1.set_xlabel('Round')
    # # ax1.set_title("Double Y axis")

    # ax2 = ax1.twinx()  # this is the important function
    # ax2.plot(x, y3, 'y', marker='^', markerfacecolor='yellow', label = 'Downlink δ')
    # ax2.plot(x, y4, 'c', marker='^', markerfacecolor='cyan', label = 'Upload δ')
    # plt.yticks(np.arange(0,1.0, 0.2)) 
    # # ax2.set_xlim([0, np.e])
    # ax2.set_ylabel('δ(1e-4)')
    # # ax2.set_xlabel('Round')

    # fig.legend(framealpha=0.5, loc='lower right', bbox_to_anchor=(1,0), bbox_transform=ax1.transAxes)

    # plt.savefig('./save/ceshi_epsilon{}.pdf'.format(args.epsilon))
    #     args.dataset, args.lr, args.num_users, args.epochs, args.an, args.local_ep, args.frac, args.iid, args.epsilon, args.delta, args.lcf, args.nl, args.ls))

    # plt.show()

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

