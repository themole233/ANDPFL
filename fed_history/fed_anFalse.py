#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import sys
import matplotlib
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
from copy import deepcopy
import models.Update

matplotlib.use('Agg')

# parse args
args = args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

# load dataset and split users
if args.dataset == 'mnist':
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # dataset_train = datasets.MNIST(root="./datasets/MNIST", train=True, transform=trans_mnist)
    # dataset_test = datasets.MNIST(root="./datasets/MNIST", train=False, transform=trans_mnist)
    dataset_train = datasets.MNIST(root="./datasets", train=True, transform=trans_mnist)
    dataset_test = datasets.MNIST(root="./datasets", train=False, transform=trans_mnist)

    # sample users
    print('iid:',args.iid)
    if args.iid:
        dataset_dict_clients = mnist_iid(dataset_train, args.num_users)
    else:
        dataset_dict_clients = mnist_noniid(dataset_train, args.num_users)
elif args.dataset == 'cifar':
    trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
    dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
    if args.iid:
        dataset_dict_clients = cifar_iid(dataset_train, args.num_users)
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
print(net_glob)  #net_glob是全局模型
net_glob.train()

# copy weights
w_global = net_glob.state_dict()
g_global = dict()

key = list(w_global.keys()) #提取每一层的名字

for name, parms in net_glob.named_parameters():
    g_global[name] = torch.zeros_like(parms.data).to(args.device)

Expectation_g_2 = dict()
# k is layer name
for k, v in g_global.items():
    Expectation_g_2[k] = 0.1 * v * v

Sqrt_Expectation_g_2 = dict()
for k, v in Expectation_g_2.items():
    Sqrt_Expectation_g_2[k] = v.sqrt()

# L is the number of times each worker uploads its model weights
L = torch.zeros(args.num_users)
delta_locals = torch.zeros(args.num_users) #计算每个客户的δ值

#将每个客户的值都存储下来
Expectation_g_2_locals = [Expectation_g_2.copy() for i in range(args.num_users)]
Sqrt_Expectation_g_2_locals = [Sqrt_Expectation_g_2.copy() for i in range(args.num_users)]

# training
loss_train = []
cv_loss, cv_acc = [], []
val_loss_pre, counter = 0, 0
net_best = None
best_loss = None
val_acc_list, net_list = [], []

alpha = torch.arange(2, 65)
epsilon = args.epsilon

C = args.lcf
beta = args.beta
sigma_0 = args.nl
G = 1e-6

T = 1
K = 1
gamma_prime = args.gamma_star

delta_0 = []
delta_local_i = []

acc_test_repro = []
acc_train_repro = []

number_of_all_data = 0
for idx in range(args.num_users):
    number_of_all_data += len(dataset_dict_clients[idx])

number_clients_epoch = max(int(args.frac * args.num_users), 1)

eta = args.lr

for k_iter in range(args.rounds):

    eta = eta*0.99

    loss_locals = []
    
    # print(k_iter)
    # print(number_clients_epoch)
    Z_k_idxs_users = np.random.choice(range(args.num_users), number_clients_epoch, replace=False)  # 一轮交互中选取的客户

    n_Z_k_number_data_epoch = 0 #一轮交互中选取的客户持有的样本总数
    for idx in Z_k_idxs_users:
        n_Z_k_number_data_epoch += len(dataset_dict_clients[idx])

    client_FL_weights = dict()  # 权重向量
    for idx in Z_k_idxs_users:
        client_FL_weights[idx] = len(dataset_dict_clients[idx]) / n_Z_k_number_data_epoch

    w_k_locals = dict() #记录下来需要进行上传的客户的参数

    for idx in range(args.num_users):  # 所有客户并行训练

        net_local = copy.deepcopy(net_glob)
        net_local.train()
        lot_size = args.ls
        batch_size = int(lot_size * len(dataset_dict_clients[idx]) / n_Z_k_number_data_epoch)

        dataset_local = DatasetSplit(dataset_train, dataset_dict_clients[idx])
        ldr_train = DataLoader(dataset_local, batch_size=batch_size, shuffle=True)
        loss_func = nn.CrossEntropyLoss()
        g = dict()
        # params = dict(net_local.named_parameters())
        # for name in params:
        #     w[name] = torch.zeros(params[name].shape).to(args.device)
        optimizer = torch.optim.SGD(net_local.parameters(), lr=args.lr)

        local_ep = int(args.local_ep)

        for iterr in range(local_ep):
            aa = 0
            for batch_idx, (images, labels) in enumerate(ldr_train):
                    
                images, labels = images.to(args.device), labels.to(args.device)  # labels的长度为60
                optimizer.zero_grad()
                log_probs = net_local(images)
                w_local = net_local.state_dict() #本地参数
                R2 = 0
                for k, v in w_local.items():
                       
                    R2 = R2 + 0.05 / 2 * torch.sum((w_local[k] - w_global[k])**2)

                loss = loss_func(log_probs, labels) + R2

                loss.backward()  # 计算梯度

                g = dict()
                for name, parms in net_local.named_parameters():
                    g[name] = parms.grad
                for k, v in g.items():
                    Expectation_g_2_locals[idx][k] = 0.1 * v * v + 0.9 * Expectation_g_2_locals[idx][k]
                for k, v in Expectation_g_2_locals[idx].items():
                    Sqrt_Expectation_g_2_locals[idx][k] = v.sqrt()

                # 最后一次不更新参数
                aa += 1
                if aa == int(len(dataset_dict_clients[idx]) / batch_size) and iterr == local_ep - 1:
                    # optimizer.zero_grad()
                    break 

                delta_w = dict()
                for k,v in Expectation_g_2_locals[idx].items():
                    delta_w[k] = g[k] / torch.sqrt(v + 1e-8)

                for k,v in w_local.items():
                    v = v - args.lr * g[k]
                    w_local[k] = v

                net_local.load_state_dict(w_local)

        

        var_expe_w = dict() #记录方差
        for k, v in Sqrt_Expectation_g_2_locals[idx].items():
            var_expe_w[k] = torch.var(v)

        s = dict()
        sigma = dict()

        Expectation_g_2_local = Expectation_g_2_locals[idx] #方便后面描述


        for idx_key in key:  # 对每一层进行运算

            

            # 截断操作
            g_shape = g[idx_key].shape
            g[idx_key].flatten()
            g[idx_key] = g[idx_key] / torch.max(torch.tensor(1.).float().to(args.device), torch.norm(g[idx_key], p=2) / C)
            g[idx_key].view(g_shape)
                
            g[idx_key] = g[idx_key] + torch.normal(0, C * sigma_0, g[idx_key].size()).to(args.device)
            
        for k, v in w_local.items():
            v = v - args.lr * g[k]  # 更新扰动过后的局部参数
            w_local[k] = v


        if idx in Z_k_idxs_users:
            w_k_locals[idx] = w_local
            L[idx] += 1

            delta = torch.zeros(alpha.shape)#每一个alpha值对应一个δ值

            for i in range(63):
                delta[i] = torch.exp(-(alpha[i]-1)*(epsilon-(k_iter+1)*alpha[i]/(2*sigma_0**2)))

            delta_locals[idx] = torch.min(delta)


    delta_c = torch.max(delta_locals)  # max value in {delta_i}

    w_global = FedAvg(w_k_locals, client_FL_weights)

    net_glob.load_state_dict(w_global)

    print(
        f"Train Round: {K} \t"
        f"(ε = {epsilon:.2f}, δ.Client = {delta_c})"
    )
    K = K + 1

    
    if delta_c > args.delta:
        break

    if (k_iter + 1) % 1 == 0:
        # delta_0.append(delta)
        delta_local_i.append(delta_c)
        net_glob.eval()
        final_acc_train, final_loss_train = test_img(net_glob, dataset_train, args)
        final_acc_test, final_loss_test = test_img(net_glob, dataset_test, args)

        acc_train_repro.append(final_acc_train)
        acc_test_repro.append(final_acc_test)

        print("Training accuracy: {:.2f}".format(final_acc_train))
        print("Testing accuracy: {:.2f}".format(final_acc_test))

x = np.arange(len(delta_local_i))
np.savetxt(
    "./savedata/round_anFalse_epsilon{}_frac{}_lr{}_nl{}_beta{}_gamma{}_ls{}_lepoch{:.2f}_an{}_iid{}.txt".format(args.epsilon,
                                                                                                         args.frac,
                                                                                                         args.lr,
                                                                                                         args.nl,
                                                                                                         args.beta,
                                                                                                         args.gamma_star,
                                                                                                         args.ls,
                                                                                                         args.local_ep,
                                                                                                         args.an,
                                                                                                         args.iid), x)
# y3 = delta_0 * 10000
# y3 = np.array([i * 10000 for i in delta_0])
# np.savetxt("./savedata/downlink_epsilon{}_frac{}_lr{}_nl{}_beta{}_gamma{}_ls{}_lepoch{:.2f}_an{}_iid{}.txt".format(
#     args.epsilon, args.frac, args.lr, args.nl, args.beta, args.gamma_star, args.ls, args.local_ep, args.an, args.iid),
#     np.array(y3))

# y4 = delta_local_i * 10000
y4 = np.array([i * 10000 for i in delta_local_i])
# np.array([i * 10000 for i in delta_local_repro])
np.savetxt(
    "./savedata/delta_anFalse_epsilon{}_frac{}_lr{}_nl{}_beta{}_gamma{}_ls{}_lepoch{:.2f}_an{}_iid{}.txt".format(args.epsilon,
                                                                                                          args.frac,
                                                                                                          args.lr,
                                                                                                          args.nl,
                                                                                                          args.beta,
                                                                                                          args.gamma_star,
                                                                                                          args.ls,
                                                                                                          args.local_ep,
                                                                                                          args.an,
                                                                                                          args.iid), np.array(y4))

y1 = np.array(acc_test_repro)
np.savetxt(
    "./savedata/training_anFalse_epsilon{}_frac{}_lr{}_nl{}_beta{}_gamma{}_ls{}_lepoch{:.2f}_an{}_iid{}.txt".format(args.epsilon,
                                                                                                           args.frac,
                                                                                                           args.lr,
                                                                                                           args.nl,
                                                                                                           args.beta,
                                                                                                           args.gamma_star,
                                                                                                           args.ls,
                                                                                                           args.local_ep,
                                                                                                           args.an,
                                                                                                           args.iid),
    y1)

y2 = np.array(acc_train_repro)
np.savetxt("./savedata/test_anFalse_epsilon{}_frac{}_lr{}_nl{}_beta{}_gamma{}_ls{}_lepoch{:.2f}_an{}_iid{}.txt".format(
    args.epsilon, args.frac, args.lr, args.nl, args.beta, args.gamma_star, args.ls, args.local_ep, args.an, args.iid),
    y2)

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
