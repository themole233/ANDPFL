# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import sys
import matplotlib
import matplotlib.pyplot as plt
import copy
import random
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
import time

matplotlib.use('Agg')

# parse args
args = args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

torch.cuda.manual_seed(1)

# load dataset and split users
if args.dataset == 'mnist':
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # dataset_train = datasets.MNIST(root="./datasets/MNIST", train=True, transform=trans_mnist)
    # dataset_test = datasets.MNIST(root="./datasets/MNIST", train=False, transform=trans_mnist)
    dataset_train = datasets.MNIST(root="./datasets", train=True, transform=trans_mnist)
    dataset_test = datasets.MNIST(root="./datasets", train=False, transform=trans_mnist)

    # sample users
    print('iid:', args.iid)
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
# print(net_glob)  #net_glob是全局模型
net_glob.train()

# copy weights
w_global = net_glob.state_dict()
g_global = dict()

key = list(w_global.keys())  # 提取每一层的名字

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
delta_locals = torch.zeros(args.num_users)  # 计算每个客户的δ值

# 将每个客户的值都存储下来
Expectation_g_2_locals = [Expectation_g_2.copy() for i in range(args.num_users)]
Expectation_g_2_locals_0 = Expectation_g_2.copy()
Sqrt_Expectation_g_2_locals_0 = [Sqrt_Expectation_g_2.copy() for i in range(args.num_users)]

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
gamma_l = 1.2 * np.asarray([np.power(0.999, i) for i in range(32)])
gamma_l = torch.from_numpy(gamma_l).to(args.device)

delta_0 = []
delta_local_i = []

acc_test_repro = []
acc_train_repro = []

number_of_all_data = 0
for idx in range(args.num_users):
    number_of_all_data += len(dataset_dict_clients[idx])

number_clients_epoch = max(int(args.frac * args.num_users), 1)

for k_iter in range(args.rounds):

    # time_start=time.time()

    loss_locals = []

    # print(k_iter)
    # print(number_clients_epoch)
    Z_k_idxs_users = np.random.choice(range(args.num_users), number_clients_epoch, replace=False)  # 一轮交互中选取的客户

    n_Z_k_number_data_epoch = 0  # 一轮交互中选取的客户持有的样本总数
    for idx in Z_k_idxs_users:
        n_Z_k_number_data_epoch += len(dataset_dict_clients[idx])

    client_FL_weights = dict()  # 权重向量
    for idx in Z_k_idxs_users:
        client_FL_weights[idx] = len(dataset_dict_clients[idx]) / n_Z_k_number_data_epoch

    w_k_locals = dict()  # 记录下来需要进行上传的客户的参数

    for idx in range(args.num_users):  # 所有客户并行训练

        net_local = copy.deepcopy(net_glob)

        w_local_0 = net_glob.state_dict()  # 提取当前的全局参数

        net_local.train()
        lot_size = args.ls
        batch_size = int(lot_size * len(dataset_dict_clients[idx]) / n_Z_k_number_data_epoch)

        dataset_local = DatasetSplit(dataset_train, dataset_dict_clients[idx])
        ldr_train = DataLoader(dataset_local, batch_size=batch_size, shuffle=True)
        loss_func = nn.CrossEntropyLoss()
        g = dict()
        optimizer = torch.optim.RMSprop(net_local.parameters(), lr=args.lr)

        local_ep = int(args.local_ep)

        aa = 0

        for batch_idx, (images, labels) in enumerate(ldr_train):
            images, labels = images.to(args.device), labels.to(args.device)  # labels的长度为60
            optimizer.zero_grad()
            log_probs = net_local(images)
            w_local = net_local.state_dict()  # 本地参数
            R2 = 0
            for k, v in w_local.items():
                R2 = R2 + 0.05 / 2 * torch.sum((w_local[k] - w_global[k]) ** 2)

            loss = loss_func(log_probs, labels) + R2

            loss.backward()  # 计算梯度

            g = dict()
            for name, parms in net_local.named_parameters():
                g[name] = parms.grad
            for k, v in g.items():
                Expectation_g_2_locals[idx][k] = 0.1 * v * v + 0.9 * Expectation_g_2_locals[idx][k]
            for k, v in Expectation_g_2_locals_0.items():
                Sqrt_Expectation_g_2_locals_0[idx][k] = v.sqrt()

            aa += 1

            delta_w = dict()
            for k, v in Expectation_g_2_locals[idx].items():
                delta_w[k] = g[k] / torch.sqrt(v + 1e-8)

            for k, v in w_local.items():
                v = v - args.lr * delta_w[k]
                w_local[k] = v

            if aa == int(32):  # 本地训练32次
                Expectation_g_2_locals_0 = Expectation_g_2_locals[idx].copy()
                break
            net_local.load_state_dict(w_local)

        var_expe_w = dict()  # 记录方差
        for k, v in Sqrt_Expectation_g_2_locals_0[idx].items():
            var_expe_w[k] = torch.var(v)

        s = dict()
        sigma = dict()

        # sigma_store = torch.empty(1).to(args.device)

        for idx_key in key:  # 对每一层进行运算

            if var_expe_w[idx_key] <= G or not (args.an):
                # 截断操作
                # print('here')
                w_local_shape = w_local[idx_key].shape
                w_local[idx_key].flatten()
                w_local[idx_key] = w_local[idx_key] / torch.max(torch.tensor(1.).float().to(args.device),
                                                                torch.norm(w_local[idx_key], p=2) / (0.001*C))
                w_local[idx_key].view(w_local_shape)
                #
                w_local[idx_key] = w_local[idx_key] + torch.normal(0, 0.001 * C * sigma_0, w_local[idx_key].size()).to(
                    args.device)
            else:
                # print('here')
                w_local_shape = w_local[idx_key].shape

                s[idx_key] = torch.zeros(w_local_shape).to(args.device)
                s[idx_key] = args.lr * torch.sqrt(gamma_prime * Expectation_g_2_locals_0[idx_key] /
                                                  ((gamma_prime * 0.1 + 0.9) * Expectation_g_2_locals_0[
                                                      idx_key] + 1e-8))

                # # 查看参数
                # if idx == 0 and k_iter == 2:  # sigma在2到3之间
                #     s_store = torch.cat((s_store, s[idx_key].flatten()), 0)

                w_local[idx_key] = torch.min(torch.max(w_local[idx_key], w_local[idx_key] - s[idx_key]),
                                             w_local[idx_key] + s[idx_key])
                m = w_local[idx_key].size()[0]
                sigma[idx_key] = s[idx_key] * sigma_0 * (m ** 0.5)

                # # 查看参数
                # if idx == 0 and k_iter == 2: # sigma在2到3之间
                #     sigma_store = torch.cat((sigma_store, sigma[idx_key].flatten()), 0)

                w_local[idx_key] = w_local[idx_key] + torch.normal(0, sigma[idx_key]).to(args.device)

        # # 查看参数
        # if idx == 0 and k_iter == 2:
        #     np.savetxt('./savedata/sigma_ceshi_rounds{}.txt'.format(k_iter), sigma_store.cpu().numpy())

        if idx in Z_k_idxs_users:
            w_k_locals[idx] = w_local
            L[idx] += 1

            delta = torch.zeros(alpha.shape)  # 每一个alpha值对应一个δ值

            for i in range(63):
                delta[i] = torch.exp(-(alpha[i] - 1) * (epsilon - (L[idx] + 1) * alpha[i] / (2 * sigma_0 ** 2)))

            delta_locals[idx] = torch.min(delta)

    delta_c = torch.max(delta_locals)  # max value in {delta_i}

    w_global = FedAvg(w_k_locals, client_FL_weights)

    net_glob.load_state_dict(w_global)

    print('rounds:', K)

    K += 1

    if delta_c > args.delta:
        break

    if (k_iter + 1) % 1 == 0:
        net_glob.eval()
        final_acc_train, final_loss_train = test_img(net_glob, dataset_train, args)
        final_acc_test, final_loss_test = test_img(net_glob, dataset_test, args)
        print("Training accuracy: {:.2f}".format(final_acc_train))
        print("Testing accuracy: {:.2f}".format(final_acc_test))

    # time_end=time.time()
    # print('time cost', time_end-time_start,'s')

# testing
net_glob.eval()
acc_train, loss_train = test_img(net_glob, dataset_train, args)
acc_test, loss_test = test_img(net_glob, dataset_test, args)
print("Training accuracy epsilon{}_frac{}_lr{}_nl{}_beta{}_gamma{}_ls{}_lepoch{:.2f}_an{}_iid{}: {:.2f}".format(
    args.epsilon, args.frac, args.lr, args.nl, args.beta, args.gamma_star, args.ls, args.local_ep, args.an, args.iid,
    acc_train))
print("Testing accuracy epsilon{}_frac{}_lr{}_nl{}_beta{}_gamma{}_ls{}_lepoch{:.2f}_an{}_iid{}: {:.2f}".format(
    args.epsilon, args.frac, args.lr, args.nl, args.beta, args.gamma_star, args.ls, args.local_ep, args.an, args.iid,
    acc_test))
print('Finish!')
