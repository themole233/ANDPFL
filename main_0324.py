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
from models.Fed import FedAvg, FedAvgCollect
from models.test import test_img
from torch.utils.data import DataLoader, Dataset
import os

from utils.vgg_net import vgg11
from copy import deepcopy
import models.Update
import time

matplotlib.use('Agg')

fed_avg_collect = FedAvgCollect()

# parse args
args = args_parser()
args.device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
args.device_c = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

ceshi = [int(10),int(20),int(30),int(40),int(50),int(60),int(70),int(80),int(90)]

def clamp(x, y, z):
    floor = torch.min(x, z).to(args.device_c)
    ceil = torch.max(x, z).to(args.device_c)
    return torch.min(torch.max(floor, y), ceil).to(args.device_c)


torch.cuda.manual_seed(1)

# 记录不同优化器下的正确率变化
acc_test_repro = []
acc_train_repro = []

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
elif args.model == 'vgg' and args.dataset == 'cifar':
    net_glob = vgg11(pretrained=False, progress=False).to(args.device)
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
    g_global[name] = torch.zeros_like(parms.data)

Expectation_g_2 = dict()
# Sqrt_Expectation_g_2 = dict()
# Expectation_g_2 = torch.empty(1).to(args.device_c)
for name, parms in net_glob.named_parameters():
    Expectation_g_2[name] = torch.zeros_like(parms.data).to(args.device_c)
# for name, parms in net_glob.named_parameters():
# Sqrt_Expectation_g_2[name] = torch.zeros_like(parms.data).to(args.device_c)


# k is layer name
for k, v in g_global.items():
    Expectation_g_2[k] = (0.1 * v * v).to(args.device_c)

# for k, v in Expectation_g_2.items():
#     Sqrt_Expectation_g_2[k] = (v.sqrt()).to(args.device_c)

# L is the number of times each worker uploads its model weights
L = torch.zeros(args.num_users)
delta_locals = torch.zeros(args.num_users)  # 计算每个客户的δ值

# 将每个客户的值都存储下来
# Expectation_g_2_locals = [Expectation_g_2.copy() for i in range(args.num_users)]
# Expectation_g_2_locals = torch.tensor(Expectation_g_2_locals).to(args.device_c)
# Expectation_g_2_locals_0 = Expectation_g_2.copy()
# Sqrt_Expectation_g_2_locals = [Sqrt_Expectation_g_2.copy() for i in range(args.num_users)]
# Sqrt_Expectation_g_2_locals = torch.tensor(Sqrt_Expectation_g_2_locals).to(args.device_c)
# ratio = Expectation_g_2.copy()

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

# gamma_prime = args.beta
# gamma_l = 1.2 * np.asarray([np.power(0.999, i) for i in range(32)])
# gamma_k = 1.0 * np.asarray([np.power(1, i) for i in range(32)])
# gamma_k = torch.from_numpy(gamma_k).to(args.device)

delta_0 = []
delta_local_i = []

number_of_all_data = 0
for idx in range(args.num_users):
    number_of_all_data += len(dataset_dict_clients[idx])

number_clients_epoch = max(int(args.frac * args.num_users), 1)

temp_file_list = os.listdir("temp")
for temp_file in temp_file_list:
    os.remove(os.path.join("temp", temp_file))

for k_iter in range(1, args.rounds+1):
    loss_locals = []
    Z_k_idxs_users = np.random.choice(range(args.num_users), number_clients_epoch, replace=False)  # 一轮交互中选取的客户

    n_Z_k_number_data_epoch = 0  # 一轮交互中选取的客户持有的样本总数
    for idx in Z_k_idxs_users:
        n_Z_k_number_data_epoch += len(dataset_dict_clients[idx])

    client_FL_weights = dict()  # 权重向量
    for idx in Z_k_idxs_users:
        client_FL_weights[idx] = len(dataset_dict_clients[idx]) / n_Z_k_number_data_epoch

    # w_k_locals = dict()  # 记录下来需要进行上传的客户的参数
    w_local_0 = copy.deepcopy(net_glob.state_dict())

    # if k_iter == 2:
    #     print(w_local_0['conv1.weight'])

    for idx in Z_k_idxs_users:  # 被选中的客户进行训练
        # net_local = copy.deepcopy(net_glob).to(args.device_c)
        net_glob.load_state_dict(w_local_0)

        # if idx == 2 and k_iter == 1:
        #     print('全局模型参数')
        #     print(net_glob.state_dict()['conv1.weight'])
        #     print('客户')
        #     print(w_local_0['conv1.weight'])
        #     exit()

        # w_local_0 = net_local.state_dict()  # 提取当前的全局参数

        # if k_iter == int(2) and idx == 3:

        #     print('修改了吗')
        #     print(w_local_0['conv1.weight'])
        aa = 0

        for k in range(100):
            net_glob.train()
            lot_size = args.ls
            # batch_size = int(lot_size * len(dataset_dict_clients[idx]) / n_Z_k_number_data_epoch)
            # batch_size = int(lot_size * client_FL_weights[idx] * 10)  # 客户数据样本量差不多的话可以用
            batch_size = 100
            # sampling_ratio = lot_size / n_Z_k_number_data_epoch
            dataset_local = DatasetSplit(dataset_train, dataset_dict_clients[idx])
            ldr_train = DataLoader(dataset_local, batch_size=batch_size, shuffle=True)
            loss_func = nn.CrossEntropyLoss()
            g = dict()
            # if args.optimizer == "RMSPROP":
            #     optimizer = torch.optim.RMSprop(net_glob.parameters(), lr=args.lr)
            # elif args.optimizer == "Momentum":
            #     optimizer = torch.optim.SGD(net_glob.parameters(), lr=args.lr, momentum=0.9)

            local_ep = int(args.local_ep)

            # delta_w_store = torch.empty(1).to(args.device_c)

            for batch_idx, (images, labels) in enumerate(ldr_train):
                # if aa == 0:
                #     Expectation_g_2_locals_0 = copy.deepcopy(Expectation_g_2_locals[idx])
                aa += 1
                # if aa == 1:

                # f"Expectation_g_2_locals_user_{idx}.pth"
                # filename = os.path.join("temp", f"Expectation_g_2_user_{idx}.pth")
                # if os.path.exists(filename):
                #     Expectation_g_2 = torch.load(filename)
                # else:
                #     Expectation_g_2 = dict()
                #     for name, parms in net_glob.named_parameters():
                #         Expectation_g_2[name] = torch.zeros_like(parms.data).to(args.device_c)

                images, labels = images.to(args.device_c), labels.to(args.device_c)  # labels的长度为60
                if aa == 1:  # 第一次本地更新执行SGD算法
                    optimizer = torch.optim.SGD(net_glob.parameters(), lr=args.lr)
                    optimizer.zero_grad()
                    log_probs = net_glob(images)
                    w_local = net_glob.state_dict()  # 本地参数
                    # R2 = 0
                    # for k, v in w_local.items():
                    #     R2 = R2 + 0.05 / 2 * torch.sum((w_local[k] - w_global[k]) ** 2)
                    # loss = loss_func(log_probs, labels) + R2
                    loss = loss_func(log_probs, labels)
                    loss.backward()  # 计算梯度
                    g = dict()
                    for name, parms in net_glob.named_parameters():
                        g[name] = parms.grad.to(args.device_c)  # 提取梯度
                    Expectation_g_2 = dict()
                    # 其他优化器的迭代中间量
                    iter_param = dict()
                    iter_param_s = dict()
                    iter_param_delta = dict()
                    iter_m = dict()
                    iter_v = dict()
                    for k, v in w_local.items():
                        v = v - args.lr * g[k]
                        w_local[k] = v
                        if args.optimizer == "RMSPROP":
                            Expectation_g_2[k] = 0.1 * g[k] * g[k]
                        elif args.optimizer == "Momentum":
                            iter_param[k] = args.lr * g[k]
                        elif args.optimizer == "Adadelta":
                            Expectation_g_2[k] = 0.1 * g[k] * g[k]
                            iter_param_delta[k] = torch.sqrt(1e-6 / (Expectation_g_2[k] + 1e-6)) * g[k]
                            iter_param_s[k] = 0.1 * iter_param_delta[k] * iter_param_delta[k]
                        elif args.optimizer == "Adam":
                            iter_m[k] = 0.1*g[k]
                            iter_v[k] = 0.001*g[k]*g[k]
                    # 当本地训练次数为2时需要执行这一步
                    g_used = g.copy()
                    w_local_previous = w_local.copy()
                    if args.optimizer == "RMSPROP":
                        Expectation_g_2_used = Expectation_g_2.copy()
                    elif args.optimizer == "Momentum":
                        iter_param_used = iter_param.copy()
                    elif args.optimizer == "Adadelta":
                        Expectation_g_2_used = Expectation_g_2.copy()
                        iter_param_s_used = iter_param_s.copy()
                    elif args.optimizer == "Adam":
                        iter_m_used = iter_m.copy()
                        iter_v_used = iter_v.copy()
                    net_glob.load_state_dict(w_local)
                    if aa == int(args.local_ep):
                        break


                else:
                    if args.optimizer == "Momentum":
                        optimizer = torch.optim.SGD(net_glob.parameters(), lr=args.lr, momentum=0.9)
                        optimizer.zero_grad()
                        log_probs = net_glob(images)
                        w_local = net_glob.state_dict()  # 本地参数
                        loss = loss_func(log_probs, labels)
                        loss.backward()  # 计算梯度
                        g = dict()
                        for name, parms in net_glob.named_parameters():
                            g[name] = parms.grad.to(args.device_c)  # 提取梯度
                        for k, v in g.items():
                            iter_param[k] = 0.9 * iter_param[k] + args.lr * v
                        for k, v in w_local.items():
                            v = v - iter_param[k]
                            w_local[k] = v
                        if aa == int(args.local_ep - 1):
                            g_used = g.copy()
                            iter_param_used = iter_param.copy()
                            w_local_previous = w_local.copy()
                        net_glob.load_state_dict(w_local)

                        if aa == int(args.local_ep):
                            break

                    elif args.optimizer == "SGD":
                        optimizer = torch.optim.SGD(net_glob.parameters(), lr=args.lr)
                        optimizer.zero_grad()
                        log_probs = net_glob(images)
                        w_local = net_glob.state_dict()  # 本地参数
                        loss = loss_func(log_probs, labels)
                        loss.backward()  # 计算梯度
                        g = dict()
                        for name, parms in net_glob.named_parameters():
                            g[name] = parms.grad.to(args.device_c)  # 提取梯度
                        for k, v in w_local.items():
                            v = v - args.lr*g[k]
                            w_local[k] = v
                        if aa == int(args.local_ep - 1):
                            g_used = g.copy()
                            w_local_previous = w_local.copy()
                        net_glob.load_state_dict(w_local)
                        if aa == int(args.local_ep):
                            break

                    elif args.optimizer == "Adadelta":
                        optimizer = torch.optim.Adadelta(net_glob.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
                        optimizer.zero_grad()
                        log_probs = net_glob(images)
                        w_local = net_glob.state_dict()  # 本地参数
                        loss = loss_func(log_probs, labels)
                        loss.backward()  # 计算梯度
                        g = dict()
                        iter_param_delta = dict()
                        for name, parms in net_glob.named_parameters():
                            g[name] = parms.grad.to(args.device_c)  # 提取梯度
                        for k, v in g.items():
                            Expectation_g_2[k] = 0.1 * v * v + 0.9 * Expectation_g_2[k]
                            iter_param_delta[k] = torch.sqrt((iter_param_s[k] + 1e-6) / (Expectation_g_2[k] + 1e-6)) * v
                            iter_param_s[k] = 0.1 * iter_param_delta[k] * iter_param_delta[k] + 0.9 * iter_param_s[k]
                        for k, v in w_local.items():
                            v = v - iter_param_delta[k]
                            w_local[k] = v
                        if aa == int(args.local_ep - 1):
                            g_used = g.copy()
                            w_local_previous = w_local.copy()
                            Expectation_g_2_used = Expectation_g_2.copy()
                            iter_param_s_used = iter_param_s.copy()
                        net_glob.load_state_dict(w_local)
                        if aa == int(args.local_ep):
                            break


                    elif args.optimizer == "RMSPROP":
                        optimizer = torch.optim.RMSprop(net_glob.parameters(), lr=args.lr)
                        optimizer.zero_grad()
                        log_probs = net_glob(images)
                        w_local = net_glob.state_dict()  # 本地参数
                        loss = loss_func(log_probs, labels)
                        loss.backward()  # 计算梯度
                        g = dict()
                        for name, parms in net_glob.named_parameters():
                            g[name] = parms.grad.to(args.device_c)  # 提取梯度
                        for k, v in g.items():
                            Expectation_g_2[k] = 0.1 * v * v + 0.9 * Expectation_g_2[k]
                        delta_w = dict()
                        for k, v in Expectation_g_2.items():
                            delta_w[k] = g[k] / torch.sqrt(v + 1e-8)
                        for k, v in w_local.items():
                            v = v - args.lr * delta_w[k]
                            w_local[k] = v
                        if aa == int(args.local_ep - 1):
                            g_used = g.copy()
                            Expectation_g_2_used = Expectation_g_2.copy()
                            w_local_previous = w_local.copy()
                        net_glob.load_state_dict(w_local)
                        if aa == int(args.local_ep):
                            break
                        # net_glob.load_state_dict(w_local)

                    elif args.optimizer == "Adam":
                        optimizer = torch.optim.Adam(net_glob.parameters(), lr=args.lr)
                        optimizer.zero_grad()
                        log_probs = net_glob(images)
                        w_local = net_glob.state_dict()  # 本地参数
                        loss = loss_func(log_probs, labels)
                        loss.backward()  # 计算梯度
                        g = dict()
                        for name, parms in net_glob.named_parameters():
                            g[name] = parms.grad.to(args.device_c)  # 提取梯度
                        for k, v in g.items():
                            iter_m[k] = 0.9*iter_m[k]+0.1*g[k]
                            iter_v[k] = 0.999*iter_v[k]+0.001*g[k]*g[k]
                        hat_m = dict()
                        hat_v = dict()
                        for k, v in iter_m.items():
                            hat_m[k]=iter_m[k]/(1-0.9**aa)
                            hat_v[k]=iter_v[k]/(1-0.999**aa)
                        for k, v in w_local.items():
                            v = v - args.lr * hat_m[k]/(torch.sqrt(hat_v[k])+1e-8)
                            w_local[k] = v
                        if aa == int(args.local_ep - 1):
                            g_used = g.copy()
                            iter_m_used = iter_m.copy()
                            iter_v_used = iter_v.copy()
                            w_local_previous = w_local.copy()
                        net_glob.load_state_dict(w_local)
                        if aa == int(args.local_ep):
                            break

            if aa == int(args.local_ep):
                break
            # # 放前面的话这俩值一样
            # if k_iter == int(2) and idx == 3:
            #     print(w_local['conv1.weight'])
            #     print('##################')
            #     print(w_local_0['conv1.weight'])

            # if aa == int(args.local_ep):  # 本地训练32次
            #     # torch.save(Expectation_g_2, filename)
            #     break

            # net_glob.load_state_dict(w_local)

            # for k, v in Expectation_g_2_locals[idx].items():
            # Sqrt_Expectation_g_2_locals[idx][k] = v.sqrt()
            # var_expe_w = dict()  # 记录方差
            # for k, v in Expectation_g_2.items():
            #     var_expe_w[k] = torch.var(v.sqrt())

        s = dict()
        sigma = dict()
        delta_w_sum = dict()  # 要估计本地参数的敏感性就是估计这个参数的敏感性

        # 记录参数
        delta_w_sum_store = torch.empty(1).to(args.device)
        s_store = torch.empty(1).to(args.device)

        for k, v in w_local.items():
            # delta_w_sum[k] = (w_local_0[k] - v)/args.lr # 仅针对RMSPROP
            delta_w_sum[k] = w_local_0[k] - v
            # if k_iter in ceshi and idx == 3:
                # delta_w_sum_store = torch.cat((delta_w_sum_store, delta_w_sum[k].flatten()), 0)


        for idx_key in key:  # 对每一层进行运算

            #     if var_expe_w[idx_key] <= G or not args.an:
            #         for k, v in w_local.items():
            #             v = v.to(args.device_c) - args.lr * (delta_w[k].to(args.device_c))
            #             w_local[k] = v

            # # 截断操作 w_local_shape = w_local[idx_key].shape w_local[idx_key].flatten() w_local[idx_key] = w_local[
            # idx_key] / torch.max(torch.tensor(1.).float().to(args.device_c), torch.norm(w_local[idx_key],
            # p=2).to(args.device_c) / C) w_local[idx_key].view(w_local_shape) # w_local[idx_key] = w_local[idx_key]
            # + torch.normal(0, args.lr * C * sigma_0, #                                                    w_local[
            # idx_key].size()).to(args.device) w_local[idx_key] = w_local[idx_key] + torch.normal(0, C * sigma_0 *
            # args.lr, w_local[idx_key].size()).to(args.device_c) else: w_local_previous = w_local.copy()

            # for k, v in w_local.items():
            #     v = v.to(args.device_c) - args.lr * (delta_w[k].to(args.device_c))
            #     w_local[k] = v

            w_local_shape = w_local[idx_key].shape

            # if True:
            # sum_delta_w_store = torch.cat((sum_delta_w_store, (w_local_0[idx_key]-w_local[idx_key]).flatten()), 0)

            s[idx_key] = torch.zeros(w_local_shape).to(args.device_c)

            if args.optimizer == "RMSPROP":
                # s[idx_key] = args.beta * torch.abs(w_local_previous[idx_key] - args.lr * g_used[idx_key] /
                # torch.sqrt(0.9 * Expectation_g_2_used[idx_key] + 0.1 * g_used[idx_key] * g_used[idx_key] + 1e-8) -
                # w_local_0[idx_key]) / args.lr
                s[idx_key] = args.beta * torch.abs(w_local_previous[idx_key] - args.lr * g_used[idx_key] /
                                                   torch.sqrt(0.9 * Expectation_g_2_used[idx_key] +
                                                              0.1 * g_used[idx_key] * g_used[idx_key] + 1e-8) -
                                                   w_local_0[idx_key])

            elif args.optimizer == "SGD":
                s[idx_key] = args.beta * torch.abs(w_local_previous[idx_key] - args.lr * g_used[idx_key] - w_local_0[idx_key])

            elif args.optimizer == "Momentum":
                s[idx_key] = args.beta * torch.abs(w_local_previous[idx_key]
                                                   - (0.9 * iter_param_used[idx_key] + args.lr * g_used[idx_key])
                                                   - w_local_0[idx_key])
            elif args.optimizer == "Adadelta":
                s[idx_key] = args.beta * torch.abs(
                    w_local_previous[idx_key] - g_used[idx_key] * torch.sqrt(
                        (iter_param_s_used[idx_key] + 1e-6) / (
                                0.9 * Expectation_g_2_used[idx_key] + 0.1 *
                                g_used[idx_key] * g_used[idx_key] + 1e-8)) - w_local_0[idx_key])

            elif args.optimizer == "Adam":
                s[idx_key] = args.beta * torch.abs(w_local_previous[idx_key] - args.lr*(0.9*iter_m_used[idx_key]+0.1*g_used[idx_key])/((1-0.9*aa)*(1e-8+torch.sqrt((0.999*iter_v_used[idx_key]+0.001*g_used[idx_key]*g_used[idx_key])/(1-0.999**aa)))) - w_local_0[idx_key])
                # # 查看参数
                # if idx == 0 and k_iter == 2:  
                # if True:
                # s_store = torch.cat((s_store, s[idx_key].flatten()), 0)
            m = 1
            for product in w_local[idx_key].size():
                m = m * product
                # m = w_local[idx_key].size()[0]

            # print(m)

            sigma[idx_key] = s[idx_key] * sigma_0 * (m ** 0.5)
            # sigma[idx_key] = s[idx_key] * 4 * (m ** 0.5)

            # 记录参数
            # if k_iter in ceshi and idx == 3:
                # s_store = torch.cat((s_store, s[idx_key].flatten()), 0)

            # print(m)
            # print(sigma[idx_key])
                # print(s[idx_key])
            # print(args.lr * g_used[idx_key] / torch.sqrt(0.9 * Expectation_g_2_used[idx_key] + 0.1 * g_used[idx_key] * g_used[idx_key] + 1e-8))
            # print('#####################')
            # print(w_local_0[idx_key] - w_local[idx_key])
                # print(delta_w_sum[idx_key])
            # print('#####################')
            # print(s[idx_key])
            
            delta_w_sum[idx_key] = torch.min(torch.max(delta_w_sum[idx_key], -s[idx_key]), s[idx_key])

            # if k_iter == int(2) and idx == 3 and idx_key == "conv1.weight":
                # print(delta_w_sum[idx_key])
            
            # delta_w_sum[idx_key] = delta_w_sum[idx_key] + \
            #                        torch.normal(0, sigma[idx_key]/batch_size).to(args.device_c)

            delta_w_sum[idx_key] = delta_w_sum[idx_key] + \
                                   torch.normal(0, sigma[idx_key]/batch_size).to(args.device_c)

            # if k_iter == int(2) and idx == 3 and idx_key == "conv1.weight":
                # print(delta_w_sum[idx_key])

            # w_local[idx_key] = [clamp(x, y, z) for x, y, z in zip(w_local_0[idx_key] - s[idx_key], w_local[idx_key],
            #                                                           w_local_0[idx_key] + s[idx_key])]

            w_local[idx_key] = w_local_0[idx_key] - delta_w_sum[idx_key]
            # w_local[idx_key] = w_local_0[idx_key] - delta_w_sum[idx_key]

            # if k_iter == int(2) and idx == 3 and idx_key == "conv1.weight":
            #     print('############')
            # print(w_local[idx_key])

            # w_local[idx_key] = torch.stack(w_local[idx_key]).to(args.device_c)
            # w_local[idx_key].view(w_local_shape)
            # w_local[idx_key] = w_local[idx_key] + torch.normal(0, sigma[idx_key]).to(args.device_c)
            # if k_iter == int(2) and idx == 3 and idx_key == "conv1.weight":
            #     print('############')
            # print(w_local[idx_key])

        # if k_iter in ceshi and idx == 3:
            # exit()
        # 查看参数
        # if idx == int(0) and k_iter == int(2):
        # if True:
        #     np.savetxt('F:/实验室/pythoncode/savedata/s_fin_rounds{}_optim{}_eps{}.txt'.format(k_iter, args.optimizer, args.epsilon), s_store.numpy())
        # np.savetxt('F:/实验室/pythoncode/savedata/sigma_fin_rounds{}.txt'.format(k_iter), sigma_store.cpu().numpy())
        # np.savetxt('F:/实验室/pythoncode/savedata/w_fin_rounds{}.txt'.format(k_iter), w_local_store.cpu().numpy())
        # np.savetxt('F:/实验室/pythoncode/savedata/truncated_w_fin_rounds{}.txt'.format(k_iter), w_local_store.cpu().numpy())
        #     np.savetxt('F:/实验室/pythoncode/savedata/delta_w_sum_fin_rounds{}_optim{}_eps{}.txt'.format(k_iter, args.optimizer, args.epsilon), delta_w_sum_store.numpy())

        # np.savetxt('F:/实验室/pythoncode/savedata/sum_delta_w_fin_rounds{}.txt'.format(k_iter), sum_delta_w_store.cpu().numpy())

        # if idx == 1 and k_iter == 1:
        #     print('客户模型参数')
        #     print(net_glob.state_dict())

        if idx in Z_k_idxs_users:
            # w_k_locals[idx] = w_local
            fed_avg_collect.add(w_local, client_FL_weights[idx])
            L[idx] += 1

            delta = torch.zeros(alpha.shape)  # 每一个alpha值对应一个δ值

            for i in range(63):
                delta[i] = torch.exp(-(alpha[i] - 1) * (epsilon - (L[idx] + 1) * alpha[i] / (2 * sigma_0 ** 2)))

            delta_locals[idx] = torch.min(delta)

    delta_c = torch.max(delta_locals)  # max value in {delta_i}

    w_global = fed_avg_collect.get_result()

    fed_avg_collect.clear()

    net_glob.load_state_dict(w_global)

    print('rounds:', k_iter)

    net_glob.eval()
    final_acc_train, final_loss_train = test_img(net_glob, dataset_train, args)
    final_acc_test, final_loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(final_acc_train))
    print("Testing accuracy: {:.2f}".format(final_acc_test))

    acc_test_repro.append(final_acc_test)
    acc_train_repro.append(final_acc_train)

    if delta_c > args.delta:
        break

# np.savetxt('F:/实验室/pythoncode/federated-learning-master-0331/saved_acc_optim/test_eps{}_optim{}.txt'.format(
#     args.epsilon, args.optimizer), np.array(acc_test_repro))
# np.savetxt('F:/实验室/pythoncode/federated-learning-master-0331/saved_acc_optim/train_eps{}_optim{}.txt'.format(
#     args.epsilon, args.optimizer), np.array(acc_train_repro))

print("Training accuracy epsilon{}_frac{}_lr{}_nl{}_gamma{}_ls{}_an{}_users{}: {:.2f}".format(
    args.epsilon, args.frac, args.lr, args.nl, args.beta, args.ls, args.an, args.num_users,
    final_acc_train))
print("Testing accuracy epsilon{}_frac{}_lr{}_nl{}_gamma{}_ls{}_an{}_users{}: {:.2f}".format(
    args.epsilon, args.frac, args.lr, args.nl, args.beta, args.ls, args.an, args.num_users,
    final_acc_test))
print('Finish!')
