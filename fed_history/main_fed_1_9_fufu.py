##师兄帮我改的

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

def B(l, sigma_star):
    B = 0
    for i in range(l + 1):
        try:
            B = B + pow(-1, i) * comb(l, i) * math.exp(i * (i - 1) / (2 * sigma_star ** 2))
        except OverflowError:
            B = float('inf')

    return B


# if __name__ == '__main__':
# parse args
args = args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

print(args.an)

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
print(net_glob)
net_glob.train()

# copy weights
w_global = net_glob.state_dict()
g_global = dict()

for name, parms in net_glob.named_parameters():
    g_global[name] = torch.zeros_like(parms.data).to(args.device)
# print(global_g['conv1.weight'])

Expectation_g_2 = dict()
# k is layer name
for k, v in g_global.items():
    Expectation_g_2[k] = 0.1 * v * v
# print(Expectation_g_2['conv1.weight'])

Sqrt_Expectation_g_2 = dict()
for k, v in Expectation_g_2.items():
    Sqrt_Expectation_g_2[k] = v.sqrt()
# print(expe_w_sqrt['conv1.weight'])

# L is the number of times each worker uploads its model weights
L = torch.zeros(args.num_users)
delta_locals = torch.zeros(args.num_users)

Expectation_g_2_locals = [Expectation_g_2.copy() for i in range(args.num_users)]
Sqrt_Expectation_g_2_locals = [Sqrt_Expectation_g_2.copy() for i in range(args.num_users)]

# training
loss_train = []
cv_loss, cv_acc = [], []
val_loss_pre, counter = 0, 0
net_best = None
best_loss = None
val_acc_list, net_list = [], []

lr_train = 0.001

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

for k_iter in range(args.epochs):

    loss_locals = []
    delta_w_locals = []
    
    # print(k_iter)
    # print(number_clients_epoch)
    Z_k_idxs_users = np.random.choice(range(args.num_users), number_clients_epoch, replace=False)  # 一轮交互中选取的客户

    n_Z_k_number_data_epoch = 0
    for idx in Z_k_idxs_users:
        n_Z_k_number_data_epoch += len(dataset_dict_clients[idx])

    client_FL_weights = dict()  # 权重向量
    for idx in Z_k_idxs_users:
        client_FL_weights[idx] = len(dataset_dict_clients[idx]) / n_Z_k_number_data_epoch

    w_k_locals = dict()

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
        optimizer = torch.optim.RMSprop(net_local.parameters(), lr=args.lr)
        if args.local_ep >= 1:
            local_ep = int(args.local_ep)
            for iterr in range(local_ep):
                aa = 0
                for batch_idx, (images, labels) in enumerate(ldr_train):
                    
                    images, labels = images.to(args.device), labels.to(args.device)  # labels的长度为60
                    optimizer.zero_grad()
                    log_probs = net_local(images)
                    theta_local = net_local.state_dict()
                    R2 = 0
                    for k, v in theta_local.items():
                        # R2 = R2 + 0.05 / 2 * torch.sum(torch.square(theta_local[k] - w_global[k]))
                        R2 = R2 + 0.05 / 2 * torch.sum((theta_local[k] - w_global[k])**2)

                    loss = loss_func(log_probs, labels) + R2

                    loss.backward()  # 计算梯度

                    g = dict()
                    for name, parms in net_local.named_parameters():
                        g[name] = parms.grad
                    for k, v in g.items():
                        Expectation_g_2_locals[idx][k] = 0.1 * v * v + 0.9 * Expectation_g_2_locals[idx][k]
                    for k, v in Expectation_g_2_locals[idx].items():
                        Sqrt_Expectation_g_2_locals[idx][k] = v.sqrt()
                    aa += 1
                    if aa == int(len(dataset_dict_clients[idx]) / batch_size) and iterr == local_ep - 1:
                        # optimizer.zero_grad()
                        break
                    # optimizer.step()  # 更新参数
                    delta_w = dict()
                    for k,v in Expectation_g_2_locals[idx].items():
                        delta_w[k] = g[k] / torch.sqrt(v + 1e-8)

                    for k,v in theta_local.items():
                        v = v - args.lr * delta_w[k]
                        theta_local[k] = v

                    net_local.load_state_dict(theta_local)

        else:
            aa = 0
            for batch_idx, (images, labels) in enumerate(ldr_train):
                images, labels = images.to(args.device), labels.to(args.device)  # labels的长度为60
                optimizer.zero_grad()
                log_probs = net_local(images)
                theta_local = net_local.state_dict()
                R2 = 0
                for k, v in theta_local.items():
                    # R2 = R2 + 0.05 / 2 * torch.sum(torch.square((theta_local[k] - w_global[k])))
                    R2 = R2 + 0.05 / 2 * torch.sum((theta_local[k] - w_global[k])**2)

                loss = loss_func(log_probs, labels) + R2

                loss.backward()  # 计算梯度

                # TODO
                g = dict()
                for name, parms in net_local.named_parameters():
                    g[name] = parms.grad
                for k, v in g.items():
                    Expectation_g_2_locals[idx][k] = 0.9 * Expectation_g_2_locals[idx][k] + 0.1 * v * v
                for k, v in Expectation_g_2_locals[idx].items():
                    Sqrt_Expectation_g_2_locals[idx][k] = v.sqrt()

                aa += 1
                if aa == int(args.local_ep * len(dataset_dict_clients[idx]) / batch_size):
                    # theta_local = net_local.state_dict()
                    # optimizer.zero_grad()
                    break
                # optimizer.step()  # 更新参数
                delta_w = dict()
                for k,v in Expectation_g_2_locals[idx].items():
                    delta_w[k] = g[k] / torch.sqrt(v + 1e-8)

                for k,v in theta_local.items():
                    v = v - args.lr * delta_w[k]
                    theta_local[k] = v

                net_local.load_state_dict(theta_local)

        delta_w = dict()
        for k, v in Expectation_g_2_locals[idx].items():
            delta_w[k] = g[k] / torch.sqrt(v + 1e-8)

        key = list(g.keys())

        # 把参数存下来
        delta_ww = []
        if k_iter == 0 and idx == Z_k_idxs_users[0] and args.save == True:
            for idx_keyy in key:
                delta_ww.append((delta_w[idx_keyy].numpy()).flatten())
            delta_ww = np.concatenate(delta_ww)
            np.savetxt("./savedata/delta_w_round{}.txt".format(k_iter), delta_ww)

        var_expe_w = dict()
        for k, v in Expectation_g_2_locals[idx].items():
            Sqrt_Expectation_g_2_locals[idx][k] = v.sqrt()
        for k, v in Sqrt_Expectation_g_2_locals[idx].items():
            var_expe_w[k] = torch.var(v)

        s = dict()
        sigma = dict()

        expe_w_star = Expectation_g_2_locals[idx]

        sigmaa = []
        ss = []
        # mm = n_Z_k_number_data_epoch/(len(dataset_dict_clients[idx])*(number_clients_epoch**0.5))  # 噪声的缩放因子
        mm = 1
        # print(number_clients_epoch)
        for idx_key in key:  # 对每一层进行运算

            if var_expe_w[idx_key] <= G or not (args.an):

                # 截断操作
                delta_w_shape = delta_w[idx_key].shape
                delta_w[idx_key].flatten()
                delta_w[idx_key] = delta_w[idx_key] / torch.max(torch.tensor(1.).float().to(args.device), torch.norm(delta_w[idx_key], p=2) / C)
                delta_w[idx_key].view(delta_w_shape)
                
                delta_w[idx_key] = delta_w[idx_key] + torch.normal(0, C *mm * sigma_0, delta_w[idx_key].size()).to(args.device)


            else:
                s[idx_key] = beta * torch.sqrt(expe_w_star[idx_key] / (gamma_prime * expe_w_star[idx_key] + 1e-8))
                delta_w[idx_key] = torch.min(torch.max(delta_w[idx_key], -s[idx_key]), s[idx_key])
                m = delta_w[idx_key].size()[0]
                sigma[idx_key] = s[idx_key] * sigma_0 * (m ** 0.5) * mm

                # print('####')
                # 提取sigma值
                if k_iter == 0 and idx == Z_k_idxs_users[0] and args.save == True:
                    # print('####')
                    ss.append((s[idx_key].numpy()).flatten())
                    sigmaa.append((sigma[idx_key].numpy()).flatten())
                    # print(len(sigmaa[-1]))
                    # print(type(sigmaa[-1])) #numpy.ndarray
                    # print(sigmaa[-1].shape) #(N,)维数组

                    # sigmaa[-1] = (sigma[idx_key].numpy()).flatten()

                delta_w[idx_key] = delta_w[idx_key] + torch.normal(0, sigma[idx_key]).to(args.device)

        if k_iter == 0 and idx == Z_k_idxs_users[0] and args.save == True:
            # sigmaa = sigmaa.flatten()
            # sigmaa = np.array(sigmaa)
            # sigmaa = sigmaa.flatten()
            ss = np.concatenate(ss)
            sigmaa = np.concatenate(sigmaa)
            np.savetxt("./savedata/prior_delta_w_round{}.txt".format(k_iter), ss / beta)
            np.savetxt("./savedata/sigma_round{}.txt".format(k_iter), sigmaa)
            # sys.exit()

        for k, v in theta_local.items():
            v = v - args.lr * delta_w[k]  # 更新扰动过后的局部参数
            theta_local[k] = v

        if args.local_ep <= 1:
            local_ep = 1
        else:
            local_ep = args.local_ep

        if idx in Z_k_idxs_users:
            w_k_locals[idx] = theta_local
            L[idx] += 1
            x1 = 4 * (math.exp(1 / sigma_0 ** 2) - 1)
            x2 = 2 * math.exp(1 / sigma_0 ** 2)
            x = min(x1, x2)

            log_delta = torch.zeros(alpha.shape)
            delta_np = torch.zeros(alpha.shape)
            q = lot_size / n_Z_k_number_data_epoch
            print(q)
            for i in range(63):
                calc_sum = 0
                if alpha[i] == 2:
                    calc_sum = 0
                else:
                    for j in range(3, alpha[i] + 1):
                        try:
                            try:
                                xx = pow(q, j) * comb(alpha[i], j) * math.sqrt(
                                    abs(B(2 * int(math.ceil(j / 2)), sigma_0) * B(2 * int(math.floor(j / 2)),
                                                                                  sigma_0)))
                            except OverflowError:
                                xx = float('inf')
                            calc_sum = calc_sum + xx
                        except OverflowError:
                            calc_sum = float('inf')
                log_delta[i] = -(epsilon - (L[idx] * (aa * local_ep) / (alpha[i] - 1)) * math.log(
                    1 + comb(alpha[i], 2) * q ** 2 * x + 4 * calc_sum)) * (alpha[i] - 1)

                try:
                    delta_np[i] = torch.exp(log_delta[i])
                except OverflowError:
                    delta_np[i] = float('inf')

            delta_locals[idx] = torch.min(delta_np)

    T = T + aa * local_ep
    # 计算隐私预算
    x1 = 4 * (math.exp(1 / sigma_0 ** 2) - 1)
    x2 = 2 * math.exp(1 / sigma_0 ** 2)
    x = min(x1, x2)

    log_delta = torch.zeros(alpha.shape)
    delta_np = torch.zeros(alpha.shape)
    q = lot_size / number_of_all_data
    print(q)
    for i in range(63):
        calc_sum = 0
        if alpha[i] == 2:
            calc_sum = 0
        else:
            for j in range(3, alpha[i] + 1):
                calc_sum = calc_sum + pow(q, j) * comb(alpha[i], j) * math.sqrt(
                    abs(B(2 * int(math.ceil(j / 2)), sigma_0) * B(2 * int(math.floor(j / 2)), sigma_0)))
        log_delta[i] = -(
                epsilon - (T / (alpha[i] - 1)) * math.log(1 + comb(alpha[i], 2) * q ** 2 * x + 4 * calc_sum)) * (
                               alpha[i] - 1)

        try:
            delta_np[i] = torch.exp(log_delta[i])
        except OverflowError:
            delta_np[i] = float('inf')

    delta = torch.min(delta_np)
    delta_c = torch.max(delta_locals)  # max value in {delta_i}

    w_global = FedAvg(w_k_locals, client_FL_weights)

    net_glob.load_state_dict(w_global)

    print(
        f"Train Round: {K} \t"
        f"(ε = {epsilon:.2f}, δ = {delta}, δ.Client = {delta_c})"
    )
    K = K + 1

    
    if delta > args.delta or delta_c > args.delta:
        break

    if (k_iter + 1) % 1 == 0:
        delta_0.append(delta)
        delta_local_i.append(delta_c)
        net_glob.eval()
        final_acc_train, final_loss_train = test_img(net_glob, dataset_train, args)
        final_acc_test, final_loss_test = test_img(net_glob, dataset_test, args)

        acc_train_repro.append(final_acc_train)
        acc_test_repro.append(final_acc_test)

        print("Training accuracy: {:.2f}".format(final_acc_train))
        print("Testing accuracy: {:.2f}".format(final_acc_test))

x = np.arange(len(delta_0))
np.savetxt(
    "./savedata/round_epsilon{}_frac{}_lr{}_nl{}_beta{}_gamma{}_ls{}_lepoch{:.2f}_an{}_iid{}.txt".format(args.epsilon,
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
y3 = np.array([i * 10000 for i in delta_0])
np.savetxt("./savedata/downlink_epsilon{}_frac{}_lr{}_nl{}_beta{}_gamma{}_ls{}_lepoch{:.2f}_an{}_iid{}.txt".format(
    args.epsilon, args.frac, args.lr, args.nl, args.beta, args.gamma_star, args.ls, args.local_ep, args.an, args.iid),
    np.array(y3))

# y4 = delta_local_i * 10000
y4 = np.array([i * 10000 for i in delta_local_i])
# np.array([i * 10000 for i in delta_local_repro])
np.savetxt(
    "./savedata/upload_epsilon{}_frac{}_lr{}_nl{}_beta{}_gamma{}_ls{}_lepoch{:.2f}_an{}_iid{}.txt".format(args.epsilon,
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
    "./savedata/testing_epsilon{}_frac{}_lr{}_nl{}_beta{}_gamma{}_ls{}_lepoch{:.2f}_an{}_iid{}.txt".format(args.epsilon,
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
np.savetxt("./savedata/training_epsilon{}_frac{}_lr{}_nl{}_beta{}_gamma{}_ls{}_lepoch{:.2f}_an{}_iid{}.txt".format(
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
