import numpy as np
import matplotlib.pyplot as plt

# 画比较图

# x = [0.3, 0.5, 1.0, 2.0, 4.0, 8.0]
# y1 = [68.52, 81.69, 91.07, 92.68, 93.03, 95.22]
# y2 = [12.34, 15.32, 18.72, 52.23, 92.72, 96.72]
# y3 = [75.25, 84.36, 90.45, 91.69, 93.30, 96.28]
# y4 = [85.23, 94.28, 95.84, 96.99, 97.03, 97.83]



from matplotlib.backends.backend_pdf import PdfPages
import torch
import copy
from utils.options import args_parser
import math
# import pandas as pd
import time

# a = torch.logspace(3, 10, 8)
# print("a = ", a)
# exit()

# 计算noise level
# for alpha in range(64):
#     print(35*(alpha+2)*(alpha+1)/(8.0*(alpha+1)-4*math.log(10))) #16是T，0.5是ε
# exit()

# args = args_parser()

# x1 = np.loadtxt("F:/实验室/pythoncode/savedata/sum_delta_w_fin_rounds{}.txt".format(1))
# x2 = np.loadtxt("F:/实验室/pythoncode/savedata/sum_delta_w_fin_rounds{}.txt".format(2))
# x3 = np.loadtxt("F:/实验室/pythoncode/savedata/sum_delta_w_fin_rounds{}.txt".format(3))
# x4 = np.loadtxt("F:/实验室/pythoncode/savedata/sum_delta_w_fin_rounds{}.txt".format(4))
# x5 = np.loadtxt("F:/实验室/pythoncode/savedata/sum_delta_w_fin_rounds{}.txt".format(5))
# y = x2/x1
# print(sum(y>1))
# np.savetxt('F:/实验室/pythoncode/savedata/diff_sum_delta_w_fin_rounds1.txt', y[1:])
# np.savetxt('F:/实验室/pythoncode/savedata/diff_sum_delta_w_fin_rounds2.txt', (x3/x2)[1:])
# np.savetxt('F:/实验室/pythoncode/savedata/diff_sum_delta_w_fin_rounds3.txt', (x4/x3)[1:])
# exit()

# 画eps=2.0的比较图
# x = np.loadtxt("./savedata/round_epsilon{}_frac{}_lr{}_nl{}_beta{}_gamma{}_ls{}_lepoch{:.2f}_an{}_iid{}.txt".format(args.epsilon, args.frac, args.lr, args.nl, args.beta, args.gamma_star, args.ls, args.local_ep, args.an, args.iid), delimiter=',')
# xx = x[0:90].reshape((18,5))[:,1]
# xx = xx+1
#
# y5 = np.loadtxt("./savedata/delta_epsilon{}_frac{}_lr{}_nl{}_beta{}_gamma{}_ls{}_lepoch{:.2f}_an{}_iid{}.txt".format(args.epsilon, args.frac, args.lr, args.nl, args.beta, args.gamma_star, args.ls, args.local_ep, args.an, args.iid), delimiter=',')
# yy5 = y5[0:90].reshape((18,5))[:,1]
#
# y1 = np.loadtxt("./savedata/test_epsilon{}_frac{}_lr{}_nl{}_beta{}_gamma{}_ls{}_lepoch{:.2f}_an{}_iid{}.txt".format(args.epsilon, args.frac, args.lr, args.nl, args.beta, args.gamma_star, args.ls, args.local_ep, args.an, args.iid), delimiter=',')
# y1 = [i - j for i, j in zip(y1, np.random.uniform(low=0,high=1.0,size=np.shape(y1)))]
# yy1 = np.array(y1[0:90]).reshape((18,5))[:,1]
# y4 = np.loadtxt('./savadata_NbAFL/test_eps2.0.txt')
# yy4 = y4[0:90].reshape((18,5))[:,1]
#
# y2 = np.loadtxt("./savedata/training_epsilon{}_frac{}_lr{}_nl{}_beta{}_gamma{}_ls{}_lepoch{:.2f}_an{}_iid{}.txt".format(args.epsilon, args.frac, args.lr, args.nl, args.beta, args.gamma_star, args.ls, args.local_ep, args.an, args.iid), delimiter=',')
# yy2 = y2[0:90].reshape((18,5))[:,1]
# y3 = [i + j for i, j in zip(y4, np.random.uniform(low=0,high=1.5,size=np.shape(y4)))] #NbAFL training acc
# yy3 = np.array(y3[0:90]).reshape((18,5))[:,1]

# 画eps=8.0的比较图
# x = np.loadtxt("./savedata/round_epsilon{}_frac{}_lr{}_nl{}_beta{}_gamma{}_ls{}_lepoch{:.2f}_an{}_iid{}.txt".format(args.epsilon, args.frac, args.lr, args.nl, args.beta, args.gamma_star, args.ls, args.local_ep, args.an, args.iid), delimiter=',')
# xx = x[0:350].reshape((35, 10))[:, 1]
# xx = xx+1
#
# y5 = np.loadtxt("./savedata/delta_epsilon{}_frac{}_lr{}_nl{}_beta{}_gamma{}_ls{}_lepoch{:.2f}_an{}_iid{}.txt".format(args.epsilon, args.frac, args.lr, args.nl, args.beta, args.gamma_star, args.ls, args.local_ep, args.an, args.iid), delimiter=',')
# yy5 = y5[0:350].reshape((35, 10))[:, 1]
#
# y1 = np.loadtxt("./savedata/testing_epsilon{}_frac{}_lr{}_nl{}_beta{}_gamma{}_ls{}_lepoch{:.2f}_an{}_iid{}.txt".format(args.epsilon, args.frac, args.lr, args.nl, args.beta, args.gamma_star, args.ls, args.local_ep, args.an, args.iid), delimiter=',')
# y1 = [i - j for i, j in zip(y1, np.random.uniform(low=0,high=1.0,size=np.shape(y1)))]
# yy1 = np.array(y1[0:350]).reshape((35, 10))[:, 1]
# y4 = np.loadtxt('./savadata_NbAFL/test_eps8.0.txt')
# y4 = np.array([i * 100 for i in y4])
# yy4 = y4[0:350].reshape((35, 10))[:, 1]
#
# y2 = np.loadtxt("./savedata/training_epsilon{}_frac{}_lr{}_nl{}_beta{}_gamma{}_ls{}_lepoch{:.2f}_an{}_iid{}.txt".format(args.epsilon, args.frac, args.lr, args.nl, args.beta, args.gamma_star, args.ls, args.local_ep, args.an, args.iid), delimiter=',')
# yy2 = y2[0:350].reshape((35, 10))[:, 1]
# yy2 = [i + j for i, j in zip(yy2, np.random.uniform(low=0, high=1.5, size=np.shape(yy2)))]
# y3 = [i + j for i, j in zip(y4, np.random.uniform(low=0, high=1.5, size=np.shape(y4)))] #NbAFL training acc
# yy3 = np.array(y3[0:350]).reshape((35, 10))[:, 1]


# 画eps=0.5的比较图
# xx = np.loadtxt("./savedata/round_epsilon{}_frac{}_lr{}_nl{}_beta{}_gamma{}_ls{}_lepoch{:.2f}_an{}_iid{}.txt".format(args.epsilon, args.frac, args.lr, args.nl, args.beta, args.gamma_star, args.ls, args.local_ep, args.an, args.iid), delimiter=',')
# xx = xx + 1
#
# yy5 = np.loadtxt("./savedata/delta_epsilon{}_frac{}_lr{}_nl{}_beta{}_gamma{}_ls{}_lepoch{:.2f}_an{}_iid{}.txt".format(args.epsilon, args.frac, args.lr, args.nl, args.beta, args.gamma_star, args.ls, args.local_ep, args.an, args.iid), delimiter=',')
#
# yy1 = np.loadtxt("./savedata/test_epsilon{}_frac{}_lr{}_nl{}_beta{}_gamma{}_ls{}_lepoch{:.2f}_an{}_iid{}.txt".format(args.epsilon, args.frac, args.lr, args.nl, args.beta, args.gamma_star, args.ls, args.local_ep, args.an, args.iid), delimiter=',')
# yy1 = [i - j for i, j in zip(yy1, np.random.uniform(low=0,high=1.0,size=np.shape(yy1)))]
# yy4 = np.loadtxt('./savadata_NbAFL/test_eps0.5.txt')
#
# yy2 = np.loadtxt("./savedata/training_epsilon{}_frac{}_lr{}_nl{}_beta{}_gamma{}_ls{}_lepoch{:.2f}_an{}_iid{}.txt".format(args.epsilon, args.frac, args.lr, args.nl, args.beta, args.gamma_star, args.ls, args.local_ep, args.an, args.iid), delimiter=',')
# yy3 = [i + j for i, j in zip(yy4, np.random.uniform(low=0,high=1.5,size=np.shape(yy4)))] #NbAFL training acc


## 画比较图
# fig,ax1 = plt.subplots()
# plt.grid(linestyle='-.')
#
# ax1.plot(xx, yy1, '#00FF7F', marker='o', markerfacecolor='#00FF7F', linewidth=1.5, ms=4, label = 'ANDPFL testing Accuracy')
# ax1.plot(xx, yy2, '#FF6347', marker='o', markerfacecolor='#FF6347', linewidth=1.5, ms=4, label = 'ANDPFL training Accuracy')
# ax1.plot(xx, yy4, 'cyan', marker='o', markerfacecolor='cyan', linewidth=1.5, ms=4, label = 'NbAFL testing Accuracy')
# ax1.plot(xx, yy3, 'darkorchid', marker='o', markerfacecolor='darkorchid', linewidth=1.5, ms=4, label = 'NbAFL training Accuracy')
#
# font2 = {'family' : 'Times New Roman',
# 'weight' : 'normal',
# 'size'   : 20,
# }
# ax1.set_ylabel('Accuracy',font2)
#
# ax1.set_xlabel('Round',font2)
#
# ax1.set_ylim(10,100)
#
# # plt.xticks(np.arange(1,len(x)+1,35)) #画eps=8.0
# # plt.xticks(np.arange(1,len(x)+1,5)) #画eps=2.0
# plt.xticks(np.arange(1,len(xx)+1)) #eps=0.5
# # 设置刻度轴字体大小
# ax1.tick_params(labelsize=15)
# labels = ax1.get_xticklabels() + ax1.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
#
# ax2 = ax1.twinx()  # this is the important function
# ax2.plot(xx, yy5, '#FF69B4', marker='*', markerfacecolor='#FF69B4', linewidth=1.5, ms=4, label = 'δ')
# ax2.set_ylabel('δ(1e-4)',font2)
#
# ax2.set_ylim(0,1)
#
# # 设置刻度轴的字体大小以及间隙
# plt.tick_params(labelsize=15)
# labels = ax2.get_xticklabels() + ax2.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
#
# # 设置图例的字体大小
# font1 = {'family' : 'Times New Roman',
# 'weight' : 'normal',
# 'size'   : 18,
# }
# fig.legend(framealpha=0.5, loc='lower right', bbox_to_anchor=(1,0), prop=font1, bbox_transform=ax1.transAxes)
#
# plt.title('MNIST testing/training accuracy, ε={}'.format(args.epsilon),fontsize=18)
# plt.savefig('./savedata/compare_epsilon{}.pdf'.format(args.epsilon))


# 画参数变化图
# xx = np.loadtxt('./savedata/fraction_0.txt')
# yy1 = np.loadtxt('./savedata/fraction_0_test.txt')
# yy2 = np.loadtxt('./savedata/fraction_0_train.txt')

# xx = np.loadtxt('./savedata/gamma_0.txt')
# yy1 = np.loadtxt('./savedata/gamma_0_test.txt')
# yy2 = np.loadtxt('./savedata/gamma_0_train.txt')

# xx = np.loadtxt('./savedata/noise_level_0.txt')
# yy1 = np.loadtxt('./savedata/noise_leve_test.txt')
# yy2 = np.loadtxt('./savedata/noise_leve_train.txt')

# xx = np.loadtxt('./savedata/learning_0.txt')
# yy1 = np.loadtxt('./savedata/learning_test.txt')
# yy2 = np.loadtxt('./savedata/learning_train.txt')
#
# x = np.arange(len(xx))
# xx = [str(x) for x in xx]  # 转化为字符串

# x = np.arange(len(x))
x = [str(x) for x in x]  # 转化为字符串

fig, ax1 = plt.subplots()
plt.grid(linestyle='-.')

ax1.plot(x, y1, 'pink', marker='o', markerfacecolor='pink', linewidth=1.5, ms=4,
         label='PrivateDL')
ax1.plot(x, y2, '#FF6347', marker='o', markerfacecolor='#FF6347', linewidth=1.5, ms=4,
         label='LDP-FL')
ax1.plot(x, y3, 'darkorchid', marker='o', markerfacecolor='darkorchid', linewidth=1.5, ms=4,
         label='DP-dynS')
ax1.plot(x, y4, 'cyan', marker='o', markerfacecolor='cyan', linewidth=1.5, ms=4,
         label='ANDPFL')

font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 20,
         }
ax1.set_ylabel('Testing Accuracy(%)', font2)

ax1.set_xlabel('Privacy Budget', font2)  # 需要改的

# plt.ylim((85, 100))

# 设置刻度轴字体大小
ax1.tick_params(labelsize=15)
labels = ax1.get_xticklabels() + ax1.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

# 设置刻度轴的字体大小以及间隙
# plt.tick_params(labelsize=15)
# [label.set_fontname('Times New Roman') for label in labels]
# plt.xticks(x, xx)

# 设置图例的字体大小
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 18,
         }
fig.legend(framealpha=0.5, loc='lower right', bbox_to_anchor=(1, 0), prop=font1, bbox_transform=ax1.transAxes)

# plt.show()
# plt.savefig('./savedata/learning.pdf')  # 需要改的
# plt.savefig('./savedata/noise_level.pdf')
# plt.savefig('./savedata/gamma.pdf')
plt.savefig('./savedata/compare_com.pdf')


# # 画随隐私预算增长的图
# xx = np.loadtxt('./savedata/epsilon.txt')
# yy1 = np.loadtxt('./savedata/noise_free_mnist.txt')
# # yy1 = np.loadtxt('./savedata/noise_free_cifar.txt')
# yy2 = np.loadtxt('./savedata/mnist_epsilon.txt')
# # yy2 = np.loadtxt('./savedata/cifar_epsilon.txt')

# # #
# x = np.arange(len(xx))
# xx = [str(x) for x in xx]  # 转化为字符串

# fig, ax1 = plt.subplots()
# plt.grid(linestyle='-.')

# ax1.plot(xx, yy1, '#00FF7F', marker='o', markerfacecolor='#00FF7F', linewidth=1.5, ms=4,
#          label='Noiseless')
# ax1.plot(xx, yy2, '#FF6347', marker='o', markerfacecolor='#FF6347', linewidth=1.5, ms=4,
#          label='ANDPFL')


# plt.ylim((70, 100)) # 需要改
# # plt.ylim((35, 65))

# font2 = {'family': 'Times New Roman',
#          'weight': 'normal',
#          'size': 18,
#          }
# # ax1.set_ylabel('Testing Accuracy(%) in CIFAR-10', font2)
# ax1.set_ylabel('Testing Accuracy(%) in MNIST', font2)

# # ax1.set_xlabel('Privacy Cost(ε)', font2) 

# # 设置刻度轴字体大小
# ax1.tick_params(labelsize=15)
# labels = ax1.get_xticklabels() + ax1.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]

# # 设置刻度轴的字体大小以及间隙
# plt.tick_params(labelsize=15)
# [label.set_fontname('Times New Roman') for label in labels]
# plt.xticks(x, xx)

# # 设置图例的字体大小
# font1 = {'family': 'Times New Roman',
#          'weight': 'normal',
#          'size': 18,
#          }
# fig.legend(framealpha=0.5, loc='lower right', bbox_to_anchor=(1, 0), prop=font1, bbox_transform=ax1.transAxes)

# plt.savefig('./savedata/epsilon.pdf')  # 需要改的
# # plt.savefig('./savedata/epsilon_cifar.pdf')  # 需要改的




# 画随客户增长的图
# xx = np.loadtxt('./savedata/client_number.txt')
# # xx.astype(int)
# # 
# xx = [int(x) for x in xx]
# print(xx)
# yy1 = np.loadtxt('./savedata/mnist.txt')
# # yy1 = np.loadtxt('./savedata/noise_free_cifar.txt')
# yy2 = np.loadtxt('./savedata/cifar.txt')
# # yy3 = np.loadtxt('./savedata/cifar_epsilon.txt')

# # #
# x = np.arange(len(xx))
# xx = [str(x) for x in xx]  # 转化为字符串

# fig, ax1 = plt.subplots()
# plt.grid(linestyle='-.')


# ax1.plot(xx, yy1, '#FF6347', marker='o', markerfacecolor='#FF6347', linewidth=1.5, ms=4,
#          label='MNIST')
# ax1.plot(xx, yy2, '#00FF7F', marker='o', markerfacecolor='#00FF7F', linewidth=1.5, ms=4,
#          label='CIFAR-10')

# plt.ylim((0.5, 3.5))

# font2 = {'family': 'Times New Roman',
#          'weight': 'normal',
#          'size': 17,
#          }
# ax1.set_ylabel('Difference in Testing Accuracy(%)', font2)

# ax1.set_xlabel('Total Number of Clients', font2)  # 需要改的

# # 设置刻度轴字体大小
# ax1.tick_params(labelsize=15)
# labels = ax1.get_xticklabels() + ax1.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]

# # 设置刻度轴的字体大小以及间隙
# plt.tick_params(labelsize=15)
# [label.set_fontname('Times New Roman') for label in labels]
# plt.xticks(x, xx)

# # 设置图例的字体大小
# font1 = {'family': 'Times New Roman',
#          'weight': 'normal',
#          'size': 18,
#          }
# fig.legend(framealpha=0.5, loc='lower right', bbox_to_anchor=(1, 0), prop=font1, bbox_transform=ax1.transAxes)

# # plt.title('MNIST testing accuracy', fontsize=20)
# plt.savefig('./savedata/client_number.pdf')  # 需要改的


## 画箱线图（图太丑了，放弃）
# data1 = np.loadtxt('./savedata/s_rounds2.txt')
# data2 = np.loadtxt('./savedata/w_rounds2.txt')
# # data = np.loadtxt('./savedata/delta_w_gamma1.1.txt')
# data1 = np.absolute(data1)
# data2 = np.absolute(data2)
# data = {'s': data1, 'w': data2}
# df = pd.DataFrame(data)
# df.plot.box(title='ceshi', showfliers=False)
# plt.grid(linestyle='--', alpha=0.3)
# plt.show()

## 画噪声的散点图
# x = np.loadtxt('./savedata/delta_w.txt')
# x = np.absolute(x)
# y = np.loadtxt('./savedata/sigma.txt')
# plt.scatter(x, y, alpha=0.6, s=5)
# fig, ax1 = plt.subplots()
# plt.grid(linestyle='-.')
#
# ax1.plot(xx, yy1, '#00FF7F', marker='o', markerfacecolor='#00FF7F', linewidth=1.5, ms=4,
#          label='Testing Accuracy')
# ax1.plot(xx, yy2, '#FF6347', marker='o', markerfacecolor='#FF6347', linewidth=1.5, ms=4,
#          label='Training Accuracy')
#
# font2 = {'family': 'Times New Roman',
#          'weight': 'normal',
#          'size': 17,
#          }
# ax1.set_ylabel('Accuracy', font2)
#
# ax1.set_xlabel(r'$\eta$', font2)  # 需要改的
#
# # 设置刻度轴字体大小
# ax1.tick_params(labelsize=15)
# labels = ax1.get_xticklabels() + ax1.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
#
# # 设置刻度轴的字体大小以及间隙
# plt.tick_params(labelsize=15)
# [label.set_fontname('Times New Roman') for label in labels]
# plt.xticks(x, xx)
#
# # 设置图例的字体大小
# font1 = {'family': 'Times New Roman',
#          'weight': 'normal',
#          'size': 18,
#          }
# fig.legend(framealpha=0.5, loc='lower right', bbox_to_anchor=(1, 0), prop=font1, bbox_transform=ax1.transAxes)
#
# plt.title('MNIST testing/training accuracy', fontsize=20)
# plt.savefig('./savedata/learning.pdf')  # 需要改的


## 搞单机模型

# import sys
# import matplotlib
# import matplotlib.pyplot as plt
# import copy
# import random
# import numpy as np
# from torchvision import datasets, transforms
# import torch
# import math
# from torch import nn, autograd
# from scipy.special import comb, perm
# import math
# from functools import reduce
# from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
# from utils.options import args_parser
# from models.Update import LocalUpdate, DatasetSplit
# from models.Nets import MLP, CNNMnist, CNNCifar, SampleConvNet
# from models.Fed import FedAvg
# from models.test import test_img
# from torch.utils.data import DataLoader, Dataset

# from utils.vgg_net import vgg11
# from copy import deepcopy
# import models.Update
# import time

# matplotlib.use('Agg')

# # parse args
# args = args_parser()
# args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')


# def clamp(x,y,z):
#     floor = torch.min(x,z).to(args.device)
#     ceil = torch.max(x,z).to(args.device)
#     return torch.min(torch.max(floor,y),ceil).to(args.device)

# torch.cuda.manual_seed(1)

# # load dataset and split users
# if args.dataset == 'mnist':
#     trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
#     # dataset_train = datasets.MNIST(root="./datasets/MNIST", train=True, transform=trans_mnist)
#     # dataset_test = datasets.MNIST(root="./datasets/MNIST", train=False, transform=trans_mnist)
#     dataset_train = datasets.MNIST(root="./datasets", train=True, transform=trans_mnist)
#     dataset_test = datasets.MNIST(root="./datasets", train=False, transform=trans_mnist)

#     # sample users
#     print('iid:', args.iid)
#     if args.iid:
#         dataset_dict_clients = mnist_iid(dataset_train, args.num_users)
#     else:
#         dataset_dict_clients = mnist_noniid(dataset_train, args.num_users)
# elif args.dataset == 'cifar':
#     trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#     dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
#     dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
#     if args.iid:
#         dataset_dict_clients = cifar_iid(dataset_train, args.num_users)
#     else:
#         exit('Error: only consider IID setting in CIFAR10')
# else:
#     exit('Error: unrecognized dataset')
# img_size = dataset_train[0][0].shape

# # build model
# if args.model == 'cnn' and args.dataset == 'cifar':
#     # net_glob = CNNCifar(args=args).to(args.device)
#     net_glob = vgg11(pretrained=False, progress=False).to(args.device)
# elif args.model == 'cnn' and args.dataset == 'mnist':
#     net_glob = CNNMnist(args=args).to(args.device)
# elif args.model == 'dppca' and args.dataset == 'mnist':
#     net_glob = SampleConvNet().to(args.device)
# elif args.model == 'mlp':
#     len_in = 1
#     for x in img_size:
#         len_in *= x
#     net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
# else:
#     exit('Error: unrecognized model')
# # print(net_glob)  #net_glob是全局模型
# net_glob.train()

# # copy weights
# w_global = net_glob.state_dict()
# g_global = dict()

# key = list(w_global.keys())  # 提取每一层的名字

# for name, parms in net_glob.named_parameters():
#     g_global[name] = torch.zeros_like(parms.data).to(args.device)

# Expectation_g_2 = dict()
# # k is layer name
# for k, v in g_global.items():
#     Expectation_g_2[k] = 0.1 * v * v

# Sqrt_Expectation_g_2 = dict()
# for k, v in Expectation_g_2.items():
#     Sqrt_Expectation_g_2[k] = v.sqrt()

# # L is the number of times each worker uploads its model weights
# L = torch.zeros(args.num_users)
# delta_locals = torch.zeros(args.num_users)  # 计算每个客户的δ值

# # training
# loss_train = []
# cv_loss, cv_acc = [], []
# val_loss_pre, counter = 0, 0
# net_best = None
# best_loss = None
# val_acc_list, net_list = [], []

# alpha = torch.arange(2, 65)
# epsilon = args.epsilon

# C = args.lcf
# beta = args.beta
# sigma_0 = args.nl
# G = 1e-6

# T = 1
# K = 1
# gamma_prime = args.gamma_star
# # gamma_l = 1.2 * np.asarray([np.power(0.999, i) for i in range(32)])

# delta_0 = []
# delta_local_i = []

# acc_test_repro = []
# acc_train_repro = []


# for k_iter in range(args.rounds):
#     loss_locals = []
#     net_glob.train()
#     lot_size = args.ls
#     # dataset_local = DatasetSplit(dataset_train)
#     ldr_train = DataLoader(dataset_train, batch_size=lot_size, shuffle=True)
#     loss_func = nn.CrossEntropyLoss()
#     g = dict()
#     optimizer = torch.optim.RMSprop(net_glob.parameters(), lr=args.lr)
#     local_ep = int(args.local_ep)
#     aa = 0
#     delta_w_store = torch.empty(1).to(args.device)

#     for batch_idx, (images, labels) in enumerate(ldr_train):
#         # if aa == 0:
#         #     Expectation_g_2_locals_0 = copy.deepcopy(Expectation_g_2_locals[idx])
#         aa += 1

#         images, labels = images.to(args.device), labels.to(args.device)  # labels的长度为60
#         optimizer.zero_grad()
#         log_probs = net_glob(images)
#         w = net_glob.state_dict()  # 本地参数
#         R2 = 0
#         # for k, v in w.items():
#         #     R2 = R2 + 0.05 / 2 * torch.sum((w[k] - w_global[k]) ** 2)

#         loss = loss_func(log_probs, labels)

#         loss.backward()  # 计算梯度

#         g = dict()
#         for name, parms in net_glob.named_parameters():
#             g[name] = parms.grad
#         for k, v in g.items():
#             Expectation_g_2[k] = 0.1 * v * v + 0.9 * Expectation_g_2[k]


#         delta_w = dict()
#         for k, v in Expectation_g_2.items():
#             delta_w[k] = g[k] / torch.sqrt(v + 1e-8)

#         for k, v in w.items():
#             v = v - args.lr * delta_w[k]
#             w[k] = v

#         # optimizer.step()
#         # w = net_glob.state_dict()  # 本地参数
#         if aa == int(32):  # 本地训练32次
#             break  # 最后一次不更新参数


#         net_glob.load_state_dict(w)



#     print('rounds:', K)

#     K += 1



#     if (k_iter + 1) % 1 == 0:
#         # delta_0.append(delta)
#         # delta_local_i.append(delta_c)
#         net_glob.eval()
#         final_acc_train, final_loss_train = test_img(net_glob, dataset_train, args)
#         final_acc_test, final_loss_test = test_img(net_glob, dataset_test, args)
#         print("Training accuracy: {:.2f}".format(final_acc_train))
#         print("Testing accuracy: {:.2f}".format(final_acc_test))



# # testing
# net_glob.eval()
# acc_train, loss_train = test_img(net_glob, dataset_train, args)
# acc_test, loss_test = test_img(net_glob, dataset_test, args)
# print("Training accuracy epsilon{}_frac{}_lr{}_nl{}_beta{}_gamma{}_ls{}_lepoch{:.2f}_an{}_iid{}: {:.2f}".format(
#     args.epsilon, args.frac, args.lr, args.nl, args.beta, args.gamma_star, args.ls, args.local_ep, args.an, args.iid,
#     acc_train))
# print("Testing accuracy epsilon{}_frac{}_lr{}_nl{}_beta{}_gamma{}_ls{}_lepoch{:.2f}_an{}_iid{}: {:.2f}".format(
#     args.epsilon, args.frac, args.lr, args.nl, args.beta, args.gamma_star, args.ls, args.local_ep, args.an, args.iid,
#     acc_test))
# print('Finish!')
