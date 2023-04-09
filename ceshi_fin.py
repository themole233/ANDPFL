from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import torch
import copy
from utils.options import args_parser
import math
import time

# 计算noise level
# for alpha in range(64):
#     print(35*(alpha+2)*(alpha+1)/(8.0*(alpha+1)-4*math.log(10))) #16是T，0.5是ε
# exit()

args = args_parser()

# 画和其他算法的比较图
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 24,
}
# x = ['0.3','0.5','1.0','2.0','4.0','8.0']
# y1 = [19.87, 30.25, 46.92, 58.60, 89.03, 94.28] # LDP-FL
# y2 = [69.59, 82.11, 90.16, 91.01, 92.14, 92.61] # PrivateFL
# y3 = [77.20, 83.08, 89.94, 91.25, 92.42, 93.88] # Dp-dyns
# y4 = [87.26, 93.70, 93.81, 94.02, 94.80, 95.01]
# plt.plot(x, y1, marker='o', ms=4, label = 'LDP-FL')
# plt.plot(x, y2, marker='o', ms=4, label = 'PrivateDL')
# plt.plot(x, y3, marker='o', ms=4, label = 'DP-dynS')
# plt.plot(x, y4, marker='o', ms=4, label = 'ANDPFL')
# plt.tick_params(labelsize=24)
# plt.yticks(np.arange(0,110,10))
# plt.xlabel('Privacy Budget', font2)
# plt.ylabel('Accuracy', font2)
# plt.grid(linestyle='-.')
# plt.legend(framealpha=0.5, loc='lower right', prop=font2)
# plt.show()

# 打印delta
# alpha = torch.arange(2, 65)
# delta_repo = torch.zeros(torch.Size([158]))
# for L in range(1, 158):
#     delta = torch.zeros(alpha.shape)  # 每一个alpha值对应一个δ值
#     for i in range(63):
#         delta[i] = torch.exp(-(alpha[i] - 1) * (args.epsilon - (L + 1) * alpha[i] / (2 * args.nl ** 2)))
#     delta_repo[L] = torch.min(delta)*1e+4
# np.savetxt('F:/实验室/pythoncode/federated-learning-master-0331/savedata/delta_eps{}.txt'.format(args.epsilon), np.array(delta_repo))

# yy1 = np.loadtxt("./saved_acc_optim/test_eps{}_optimSGD.txt".format(args.epsilon))
# yy2 = np.loadtxt("./saved_acc_optim/test_eps{}_optimMomentum.txt".format(args.epsilon))
# yy3 = np.loadtxt("./saved_acc_optim/test_eps{}_optimRMSPROP.txt".format(args.epsilon))
# yy4 = np.loadtxt("./saved_acc_optim/test_eps{}_optimAdam.txt".format(args.epsilon))
# #
# x = np.loadtxt("./savedata/round_epsilon{}.txt".format(args.epsilon), delimiter=',')
# yy5 = np.loadtxt("./savedata/delta_epsilon{}.txt".format(args.epsilon), delimiter=',')
# 对eps=8.0的处理
# xx = x[0:155].reshape((31, 5))[:, 1]
# yy1 = yy1[0:155].reshape((31, 5))[:, 1]
# yy2 = yy2[0:155].reshape((31, 5))[:, 1]
# yy3 = yy3[0:155].reshape((31, 5))[:, 1]
# yy4 = yy4[0:155].reshape((31, 5))[:, 1]
# yy5 = yy5[0:155].reshape((31, 5))[:, 1]
# 对eps=2.0的处理
# xx = x[0:90].reshape((18, 5))[:, 1]
# yy1 = yy1[0:90].reshape((18, 5))[:, 1]
# yy2 = yy2[0:90].reshape((18, 5))[:, 1]
# yy3 = yy3[0:90].reshape((18, 5))[:, 1]
# yy4 = yy4[0:90].reshape((18, 5))[:, 1]
# yy5 = yy5[0:90].reshape((18, 5))[:, 1]
# 对eps=0.5的处理
# xx = x
# xx = xx + 1
# #
# fig,ax1 = plt.subplots()
# plt.grid(linestyle='-.')
# #
# ax1.plot(xx, yy1, '#00FF7F', marker='o', markerfacecolor='#00FF7F', linewidth=1.5, ms=4, label = 'SGD testing Accuracy')
# ax1.plot(xx, yy2, '#FF6347', marker='o', markerfacecolor='#FF6347', linewidth=1.5, ms=4, label = 'Momentum testing Accuracy')
# ax1.plot(xx, yy3, 'blue', marker='o', markerfacecolor='blue', linewidth=1.5, ms=4, label = 'RMSPROP testing Accuracy')
# ax1.plot(xx, yy4, '#DB70DB', marker='o', markerfacecolor='#DB70DB', linewidth=1.5, ms=4, label = 'Adam testing Accuracy')

# font2 = {'family' : 'Times New Roman',
# 'weight' : 'normal',
# 'size'   : 24,
# }
# ax1.set_ylabel('Accuracy',font2)
# ax1.set_xlabel('Round',font2)
# ax1.set_yticks(np.arange(10,110,10))
# ax1.set_ylim(10,100)
#
# plt.xticks(np.arange(0,len(x)+1,20)) #画eps=8.0
# plt.xticks(np.arange(0,len(x)+1,10)) #画eps=2.0
# plt.xticks(np.arange(0,len(xx)+1)) #eps=0.5

# # 设置刻度轴字体大小
# ax1.tick_params(labelsize=24)
# labels = ax1.get_xticklabels() + ax1.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
# ax2 = ax1.twinx()  # this is the important function
# ax2.plot(xx, yy5, '#FF69B4', marker='*', markerfacecolor='#FF69B4', linewidth=1.5, ms=4, label = 'δ')
# ax2.set_ylabel('δ(1e-4)',font2)
# ax2.set_ylim(0,1)

# # 设置刻度轴的字体大小以及间隙
# plt.tick_params(labelsize=24)
# labels = ax2.get_xticklabels() + ax2.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
# # 设置图例的字体大小
# font1 = {'family' : 'Times New Roman',
# 'weight' : 'normal',
# 'size'   : 24,
# }
# fig.legend(framealpha=0.5, loc='lower right', bbox_to_anchor=(1,0), prop=font1, bbox_transform=ax1.transAxes)
# plt.show()
# # plt.title('MNIST Testing Accuracy for Optimizers, ε={}'.format(args.epsilon),fontsize=18)
# plt.savefig('./savedata/optim_epsilon{}.pdf'.format(args.epsilon))

font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 24,
         }
# 画参数变化图
# xx = np.loadtxt('./savedata/frac.txt')
# yy1 = np.loadtxt('./savedata/frac_sgd.txt')
# yy2 = np.loadtxt('./savedata/frac_momentum.txt')
# yy3 = np.loadtxt('./savedata/frac_rmsprop.txt')
# yy4 = np.loadtxt('./savedata/frac_adam.txt')
#
xx = np.loadtxt('./savedata/noise_level.txt')
yy1 = np.loadtxt('./savedata/noise_level_sgd.txt')
yy2 = np.loadtxt('./savedata/noise_level_momentum.txt')
yy3 = np.loadtxt('./savedata/noise_level_rmsprop.txt')
yy4 = np.loadtxt('./savedata/noise_level_adam.txt')
#
# xx = np.loadtxt('./savedata/gamma_0.txt')
# yy1 = np.loadtxt('./savedata/gamma_sgd.txt')
# yy2 = np.loadtxt('./savedata/gamma_momentum.txt')
# yy3 = np.loadtxt('./savedata/gamma_rmsprop.txt')
# yy4 = np.loadtxt('./savedata/gamma_adam.txt')
#
# xx = np.loadtxt('./savedata/number_0.txt')
# yy1 = np.loadtxt('./savedata/number_sgd.txt')
# yy2 = np.loadtxt('./savedata/number_momentum.txt')
# yy3 = np.loadtxt('./savedata/number_rmsprop.txt')
# yy4 = np.loadtxt('./savedata/number_adam.txt')
# # #
x = np.arange(len(xx))
xx = [str(x) for x in xx]  # 转化为字符串
# xx = [str(int(x)) for x in xx]
# #
fig, ax1 = plt.subplots()
plt.grid(linestyle='-.')
#
ax1.plot(xx, yy1, '#00FF7F', marker='o', markerfacecolor='#00FF7F', linewidth=1.5, ms=4, label='SGD testing Accuracy')
ax1.plot(xx, yy2, '#FF6347', marker='o', markerfacecolor='#FF6347', linewidth=1.5, ms=4,
         label='Momentum testing Accuracy')
ax1.plot(xx, yy3, 'blue', marker='o', markerfacecolor='blue', linewidth=1.5, ms=4, label='RMSPROP testing Accuracy')
ax1.plot(xx, yy4, '#DB70DB', marker='o', markerfacecolor='#DB70DB', linewidth=1.5, ms=4, label='Adam testing Accuracy')

# ax1.plot(xx, yy1, '#00FF7F', marker='o', markerfacecolor='#00FF7F', linewidth=1.5, ms=4,
#          label='Testing Accuracy')
# ax1.plot(xx, yy2, '#FF6347', marker='o', markerfacecolor='#FF6347', linewidth=1.5, ms=4,
#          label='Training Accuracy')
# font1 = {'family': 'Times New Roman',
#          'weight': 'normal',
#          'size': 16,
#          }

ax1.set_ylabel('Accuracy', font2)
#
ax1.set_xlabel('$\sigma_0$', font2)  # 需要改的
# #
plt.ylim((50, 100))
# #
# # # 设置刻度轴字体大小
ax1.tick_params(labelsize=24)
labels = ax1.get_xticklabels() + ax1.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
# #
# # # 设置刻度轴的字体大小以及间隙
plt.tick_params(labelsize=24)
[label.set_fontname('Times New Roman') for label in labels]
plt.xticks(x, xx)
#
# # 设置图例的字体大小
# # font1 = {'family': 'Times New Roman',
# #          'weight': 'normal',
# #          'size': 18,
# # #          }
fig.legend(framealpha=0.5, loc='lower right', bbox_to_anchor=(1, 0), prop=font2, bbox_transform=ax1.transAxes)
plt.show()
# # plt.savefig('./savedata/number.pdf')  # 需要改的
# # plt.savefig('./savedata/noise_level.pdf')
# # plt.savefig('./savedata/gamma.pdf')
# plt.savefig('./savedata/frac.pdf')

# 画随隐私预算增长的堆叠直方图
# labels = [0.5, 1.0, 2.0, 4.0, 8.0]
# labels = [str(x) for x in labels]  # 转化为字符串
# # y1 = np.loadtxt('./savedata/noise_free_mnist.txt')
# # y2 = [97.82, 97.94, 98.09, 98.11, 98.19]  # RMSPROP
# # y3 = [96.5, 96.84, 97.07, 97.18, 97.22]  # Adam
# # y4 = [95.83, 95.90, 96.03, 96.43, 96.89]  # Momentum
# # y5 = [94.92, 95.09, 95.13, 95.34, 95.99]  # SGD

# y1 = np.loadtxt('./savedata/noise_free_cifar.txt')
# y2 = [54.25, 58.22, 60.01, 60.29, 60.92]  # RMSPROP
# y3 = [49.01, 55.26, 59.53, 59.62, 59.96]  # Adam
# y4 = [44.97, 49.86, 58.08, 58.32, 58.41]  # Momentum
# y5 = [42.99, 47.80, 57.87, 58.03, 58.28]  # SGD

# x = np.arange(len(labels))  # x轴刻度标签位置

# width = 0.15  # 柱子的宽度
# plt.bar(x - 2*width, y1, width, label='Noiseless')
# plt.bar(x - 1*width, y2, width, label='RMSPROP')
# plt.bar(x, y3, width, label='Adam')
# plt.bar(x + 1*width, y4, width, label='Momentum')
# plt.bar(x + 2*width, y5, width, label='SGD')
# plt.ylabel('Accuracy', font2)
# plt.xticks(x, labels=labels)
# plt.legend(framealpha=0.5, loc='lower right', prop=font2)
# plt.yticks(np.arange(35,70,5))
# plt.ylim((35,65))
# plt.tick_params(labelsize=24)
# plt.xlabel('Privacy Budget', font2)
# plt.show()
# plt.savefig('./savedata/epsilon.pdf')  # 需要改的

# # 画随隐私预算增长的直方图
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


# 画随客户增长的直方图
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


# 画箱线图 s代表估计结果
# import pandas as pd
# data1 = np.loadtxt('F:/实验室/pythoncode/savedata/s_fin_rounds10_optim{}_eps{}.txt'.format(args.optimizer, args.epsilon))
# data11 = np.loadtxt('F:/实验室/pythoncode/savedata/delta_w_sum_fin_rounds10_optim{}_eps{}.txt'.format(args.optimizer, args.epsilon))

# data2 = np.loadtxt('F:/实验室/pythoncode/savedata/s_fin_rounds20_optim{}_eps{}.txt'.format(args.optimizer, args.epsilon))
# data22 = np.loadtxt('F:/实验室/pythoncode/savedata/delta_w_sum_fin_rounds20_optim{}_eps{}.txt'.format(args.optimizer, args.epsilon))

# data3 = np.loadtxt('F:/实验室/pythoncode/savedata/s_fin_rounds30_optim{}_eps{}.txt'.format(args.optimizer, args.epsilon))
# data33 = np.loadtxt('F:/实验室/pythoncode/savedata/delta_w_sum_fin_rounds30_optim{}_eps{}.txt'.format(args.optimizer, args.epsilon))

# data4 = np.loadtxt('F:/实验室/pythoncode/savedata/s_fin_rounds40_optim{}_eps{}.txt'.format(args.optimizer, args.epsilon))
# data44 = np.loadtxt('F:/实验室/pythoncode/savedata/delta_w_sum_fin_rounds40_optim{}_eps{}.txt'.format(args.optimizer, args.epsilon))

# data5 = np.loadtxt('F:/实验室/pythoncode/savedata/s_fin_rounds50_optim{}_eps{}.txt'.format(args.optimizer, args.epsilon))
# data55 = np.loadtxt('F:/实验室/pythoncode/savedata/delta_w_sum_fin_rounds50_optim{}_eps{}.txt'.format(args.optimizer, args.epsilon))

# data6 = np.loadtxt('F:/实验室/pythoncode/savedata/s_fin_rounds60_optim{}_eps{}.txt'.format(args.optimizer, args.epsilon))
# data66 = np.loadtxt('F:/实验室/pythoncode/savedata/delta_w_sum_fin_rounds60_optim{}_eps{}.txt'.format(args.optimizer, args.epsilon))

# data7 = np.loadtxt('F:/实验室/pythoncode/savedata/s_fin_rounds70_optim{}_eps{}.txt'.format(args.optimizer, args.epsilon))
# data77 = np.loadtxt('F:/实验室/pythoncode/savedata/delta_w_sum_fin_rounds70_optim{}_eps{}.txt'.format(args.optimizer, args.epsilon))

# data8 = np.loadtxt('F:/实验室/pythoncode/savedata/s_fin_rounds80_optim{}_eps{}.txt'.format(args.optimizer, args.epsilon))
# data88 = np.loadtxt('F:/实验室/pythoncode/savedata/delta_w_sum_fin_rounds80_optim{}_eps{}.txt'.format(args.optimizer, args.epsilon))

# data9 = np.loadtxt('F:/实验室/pythoncode/savedata/s_fin_rounds90_optim{}_eps{}.txt'.format(args.optimizer, args.epsilon))
# data99 = np.loadtxt('F:/实验室/pythoncode/savedata/delta_w_sum_fin_rounds90_optim{}_eps{}.txt'.format(args.optimizer, args.epsilon))

# data1 = np.absolute(data1)
# data11 = np.absolute(data11)

# data2 = np.absolute(data2)
# data22 = np.absolute(data22)

# data3 = np.absolute(data3)
# data33 = np.absolute(data33)

# data4 = np.absolute(data4)
# data44 = np.absolute(data44)

# data5 = np.absolute(data5)
# data55 = np.absolute(data55)

# data6 = np.absolute(data6)
# data66 = np.absolute(data66)

# data7 = np.absolute(data7)
# data77 = np.absolute(data77)

# data8 = np.absolute(data8)
# data88 = np.absolute(data88)

# data9 = np.absolute(data9)
# data99 = np.absolute(data99)

# data = {'10': data1, '20': data2, '30': data3, '40': data4, '50': data5, '60': data6, '70': data7, '80': data8, '90': data9}
# df = pd.DataFrame(data)

# dataa = {'10': data11, '20': data22, '30': data33, '40': data44, '50': data55, '60': data66, '70': data77, '80': data88, '90': data99}
# dff = pd.DataFrame(dataa)

# font2 = {'family': 'Times New Roman',
#          'weight': 'normal',
#          'size': 24,
#          }

# labels = ['10', '20', '30', '40', '50', '60', '70', '80', '90']

# plt.subplot(211)
# plt.boxplot(df, showfliers=False, labels = labels, patch_artist = True,
#             boxprops = {'color':'pink','linewidth':'2.0','facecolor':'pink'}, 
#             capprops={'color':'black','linewidth':'1.0'}, 
#             whiskerprops = {'color':'black','linewidth':'1.0'})
# plt.ylabel('Original Value', font2)
# plt.yticks(np.arange(0,0.015,0.003))
# plt.tick_params(labelsize=24)
# plt.grid(linestyle='--', alpha=0.3)

# plt.subplot(212)
# plt.boxplot(dff, showfliers=False, labels = labels, patch_artist = True,
#             boxprops = {'color':'SkyBlue','linewidth':'2.0','facecolor':'SkyBlue'}, 
#             capprops={'color':'black','linewidth':'1.0'}, 
#             whiskerprops = {'color':'black','linewidth':'1.0'})
# plt.xlabel('Round', font2)
# plt.ylabel('Estimation', font2)
# plt.yticks(np.arange(0,0.015,0.003))
# plt.tick_params(labelsize=24)
# plt.grid(linestyle='--', alpha=0.3)
# plt.savefig('./savedata/parameters_compare_box.pdf')

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
