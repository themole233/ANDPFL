import sys
import matplotlib
import matplotlib.pyplot as plt
import copy
import numpy as np
from torch.nn import parameter
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

def B(l, sigma_star):
    B = 0
    for i in range(l + 1):
        try:
            B = B + pow(-1, i) * comb(l, i) * math.exp(i * (i - 1) / (2 * sigma_star ** 2))
        except OverflowError:
            B = float('inf')

    return B


L = 0
# delta = 1e-4
epsilon = 0.5
q = 0.01
sigma_0 = 3 
alpha = np.arange(2, 65)
for i in range(10):
    L+=1
    x1 = 4 * (math.exp(1 / sigma_0 ** 2) - 1)
    x2 = 2 * math.exp(1 / sigma_0 ** 2)
    x = min(x1, x2)

    log_delta = np.zeros(alpha.shape)
    delta_np = np.zeros(alpha.shape)

    for i in range(63):
        calc_sum = 0
        if alpha[i] == 2:
            calc_sum = 0
        else:
            for j in range(3, alpha[i] + 1):
                try:
                    try:
                        xx = pow(q, j) * comb(alpha[i], j) * math.sqrt(abs(B(2 * int(math.ceil(j / 2)), sigma_0) * B(2 * int(math.floor(j / 2)),sigma_0)))
                    except OverflowError:
                        xx = float('inf')
                    calc_sum = calc_sum + xx
                except OverflowError:
                    calc_sum = float('inf')
        log_delta[i] = -(epsilon - (L / (alpha[i] - 1)) * math.log(1 + comb(alpha[i], 2)* q ** 2 * x + 4 * calc_sum)) * (alpha[i] - 1)

        try:
            delta_np[i] = math.exp(log_delta[i])
        except OverflowError:
            delta_np[i] = float('inf')

    delta = np.min(delta_np)
    print(delta)