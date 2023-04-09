#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
import sys


calc = 40
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label



class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, net=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.net = net
        self.optim = torch.optim.SGD(self.net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

    def train(self, batch_size, lr_train):
        self.net.train()
        # train and update
        self.args.lr = lr_train
        # optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        # optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        # optimizer = torch.optim.RMSprop(net.parameters(), lr=self.args.lr)

        epoch_loss = []
        # for iter in range(self.args.local_ep):
        for iter in range(batch_size):
            batch_loss = []
            # print(enumerate(self.ldr_train))
            # sys.exit()
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                self.net.zero_grad()
                log_probs = self.net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            new_grad = dict()
            for name, parms in self.net.named_parameters():	
                # print('-->name:', name, '-->grad_requirs:',parms.requires_grad, \
                # ' -->grad_value:',parms.grad)
                new_grad[name] = parms.grad
        return new_grad, sum(epoch_loss) / len(epoch_loss)

    def modify_grad(new_grad):
        #######
        for name, parms in self.net.named_parameters():	
            # print('-->name:', name, '-->grad_requirs:',parms.requires_grad, \
            # ' -->grad_value:',parms.grad)
            parms.grad = new_grad[name]

        self.optim.step()
