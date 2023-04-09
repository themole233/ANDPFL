#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
import sys


def FedAvg(w, p):
    theta_glob = dict()
    # print(p.keys())
    # print(w.keys())
    list_key = list(p.keys())
    key = list_key[0]
    theta = w[key]
    for k,v in theta.items():
        theta_glob[k] = p[key]*v

    for key in list_key[1:]:
        theta = w[key]
        for k,v in theta.items():
            theta_glob[k] += p[key]*v
    return theta_glob


class FedAvgCollect:
    def __init__(self):
        self.w_glob = None

    def get_result(self):
        return self.w_glob

    def add(self, w, p):
        if self.w_glob == None:
            self.w_glob = dict()
            for k, v in w.items():
                self.w_glob[k] = p * v
        else:
            for k, v in w.items():
                self.w_glob[k] += p * v

    def clear(self):
        self.w_glob = None



