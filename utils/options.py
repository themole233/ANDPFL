#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--optimizer', type=str, default="RMSPROP", help="selected optimizer")
    parser.add_argument('--rounds', type=int, default=300, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=1.0, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=float, default=32, help="the number of local epochs: E")

    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    # model arguments
    parser.add_argument('--save', action='store_true', default=False, help='Whether save parameter')
    parser.add_argument('--beta', type=float, default=1.2, help='beta')
    parser.add_argument('--gamma_star', type=float, default=0.95, help='gamma_star')
    parser.add_argument('--model', type=str, default='dppca', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    parser.add_argument('--lcf', type=float, default=4.0, help='Local clip factor')
    parser.add_argument('--nl', type=float, default=22.0, help='Noise level')
    parser.add_argument('--ls', type=int, default=600, help='Lot size')
    parser.add_argument('--an', action='store_true', default=True, help='Whether adaptive noise or not')

    # 隐私预算
    parser.add_argument('--epsilon', type=float, default=8.0, help='Target epsilon')
    parser.add_argument('--delta', type=float, default=1e-5, help='Target delta')

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")  # 类别数
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of imges")  # 图片是几维的
    parser.add_argument('--gpu', type=int, default=-1, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')
    args = parser.parse_args()
    return args
