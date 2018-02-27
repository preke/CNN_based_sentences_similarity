import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F


def train(train_set, test_set, nn_model, args):
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)