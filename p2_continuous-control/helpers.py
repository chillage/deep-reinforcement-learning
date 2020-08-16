import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import argparse
from distutils.util import strtobool
import collections
import numpy as np
import time
import random
import os

from typing import List, Any


# modified from https://github.com/seungeunrho/minimalRL/blob/master/dqn.py#
class ReplayBuffer():
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transitions : List[Any]):
        self.buffer.extend(transitions)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask_lst.append(done_mask)

        return np.array(s_lst), np.array(a_lst), \
               np.array(r_lst), np.array(s_prime_lst), \
               np.array(done_mask_lst)


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, device, action_size, observation_size):
        super(QNetwork, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(
            np.array((observation_size,)).prod() + np.prod((action_size,)), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.Tensor(x).to(self.device)
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, device, action_size, observation_size):
        super(Actor, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(np.array((observation_size,)).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod((action_size,)))

    def forward(self, x):
        x = torch.Tensor(x).to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc_mu(x))
