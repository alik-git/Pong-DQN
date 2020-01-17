import gym
from collections import namedtuple
import numpy as np 
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim

HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70

class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, num_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            # connects between input layer and hidden layer
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),  # makes output look/behave nicer
            # connects between hidden layer and output layer
            nn.Linear(hidden_size, num_actions)
        )
    
    # one forward pass through the whole network. input: vector x
    def forward(self, x):
        return self.net(x)

Episode = namedtuple('Episode', field_names=['reward', 'steps'])

EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])

