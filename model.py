import random
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


"""
################################################################################
# Replay Memory 
################################################################################
"""
# same as tuple, but access by name, self documentation and neat __repr__
Transition = namedtuple('Transition', 
        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        # Deque for faster append / pop than list
        self.memory = deque([], maxlen=capacity)

    
    # any num of args, * transforms into iterable
    def push(self, *args):
        self.memory.append(Transition(*args))


    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


    def __len__(self):
        return len(self.memory)

"""
################################################################################
# Model definition 
################################################################################
"""
class DQN_conv(nn.Module):
    # TODO: this is up for experimentation
    def __init__(self, c, h, w, outputs, device):
        super(DQN_conv, self).__init__()
        self.conv1 = nn.Conv2d(
                in_channels=c,   # gray
                out_channels=16, # arbitrary? 
                kernel_size=5,   # arbitrary? 
                stride=2)        # arbitrary? 
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        
        # TODO: we should examine that so we understand linear algebra behind
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        
        # Initialization  
        convw = w
        convh = h
        
        # Is the range in c correct?
        for _ in range(3):
            convw = conv2d_size_out(convw)
            convh = conv2d_size_out(convh)
            #print(convw)
            #print(convh)
        #print('end')

        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)
        print(linear_input_size)
        self.device = device


    def forward(self, x):
        """
        Receive a Tensor containing the input and return a Tensor containing
        the outptut.
        """
        x = x.to(self.device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))  


class DQN_fully_conn(nn.Module):
    def __init__(self, dev):
        super(DQN_fully_conn, self).__init__()
        self.device = dev
        self.linear_relu_stack = nn.Sequential(
                nn.Linear(5, 8),
                nn.ReLU(),
                nn.Linear(8, 8),
                nn.ReLU(),
                nn.Linear(8, 5),
                nn.ReLU()
                )

                #nn.linear(5, 16),
                #nn.relu(),
                #nn.linear(16, 32),
                #nn.relu(),
                #nn.linear(32, 32), 
                #nn.ReLU(),
                #nn.Linear(32, 5))




    def forward(self, x):
        #print('X: ', x)
        x = x.to(self.device)
        logits = self.linear_relu_stack(x)
        return logits 


"""
################################################################################
# Training 
################################################################################
"""

    



















