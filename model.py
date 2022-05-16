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

