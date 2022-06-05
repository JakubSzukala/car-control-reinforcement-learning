import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch import optim

from image_extractor import get_screen
from model import DQN
from model import ReplayMemory
from model import Transition

import random
import math
from itertools import count
from collections import deque
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make('CarRacing-v1', continuous=False)

# For gray scale we can use wrapper:
# env = GrayScaleObservation(gym.make('CarRacing-v1'))

env.action_space.seed(42)

observation, info = env.reset(seed=42, return_info=True)

"""
###############################################################################
# Initialization of models and training procedure 
###############################################################################
"""
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

AS_GRAY = True 

init_screen = get_screen(env, as_gray=AS_GRAY)
_, channels, screen_h, screen_w = init_screen.shape
n_actions = env.action_space.n # type: ignore 

# https://stackoverflow.com/questions/54237327
policy_net = DQN(channels, screen_h, screen_w, n_actions, device).to(device)
target_net = DQN(channels, screen_h, screen_w, n_actions, device).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval() # enable evaluation mode

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


steps_done = 0

# TODO: change this global EPS_START etc...
def select_action(state):
    global steps_done 
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # largest col value of each row
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor(
                [[random.randrange(n_actions)]], 
                device=device, dtype=torch.long)

episode_durations = []


# TODO: change to plot rewards or dist or smth like that 
def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(1)  # pause a bit so that plots are updated


def optimize_model(memory, device, policy_net, target_net, optimizer):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    # https://stackoverflow.com/a/19343/3343043
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None,
            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
    # Compute V(s_{t+1}) for all next states.
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(
            non_final_next_states).max(1)[0].detach()
    
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(
            state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def queue2frame_stack(deque):
    frame_stack = torch.cat(tuple(deque), 0)
    frame_stack = torch.squeeze(frame_stack, 1)
    print(frame_stack.shape)
    # This should probably be in pytorch convention (CHW)? if not transpose?
    return frame_stack


for n_episode in range(90):
    total_reward = 0

    # Getting input 
    env.reset()
    """
    last_screen = get_screen(env, as_gray=AS_GRAY)
    current_screen = get_screen(env, as_gray=AS_GRAY) 
    state = current_screen - last_screen
    """
    # State is 3 previous frames stacked together and fed to the network
    # Credits: https://github.com/andywu0913/OpenAI-GYM-CarRacing-DQN
    # TODO: improve this, remove hard coding and abstract it
    state_queue = deque([get_screen(env, as_gray=AS_GRAY) * 3],
            maxlen=3)
    for t in count():
        # Convert current frames q into state
        state = queue2frame_stack(state_queue)
        next_state = state # temporary

        action = select_action(state)
        _, reward, done, _ = env.step(action.item()) # type: ignore
        reward = torch.tensor([reward], device=device)
        total_reward += reward

        # Observe new state
        next_frame = get_screen(env, as_gray=AS_GRAY)
        if not done:
            # Add to state q new frame and convert it into new state
            state_queue.append(next_frame)
            next_state = queue2frame_stack(state_queue)
        
        # Add to replay memory 
        memory.push(state, action, next_state, reward)
        
        state = next_state

        optimize_model(memory, device, policy_net, target_net, optimizer)
        if done:
            print('Total reward gained: {}'\
                  'for episode {}: and duration: {}'.format(
                      total_reward, n_episode, t+1))
            episode_durations.append(t + 1)
            # TODO: we do not care about duration.... 
            # that much plot smth more relevant
            #plot_durations()
            break
    
    if n_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()    
env.close()
#plt.ioff()
#plt.show()
# https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference
torch.save(target_net, './models/target_net.pt')
torch.save(policy_net, './models/policy_net.pt')

"""
    # Displaying  
    plt.figure()
    plt.imshow(screen.cpu().squeeze(0).permute(1, 2, 0).numpy(),
            interpolation='none')
    plt.show()

    observation, reward, done, info = env.step(
            env.action_space.sample())
    #print('Observation: ', observation)
    print('Random action: ', env.action_space.sample())
    print('Random action shape: ', env.action_space.shape)
    print('Reward: ', reward)
    if done:
        observation, info = env.reset(return_info=True)
    """
