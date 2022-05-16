import gym
import matplotlib
import matplotlib.pyplot as plt

import torch

from image_extractor import get_screen

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make('CarRacing-v1')
#env = gym.make('LunarLander-v2')
env.action_space.seed(42)

observation, info = env.reset(seed=42, return_info=True)

for _ in range(1000):

    screen = get_screen(env)
    plt.figure()
    plt.imshow(screen.cpu().squeeze(0).permute(1, 2, 0).numpy(),
            interpolation='none')
    plt.show()

    #env.render()
    observation, reward, done, info = env.step(
            env.action_space.sample())
    #print('Observation: ', observation)
    print('Random action: ', env.action_space.sample())
    print('Reward: ', reward)
    if done:
        observation, info = env.reset(return_info=True)

env.close()
