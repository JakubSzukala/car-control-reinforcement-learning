from itertools import count

import torch
import gym
from model import DQN_fully_conn
from distance_extractor import get_state
from distance_extractor import Ray
from distance_extractor import CAR_X, CAR_Y, SCREEN_HEIGHT, SCREEN_WIDTH
from stable_baselines3.ppo.ppo import PPO

env = gym.make('CarRacing-v2', continuous=False)

model = PPO.load('logs/PPO-CnnPolicy-19-09-2022-00-58-59_best/best_model.zip')

obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

