from itertools import count

import torch
import gym
from model import DQN_fully_conn
from distance_extractor import get_state
from distance_extractor import Ray
from distance_extractor import CAR_X, CAR_Y, SCREEN_HEIGHT, SCREEN_WIDTH
from stable_baselines3.ppo.ppo import PPO
from car_racing_env_wrapper import CarRacingDistanceStateWrapper
root_env = gym.make('CarRacing-v2', continuous=False)
env = CarRacingDistanceStateWrapper(root_env)
#logs/PPO-MlpPolicy-26-09-2022-00-02-47_final.zip
# Best best 
# logs/PPO-CnnPolicy-19-09-2022-00-58-59_best/best_model.zip
model = PPO.load('logs/PPO-MlpPolicy-27-09-2022-21-36-09_best/best_model.zip')

obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

