from itertools import count

import torch
import gym
from model import DQN_fully_conn
from distance_extractor import get_state
from distance_extractor import Ray
from distance_extractor import CAR_X, CAR_Y, SCREEN_HEIGHT, SCREEN_WIDTH


AS_GRAY = True

# Use CUDA 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Init CarRacing env in discrete control
env = gym.make('CarRacing-v1', continuous=False)

# Prep model
rays = [
            Ray(CAR_X, CAR_Y, 0, (SCREEN_HEIGHT, SCREEN_WIDTH)),
            Ray(CAR_X, CAR_Y, 180, (SCREEN_HEIGHT, SCREEN_WIDTH)),
            Ray(CAR_X, CAR_Y, -45, (SCREEN_HEIGHT, SCREEN_WIDTH)),
            Ray(CAR_X, CAR_Y, -135, (SCREEN_HEIGHT, SCREEN_WIDTH)),
            Ray(CAR_X, CAR_Y, -90, (SCREEN_HEIGHT, SCREEN_WIDTH))
            ]

model = torch.load("models/target_net.pt")
model.eval()

env.reset()
state = get_state(env, rays).float()

for t in count():
    env.render()
    with torch.no_grad():
        # largest col value of each row
        action = model(state).max(1)[1].view(1, 1)
    print(action)
    
    _, reward, done, _ = env.step(action.item())
    reward = torch.tensor([reward], device=device)
    
    state = get_state(env, rays).float()
    print(state)

    if done:
        print('Done')
        break 

env.render()    
env.close()
