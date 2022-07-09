from itertools import count

import torch
import gym
from model import DQN
from image_extractor import get_screen

AS_GRAY = True

# Use CUDA 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Init CarRacing env in discrete control
env = gym.make('CarRacing-v1', continuous=False)

# Prep model
init_screen = get_screen(env, as_gray=AS_GRAY)
_, channels, screen_h, screen_w = init_screen.shape
n_actions = env.action_space.n # type: ignore 

#model = DQN(channels, screen_h, screen_w, n_actions, device).to(device)

model = torch.load("models/target_net.pt")
model.eval()

env.reset()
last_screen = get_screen(env, as_gray=AS_GRAY)
current_screen = get_screen(env, as_gray=AS_GRAY) 
state = current_screen - last_screen

for t in count():
    env.render()    
    with torch.no_grad():
        # largest col value of each row
        action = model(state).max(1)[1].view(1, 1)
    
    _, reward, done, _ = env.step(action.item())
    reward = torch.tensor([reward], device=device)
    
    last_screen = current_screen
    current_screen = get_screen(env, as_gray=AS_GRAY)
    if not done:
        next_state = current_screen - last_screen
    else:
        next_state = None

    state = next_state
    if done:
        print('Done')
        break 

env.render()    
env.close()
