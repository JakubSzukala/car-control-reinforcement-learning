# Car control with reinforcement learning 

## Some relevant materials on reinforcement learning

- [PyTorch](https://pytorch.org/)
- [Stable baselines3 - like sklearn but for RL, lot of abstraction, backend in PyTorch](https://github.com/DLR-RM/stable-baselines3)
- [Open AI Gym - Toolkit for reinforcement learning](https://gym.openai.com/)
- [SB3 Zoo - Alternative to Open AI Gym speciffically for SB3](https://github.com/DLR-RM/rl-baselines3-zoo)
- [Intro to RL and Stable Baselines + AI Gym](https://www.youtube.com/watch?v=XbWhJdQgi7E&list=PLQVvvaa0QuDf0O2DWwLZBfJeYY-JOeZB1)

Initial step when solving the problem will be to get hands on experience with
the toolkit and some basics of RL. So probably we will start from playing with 
examples.

## Installation steps and potential cavetas

### PyTorch

Start from installing PyTorch, ideally operate in virtual enviroment [PyTorch local installation](https://pytorch.org/get-started/locally/):
For ubuntu 20.04 use pip3 command:
```
$ pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```
This will install PyTorch with Pip package manager and with CUDA 11.3 support 
for Ubuntu 20.04. If You do not have a graphics card with CUDA support or use
other operating system refer to [documentation](https://pytorch.org/get-started/locally/).
**Make sure that You pick correct system and correct CUDA support!**

### Open AI Gym
Simply run a command:
```
$ pip3 install gym
```
For details refer to [documentation](https://gym.openai.com/docs/)

### Stable Baselines3 
Simply run a command:
```
$ pip3 install stable-baselines3[extra]
```
For details refer to [documentation](https://stable-baselines3.readthedocs.io/en/master/guide/install.html)



