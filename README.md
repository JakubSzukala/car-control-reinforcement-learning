# Car control with reinforcement learning 

## Materials
#### Some relevant materials on reinforcement learning
- [PyTorch](https://pytorch.org/)
- [Stable baselines3 - like sklearn but for RL, lot of abstraction, backend in PyTorch](https://github.com/DLR-RM/stable-baselines3)
- [Open AI Gym - Toolkit for reinforcement learning](https://gym.openai.com/)
- [SB3 Zoo - Alternative to Open AI Gym speciffically for SB3](https://github.com/DLR-RM/rl-baselines3-zoo)
- [Intro to RL and Stable Baselines + AI Gym](https://www.youtube.com/watch?v=XbWhJdQgi7E&list=PLQVvvaa0QuDf0O2DWwLZBfJeYY-JOeZB1)

#### Materials from our supervisor
- [Reinforcement learning DQN algorithm tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [Example of DQN car racing in Gym env](https://github.com/andywu0913/OpenAI-GYM-CarRacing-DQN/blob/master/resources/trial_600.gif)
- [Example of DQN and multicar racing in Gym env](https://github.com/igilitschenski/multi_car_racing)

Initial step when solving the problem will be to get hands on experience with
the toolkit and some basics of RL. So probably we will start from playing with 
examples.

## Installation steps and potential cavetas
When installed with pipenv (regular Python venvs manager), there were some [errors](https://stackoverflow.com/questions/44198228)
with LunarLander example. Additional necessary packages were missing and the 
easiest and most coherent way to install them was to use Anaconda package manager 
for data science. So I would recommend using conda. Check out [conda installation guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)
and [getting started with conda guide](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html)

conda and pip [should not be used together](https://www.anaconda.com/blog/using-pip-in-a-conda-environment)
as they might create hard to reproduce state and may break some things. This is
due a fact that conda cannot manage a packages installed via other package 
managers. 

**If You pick one, just stick with it to the end, do not merge or mix them.**

### PyTorch
Start from installing PyTorch:
```
$ conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```
This will install PyTorch with CUDA 11.3 support 
for Ubuntu 20.04. If You do not have a graphics card with CUDA support or use
other operating system refer to [documentation](https://pytorch.org/get-started/locally/).
**Make sure that You pick correct system and correct CUDA support!**


### Open AI Gym
Simply run a command:
```
$ conda install -c conda-forge gym 
```
For details refer to [gym](https://gym.openai.com/docs/) or 
[conda gym installation documentation](https://anaconda.org/conda-forge/gym)


### Stable Baselines3 
Simply run a command:
```
$ conda install -c conda-forge stable-baselines3 
```
For details refer to [documentation](https://stable-baselines3.readthedocs.io/en/master/guide/install.html)
or [conda sb3 installation documentation](https://anaconda.org/conda-forge/stable-baselines3)


### Fixing missing dependencies
At this point, theoretically environment should be ready to go, but there are 
still some [dependencies missing](https://stackoverflow.com/questions/44198228/)
(at least for LunarLander but maybe for more envs) so install them with the
following commands:
```
$ conda install -c anaconda swig
$ conda install -c conda-forge gym-box2d
```
Now, there should be no errors when running examples.

## Relevant examples and experimenting scenarios
- [LunarLander-v2](https://gym.openai.com/envs/LunarLander-v2/) - one of examples provided in **Materials** section uses that scenario to explain basic RL concepts and training strategies
- [CarRacing-v0](https://gym.openai.com/envs/CarRacing-v0/) - the closest enviroment to the task of the project
- [Enduro-v0](https://gym.openai.com/envs/Enduro-v0/) - Atari race game, inputs available as images or RAM state of Atari (128B)
- [Riverraid-v0](https://gym.openai.com/envs/Riverraid-v0/) - Atari boat swimming game, inputs as images or RAM state

