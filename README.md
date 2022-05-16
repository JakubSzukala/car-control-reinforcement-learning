# Car control with reinforcement learning 

## Materials
#### Some relevant materials on reinforcement learning
- [PyTorch](https://pytorch.org/)
- [Stable baselines3 - like sklearn but for RL, lot of abstraction, backend in PyTorch](https://github.com/DLR-RM/stable-baselines3)
- [Open AI Gym - Toolkit for reinforcement learning](https://gym.openai.com/)
- [SB3 Zoo - Alternative to Open AI Gym speciffically for SB3](https://github.com/DLR-RM/rl-baselines3-zoo)
- [Intro to RL and Stable Baselines + AI Gym](https://www.youtube.com/watch?v=XbWhJdQgi7E&list=PLQVvvaa0QuDf0O2DWwLZBfJeYY-JOeZB1)

#### Materials from supervisor
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

##### PyTorch
Start from installing PyTorch:
```
$ conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```
This will install PyTorch with CUDA 11.3 support 
for Ubuntu 20.04. If You do not have a graphics card with CUDA support or use
other operating system refer to [documentation](https://pytorch.org/get-started/locally/).
**Make sure that You pick correct system and correct CUDA support!**


##### Open AI Gym
Simply run a command:
```
$ conda install -c conda-forge gym 
```
For details refer to [gym](https://gym.openai.com/docs/) or 
[conda gym installation documentation](https://anaconda.org/conda-forge/gym)


##### Stable Baselines3 
Simply run a command:
```
$ conda install -c conda-forge stable-baselines3 
```
For details refer to [documentation](https://stable-baselines3.readthedocs.io/en/master/guide/install.html)
or [conda sb3 installation documentation](https://anaconda.org/conda-forge/stable-baselines3)


##### Fixing missing dependencies
At this point, theoretically environment should be ready to go, but there are 
still some [dependencies missing](https://stackoverflow.com/questions/44198228/)
(at least for LunarLander but maybe for more envs) so install them with the
following commands:
```
$ conda install -c anaconda swig
$ conda install -c conda-forge gym-box2d
$ conda install -c cogsci pygame 
```
Now, there should be no errors when running examples. **Check if You have the 
version 0.21 so that we have newest version possible and all of us are on the 
same page:**
```
$ conda list | grep gym
```
Check that bcs I got it downgraded somehow during the installation process to 
0.19. Not sure why and at what point, but unninstalling and rerunning 
installation commands fixed it. In case You have similar problem, run:
```
$ conda remove gym
$ conda install -c conda-forge gym 
$ conda install -c conda-forge gym-box2d
```

## TODO: when all working, compile into one convinient script

### Relevant examples and experimenting scenarios
- [LunarLander-v2](https://gym.openai.com/envs/LunarLander-v2/) - one of examples provided in **Materials** section uses that scenario to explain basic RL concepts and training strategies
- [CarRacing-v0](https://gym.openai.com/envs/CarRacing-v0/) - the closest enviroment to the task of the project
- [Enduro-v0](https://gym.openai.com/envs/Enduro-v0/) - Atari race game, inputs available as images or RAM state of Atari (128B)
- [Riverraid-v0](https://gym.openai.com/envs/Riverraid-v0/) - Atari boat swimming game, inputs as images or RAM state

### Initial ideas to make the project more intresting and similar to autonomous car
- We were thinking about creating a proceduraly generated map of simple road 
net, with crossroads traffic lights maybe some car traffic
- Procedural generation of road system may be a separate project itself so maybe
it is a little overkill, but inspiration for that was *relative* simplicity of
a car racing env
- But maybe taking a 'satelite' images of some cities and simplifying them to 
feed them into an algorithm could be feasible. Simplification probably could be
quite straight forward with OpenCV. 

## Algorithm development

In this chapter we will describe the development of the application. We will use
mentioned sources but mostly we will use a [official PyTorch tutorial on RL](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html).

#### Input extraction

This chapter follows the chapter from [official PyTorch tutorial on RL](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
called exactly the same. We will try to extract the car from the image in 
similar way the authors of original Pytorch tutorial did. For comparrission 
original image of the cart pole and extracted image:

![original](img/cart_pole.png)
![extracted](img/cart_pole_extracted.png)

From analysing the code we can see that we basically extract ROI, removing
unnecessary parts of the image and centering the cart in extracted image. After
that we convert to float, normalize, rescale and convert to tensor.

In [one of the examples](https://github.com/andywu0913/OpenAI-GYM-CarRacing-DQN/blob/master/common_functions.py)
speciffically for the CarRacing the image extraction is even simpler and uses 
cv2 for processing:
```python
def process_state_image(state):
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    state = state.astype(float)
    state /= 255.0
    return state
```
Probably some middle ground will be the best choice, as simplification of the 
input in official pytorch tutorial seems very appealing.




