import gym

env = gym.make("LunarLander-v2")
env.reset()
for i in range(0, 100):
    env.render()
    env.step(env.action_space.sample())

    print('Sample action: ', env.action_space.sample())
    print('Observation space shape: ', env.observation_space.shape)
    print('Sample observation: ', env.observation_space.sample())

env.close()
