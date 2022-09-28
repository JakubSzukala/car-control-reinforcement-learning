import gym

from collections import namedtuple
from datetime import datetime
from os.path import join

from stable_baselines3.a2c.a2c import A2C
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.callbacks import EvalCallback

from car_racing_env_wrapper import CarRacingDistanceStateWrapper
root_env = gym.make('CarRacing-v2', continuous=False)
env = CarRacingDistanceStateWrapper(root_env)

# Parameters
algorithm = "PPO"
policy = 'MlpPolicy'

# Get time stamp for output filename
now = datetime.now()
date_string = now.strftime("%d-%m-%Y-%H-%M-%S")

filename_best = algorithm + "-" + policy + "-" + date_string + "_best"

# Periodically evaluate and save the best model
eval_callback = EvalCallback(env,
        best_model_save_path=join('logs', filename_best),
        log_path=join('logs', filename_best),
        eval_freq=750, deterministic=True, render=False)

# Model
model = eval(algorithm)(policy, env, verbose=1)
model.learn(total_timesteps=500_000, callback=eval_callback)

# Save final trained model
filename_final = algorithm + "-" + policy + "-" + date_string + "_final"
model.save(join('logs', filename_final))

input("##############################\nTraining finished. Press a key to evaluate the model...\n##############################")

models = []
Model = namedtuple("Model",("model", "model_name"))
models.append(Model(eval(algorithm).load(join('logs', filename_final)), 'model_final'))
models.append(Model(eval(algorithm).load(join('logs', filename_best)), 'model_best'))

for model in models:
    for i in range(1000):
        print("Current model: ".format(model.model_name))
        action, _state = model.predict(obs, deterministic=True) # type: ignore
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
          obs = env.reset()
