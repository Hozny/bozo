import faulthandler
import gym

import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from environment import Voronoi2DEnv

import sys
import numpy as np


def trace(frame, event, arg):
    print("%s, %s:%d" % (event, frame.f_code.co_filename, frame.f_lineno))
    return trace
# sys.settrace(trace)


faulthandler.enable()

env = Voronoi2DEnv(1, 1, 20)
print("YOOOOOOOO")
print(stable_baselines3.common.vec_env.util.obs_space_info(env.observation_space))

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
steps = 10000

# Create an empty list to store the rewards of all the states
rewards_list = []

for i in range(steps):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    rewards_list.append(rewards)

# Get the indices of the 20 states with the highest reward
top_indices = np.argsort(rewards_list)[-20:]

for i in range(steps):
    if i in top_indices:
        obs = env.reset()
        for j in range(i):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
        env.render()
