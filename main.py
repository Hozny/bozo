import gym

import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from environment import Voronoi2DEnv

import sys
def trace(frame, event, arg):
    print("%s, %s:%d" % (event, frame.f_code.co_filename, frame.f_lineno))
    return trace
#sys.settrace(trace)

import faulthandler
faulthandler.enable()

env = Voronoi2DEnv(100, 100, 40)
print("YOOOOOOOO")
print(stable_baselines3.common.vec_env.util.obs_space_info(env.observation_space))

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
for i in range(10000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print("EYEYEYE: ", i)
    env.render()
