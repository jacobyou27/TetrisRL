"""
This script is for inspecting the observation and info returned by the Tetris environment.
"""
import gymnasium as gym
from tetris_gymnasium.envs.tetris import Tetris

env = gym.make("tetris_gymnasium/Tetris")
obs, info = env.reset(seed=42)

print("obs type:", type(obs))

if hasattr(obs, "shape"):
    print("obs shape:", obs.shape)

print("obs:")
print(obs)

print("info:")
print(info)

env.close()