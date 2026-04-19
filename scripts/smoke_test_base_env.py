"""
A simple smoke test to check that the Tetris environment can be created and stepped through without errors.
"""
import gymnasium as gym
from tetris_gymnasium.envs.tetris import Tetris

env = gym.make("tetris_gymnasium/Tetris")
obs, info = env.reset(seed=42)

print("Action space:", env.action_space)
print("Observation space:", env.observation_space)

for step in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(step, reward, terminated, truncated)
    if terminated or truncated:
        obs, info = env.reset()

env.close()