# test_environment.py

from src.environment import BeeSwarmEnv

env = BeeSwarmEnv()
obs, _ = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
print(f"Observation: {obs.shape}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
