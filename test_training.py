# test_training.py

from src.environment import BeeSwarmEnv
from src.agent import train_agent
import config

def main():
    env = BeeSwarmEnv()
    model = train_agent(env, total_timesteps=1000, device='cuda')  # Utilisez un nombre réduit pour le test

if __name__ == "__main__":
    main()
