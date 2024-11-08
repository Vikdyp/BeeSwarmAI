# test_agent.py

from src.environment import BeeSwarmEnv
from src.agent import train_agent, test_agent
from src.logger import logger
import config
import os

def test_agent_functions():
    env = BeeSwarmEnv()
    try:
        logger.info("Démarrage du test des fonctions d'agent.")
        
        # Entraînement avec un nombre réduit de timesteps pour le test
        test_timesteps = 1000
        logger.info(f"Entraînement de l'agent avec {test_timesteps} timesteps.")
        agent = train_agent(env, total_timesteps=test_timesteps)
        logger.info("Entraînement de l'agent terminé.")
        
        # Sauvegarder le modèle entraîné
        model_path = "models/ppo_bee_swarm_test"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        agent.save(model_path)
        logger.info(f"Modèle d'agent de test sauvegardé à {model_path}.")
        
        # Tester l'agent entraîné
        logger.info("Début des tests de l'agent entraîné.")
        test_agent(agent, env)
        logger.info("Tests de l'agent entraîné terminés.")
    
    except Exception as e:
        logger.error(f"Erreur lors du test des fonctions d'agent: {e}")
    finally:
        env.close()
        logger.info("Environnement fermé après le test des fonctions d'agent.")
        
if __name__ == "__main__":
    test_agent_functions()
