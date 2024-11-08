# src/agent.py

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from src.logger import logger
import os
import config

class SimpleCallback(BaseCallback):
    """
    Callback simple pour ajouter des logs personnalisés.
    """
    def __init__(self, verbose=0):
        super(SimpleCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        if self.n_calls % 100 == 0:
            logger.info(f"Étape d'entraînement: {self.n_calls}")
        return True

def train_agent(total_timesteps=5000, device='cpu'):
    """
    Entraîne un modèle PPO.
    """
    logger.info("Début de l'entraînement du modèle PPO.")
    
    # Configurer le répertoire de logs TensorBoard
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'tensorboard_logs')
    os.makedirs(log_dir, exist_ok=True)
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    
    # Initialiser le modèle PPO avec les paramètres du config
    env = PPO.env  # Assurez-vous que l'environnement est correctement passé ou accessible
    model = PPO(
        policy=config.POLICY_TYPE,  # e.g., "CnnPolicy"
        env=env,
        learning_rate=config.LEARNING_RATE,
        batch_size=config.BATCH_SIZE,
        verbose=1,  # Activer les logs
        policy_kwargs={"normalize_images": False},
        tensorboard_log=log_dir,
        device=device  # Spécifier l'appareil ("cuda" ou "cpu")
    )
    model.set_logger(new_logger)
    
    # Initialiser le callback simple
    simple_callback = SimpleCallback()
    
    try:
        logger.info(f"Entraînement en cours pour {total_timesteps} pas de temps.")
        model.learn(total_timesteps=total_timesteps, callback=simple_callback, log_interval=10)
        logger.info("Entraînement terminé avec succès.")
    except Exception as e:
        logger.error(f"Erreur durant l'entraînement : {e}")
    
    # Sauvegarder le modèle
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'ppo_bee_swarm')
    model.save(model_path)
    logger.info(f"Modèle sauvegardé à {model_path}.")

    return model

def test_agent(model, env):
    """
    Teste l'agent entraîné sur l'environnement donné.
    """
    logger.info("Début des tests du modèle.")
    try:
        obs, _ = env.reset()
        terminated = False
        truncated = False
        step = 0
        max_steps = 1000
        
        while not (terminated or truncated) and step < max_steps:
            action, _states = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1

            if step % 100 == 0:
                logger.info(f"Test - Action: {action}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
    except Exception as e:
        logger.error(f"Erreur durant le test de l'agent : {e}")
    finally:
        logger.info("Test terminé.")
