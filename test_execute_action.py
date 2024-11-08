# test_execute_action.py

from src.utils.control import execute_action
from src.logger import logger

def test_execute_action():
    logger.info("Début du test de execute_action.")
    execute_action(4)  # Test de l'action 'collecter' (space)
    logger.info("Test de execute_action terminé.")

if __name__ == "__main__":
    test_execute_action()
