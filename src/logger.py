# src/logger.py

import logging
import os

# Créer le répertoire logs s'il n'existe pas
LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# Configuration du logger
logger = logging.getLogger('BeeSwarmAI')
logger.setLevel(logging.DEBUG)  # Niveau global

# Formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Handler pour le fichier de log
file_handler = logging.FileHandler(os.path.join(LOG_DIR, 'bee_swarm.log'))
file_handler.setLevel(logging.DEBUG)  # Niveau pour le fichier
file_handler.setFormatter(formatter)

# Handler pour la console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Niveau pour la console
console_handler.setFormatter(formatter)

# Ajouter les handlers au logger s'ils ne sont pas déjà ajoutés
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
