# src/game_data.py

import json
import os
import logging

logger = logging.getLogger('BeeSwarmAI')

# Exemple de structure de GAME_DATA
GAME_DATA = {
    'equipments': {},
    'consumables': {},
    'amulets': {},
    'bees': {}
}

def save_game_data(game_data, filepath=None):
    """
    Sauvegarde les données du jeu dans un fichier JSON.
    """
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), '..', 'data', 'game_data.json')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(game_data, f, indent=4)
        logger.info(f"Données du jeu sauvegardées dans {filepath}.")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde des données du jeu : {e}")

def load_game_data(filepath=None):
    """
    Charge les données du jeu depuis un fichier JSON.
    """
    global GAME_DATA
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), '..', 'data', 'game_data.json')
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                GAME_DATA = json.load(f)
            logger.info(f"Données du jeu chargées depuis {filepath}.")
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données du jeu : {e}")
    else:
        logger.warning(f"Fichier de données du jeu {filepath} non trouvé. Utilisation des données par défaut.")
