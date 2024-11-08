# config.py

import os
import json
from src.game_data import GAME_DATA

# Configuration pour l'entraînement
TOTAL_TIMESTEPS = 5000
LEARNING_RATE = 0.0003
BATCH_SIZE = 64
POLICY_TYPE = "CnnPolicy"  # Options possibles : "CnnPolicy", "MlpPolicy", etc.

# Configuration des captures
SCREEN_WIDTH = 1920  # Modifier selon votre résolution
SCREEN_HEIGHT = 1080  # Modifier selon votre résolution
CAPTURE_FPS = 10

# Mapping des actions
ACTIONS = {
    0: {'type': 'key', 'key': 'a'},
    1: {'type': 'key', 'key': 'b'},
    2: {'type': 'key', 'key': 'c'},
    3: {'type': 'key', 'key': 'd'},
    4: {'type': 'key', 'key': 'e'},
    5: {'type': 'key', 'key': 'f'},
    6: {'type': 'key', 'key': 'g'},
    7: {'type': 'key', 'key': 'h'},
    8: {'type': 'key', 'key': 'i'},
    9: {'type': 'key', 'key': 'j'},
    10: {'type': 'key', 'key': 'k'},
    11: {'type': 'key', 'key': 'l'},
    12: {'type': 'key', 'key': 'm'},
    13: {'type': 'key', 'key': 'n'},
    14: {'type': 'key', 'key': 'o'},
    15: {'type': 'key', 'key': 'p'},
    16: {'type': 'key', 'key': 'q'},
    17: {'type': 'key', 'key': 'r'},
    18: {'type': 'key', 'key': 's'},
    19: {'type': 'key', 'key': 't'},
    20: {'type': 'key', 'key': 'u'},
    21: {'type': 'key', 'key': 'v'},
    22: {'type': 'key', 'key': 'w'},
    23: {'type': 'key', 'key': 'x'},
    24: {'type': 'key', 'key': 'y'},
    25: {'type': 'key', 'key': 'z'},
    26: {'type': 'key', 'key': '1'},
    27: {'type': 'key', 'key': '2'},
    28: {'type': 'key', 'key': '3'},
    29: {'type': 'key', 'key': '4'},
    30: {'type': 'key', 'key': '5'},
    31: {'type': 'key', 'key': '6'},
    32: {'type': 'key', 'key': '7'},
    33: {'type': 'key', 'key': 'shift'},
    34: {'type': 'key', 'key': 'space'},
    35: {'type': 'mouse', 'action': 'left_click'},
    36: {'type': 'mouse', 'action': 'right_click'},
    37: {'type': 'mouse', 'action': 'move_up'},
    38: {'type': 'mouse', 'action': 'move_down'},
    39: {'type': 'mouse', 'action': 'move_left'},
    40: {'type': 'mouse', 'action': 'move_right'},
    # Ajouter d'autres actions si nécessaire
}

# Zones d'observation (initialisées avec des valeurs par défaut)
POLLEN_ZONE = (100, 100, 200, 200)  # Exemple de coordonnées, à ajuster
HONEY_ZONE = (300, 300, 400, 400)   # Exemple de coordonnées, à ajuster
