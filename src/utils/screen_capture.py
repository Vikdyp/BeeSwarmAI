# src/utils/screen_capture.py

import numpy as np
from PIL import ImageGrab
import config
import logging

logger = logging.getLogger('BeeSwarmAI')

def capture_screen():
    """
    Capture une partie de l'écran définie par la résolution dans config.py.
    Retourne une image sous forme de tableau NumPy.
    """
    bbox = (0, 0, config.SCREEN_WIDTH, config.SCREEN_HEIGHT)  # Modifier si nécessaire
    try:
        img = ImageGrab.grab(bbox)
        img_np = np.array(img)
        # Convertir l'image de RGB à BGR si nécessaire
        img_np = img_np[:, :, ::-1]
        # Normaliser les pixels entre 0 et 1
        img_np = img_np.astype(np.float32) / 255.0
        # Transposer pour obtenir (C, H, W)
        img_np = img_np.transpose(2, 0, 1)
        return img_np
    except Exception as e:
        logger.error(f"Erreur lors de la capture de l'écran : {e}")
        return np.zeros((3, config.SCREEN_HEIGHT, config.SCREEN_WIDTH), dtype=np.float32)  # Retourner une image noire en cas d'erreur
