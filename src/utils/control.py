# src/utils/control.py

import pyautogui
import time
import logging
from src.logger import logger

def execute_action(action):
    """
    Exécute une action basée sur le mapping défini dans config.ACTIONS.
    """
    import config  # Importer ici pour éviter les problèmes de circularité

    action_detail = config.ACTIONS.get(action)
    if not action_detail:
        logger.error(f"Action inconnue : {action}")
        return

    action_type = action_detail['type']

    try:
        if action_type == 'key':
            key = action_detail['key']
            pyautogui.press(key)
            logger.debug(f"Action clé exécutée : {key}")
        elif action_type == 'mouse':
            mouse_action = action_detail['action']
            if mouse_action == 'left_click':
                pyautogui.click(button='left')
                logger.debug("Action souris : clic gauche")
            elif mouse_action == 'right_click':
                pyautogui.click(button='right')
                logger.debug("Action souris : clic droit")
            elif mouse_action == 'move_up':
                pyautogui.moveRel(0, -10, duration=0.1)
                logger.debug("Action souris : déplacer vers le haut")
            elif mouse_action == 'move_down':
                pyautogui.moveRel(0, 10, duration=0.1)
                logger.debug("Action souris : déplacer vers le bas")
            elif mouse_action == 'move_left':
                pyautogui.moveRel(-10, 0, duration=0.1)
                logger.debug("Action souris : déplacer vers la gauche")
            elif mouse_action == 'move_right':
                pyautogui.moveRel(10, 0, duration=0.1)
                logger.debug("Action souris : déplacer vers la droite")
            else:
                logger.warning(f"Action souris inconnue : {mouse_action}")
        else:
            logger.warning(f"Type d'action inconnu : {action_type}")
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution de l'action {action} : {e}")

def focus_game_window():
    """
    Met le jeu en focus. Cette fonction doit être adaptée en fonction du jeu.
    Par exemple, vous pouvez utiliser des bibliothèques comme pygetwindow pour focaliser la fenêtre du jeu.
    """
    try:
        import pygetwindow as gw
        # Remplacer 'NomDuJeu' par le titre exact de la fenêtre du jeu
        windows = gw.getWindowsWithTitle('Roblox')
        if windows:
            game_window = windows[0]
            game_window.activate()
            logger.debug("Fenêtre du jeu mise au focus.")
            time.sleep(0.5)  # Attendre que la fenêtre soit active
        else:
            logger.warning("Fenêtre du jeu non trouvée.")
    except Exception as e:
        logger.error(f"Erreur lors de la mise au focus de la fenêtre du jeu : {e}")
