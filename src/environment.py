# src/environment.py

import gymnasium as gym
from gymnasium import spaces
from src.utils.screen_capture import capture_screen
from src.utils.control import execute_action, focus_game_window
from src.game_data import GAME_DATA, save_game_data, load_game_data
import numpy as np
from src.logger import logger
import os
from datetime import datetime
import config
import pytesseract
from PIL import Image
import threading

class BeeSwarmEnv(gym.Env):
    def __init__(self):
        super(BeeSwarmEnv, self).__init__()
        self.action_space = spaces.Discrete(len(config.ACTIONS))  # Nombre total d'actions définies
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(3, config.SCREEN_HEIGHT, config.SCREEN_WIDTH), dtype=np.float32
        )
        self.step_count = 0  # Compteur de pas
        self.screenshot_dir = os.path.join(os.path.dirname(__file__), '..', 'screenshots')
        os.makedirs(self.screenshot_dir, exist_ok=True)  # Créer le répertoire des screenshots

        # Variables d'état pour le suivi des récompenses
        self.current_pollen = 0
        self.current_honey = 0
        self.previous_pollen = 0
        self.previous_honey = 0
        self.max_backpack_capacity = 100  # Exemple de capacité max, ajuster selon le jeu
        self.is_alive = True  # Statut de l'agent
        self.max_steps = 10000  # Nombre maximum de pas par épisode

        # Zones d'observation (initialisées à partir du config)
        self.pollen_zone = config.POLLEN_ZONE
        self.honey_zone = config.HONEY_ZONE

        # Charger les données du jeu
        load_game_data()

        # Lock pour les opérations thread-safe
        self.lock = threading.Lock()

    def reset(self, **kwargs):
        with self.lock:
            # Réinitialisez le jeu et capturez l'état initial
            observation = capture_screen()
            logger.info("Environnement réinitialisé.")
            self.save_screenshot("reset")

            # Réinitialiser les variables d'état
            self.current_pollen = self.extract_pollen(observation)
            self.current_honey = self.extract_honey(observation)
            self.previous_pollen = self.current_pollen
            self.previous_honey = self.current_honey
            self.step_count = 0
            self.is_alive = True

            return observation, {}

    def step(self, action):
        with self.lock:
            if not self.is_alive:
                # Si l'agent est mort, terminer l'épisode
                return self._get_observation(), -100, True, False, {}

            # Assurer que la fenêtre de jeu est en focus avant d'envoyer l'action
            focus_game_window()

            # Envoyer l'action au jeu
            execute_action(action)
            self.step_count += 1

            # Capturer la nouvelle observation
            observation = capture_screen()

            # Extraire les nouvelles valeurs de pollen et de miel
            self.previous_pollen = self.current_pollen
            self.previous_honey = self.current_honey
            self.current_pollen = self.extract_pollen(observation)
            self.current_honey = self.extract_honey(observation)

            # Calculer la récompense
            reward = self.calculer_recompense()

            # Vérifier si l'agent est mort ou si le sac à dos est saturé
            terminated, truncated = self.verifier_si_fini(observation)

            # Terminer l'épisode si la limite de pas est atteinte
            if self.step_count >= self.max_steps:
                truncated = True
                logger.info("Limite de pas atteinte, terminant l'épisode.")

            # Imprimer et enregistrer des captures tous les 100 pas
            if self.step_count % 100 == 0:
                action_detail = config.ACTIONS[action]
                action_type = action_detail['type']
                action_key_or_mouse = action_detail.get('key', action_detail.get('action', ''))
                logger.info(f"Action exécutée : {action} ({action_type} - {action_key_or_mouse})")
                logger.info(f"Pollen: {self.current_pollen}, Honey: {self.current_honey}")
                self.save_screenshot(f"step_{self.step_count}")

            return observation, reward, terminated, truncated, {}

    def calculer_recompense(self):
        reward = 0

        # Récompense pour la récolte de pollen
        pollen_collected = self.current_pollen - self.previous_pollen
        if pollen_collected > 0:
            reward += 1 * pollen_collected  # Ajuster le facteur de récompense
            logger.debug(f"Pollen collecté: {pollen_collected}, Récompense: {1 * pollen_collected}")

        # Récompense pour la conversion de pollen en miel
        honey_produced = self.current_honey - self.previous_honey
        if honey_produced > 0:
            reward += 5 * honey_produced  # Récompense plus élevée pour le miel
            logger.debug(f"Miel produit: {honey_produced}, Récompense: {5 * honey_produced}")

        # Récompense pour la gestion des abeilles
        # Implémenter la détection et la récompense pour la gestion des abeilles

        # Récompense pour l'achat d'équipements
        # Implémenter la détection et la récompense pour l'achat d'équipements

        # Récompense pour l'utilisation de buffs et amulettes
        # Implémenter la détection et la récompense pour l'utilisation des buffs et amulettes

        # Pénalité pour la saturation du sac à dos
        if self.current_pollen >= self.max_backpack_capacity:
            reward -= 10  # Pénalité fixe
            logger.warning("Sac à dos saturé !")

        # Pénalité pour la mort de l'agent
        if not self.is_alive:
            reward -= 100  # Pénalité importante
            logger.warning("Agent est mort !")

        # Pénalité pour actions inefficaces
        if pollen_collected <= 0 and honey_produced <= 0:
            reward -= 1  # Pénalité pour action inefficace
            logger.debug("Action inefficace, pénalité de -1 appliquée.")

        return reward

    def verifier_si_fini(self, observation):
        terminated = False
        truncated = False

        # Détecter la mort de l'agent
        if self.detect_death(observation):
            terminated = True
            self.is_alive = False
            logger.info("Agent est mort.")

        # Détecter la saturation du sac à dos
        if self.current_pollen >= self.max_backpack_capacity:
            truncated = True
            logger.info("Sac à dos saturé, terminant l'épisode.")

        return terminated, truncated

    def detect_death(self, observation):
        """
        Implémentez une méthode pour détecter si l'agent est mort.
        Cela pourrait être basé sur la reconnaissance d'un certain motif ou couleur dans l'image.
        """
        # Exemple simplifié : vérifier si une certaine région de l'image est d'une couleur spécifique
        # À adapter selon le jeu
        death_region = observation[:, 0:100, 0:100]  # Région hypothétique
        # Supposons que la mort est indiquée par une zone rouge
        red_channel = death_region[0]
        if np.mean(red_channel) > 0.8:  # Seuil hypothétique
            return True
        return False

    def extract_pollen(self, observation):
        """
        Implémentez une méthode pour extraire la quantité de pollen de l'observation.
        Cela pourrait impliquer la reconnaissance de texte ou la détection de motifs.
        """
        if not self.pollen_zone:
            logger.error("Zone de pollen non définie.")
            return self.previous_pollen

        try:
            # Convertir l'observation en image PIL
            img = Image.fromarray((observation.transpose(1, 2, 0) * 255).astype(np.uint8))
            # Définir la région où le pollen est affiché
            pollen_region = img.crop(self.pollen_zone)  # (x1, y1, x2, y2)
            # Prétraitement pour améliorer la précision de l'OCR
            pollen_region = pollen_region.convert('L')  # Convertir en niveaux de gris
            pollen_region = pollen_region.point(lambda x: 0 if x < 140 else 255, '1')  # Seuil binaire
            pollen_text = pytesseract.image_to_string(pollen_region, config='--psm 6 digits')
            pollen = int(pollen_text.strip()) if pollen_text.strip().isdigit() else 0
            return pollen
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction du pollen : {e}")
            return self.previous_pollen  # Retourner la valeur précédente en cas d'erreur

    def extract_honey(self, observation):
        """
        Implémentez une méthode pour extraire la quantité de miel de l'observation.
        Cela pourrait impliquer la reconnaissance de texte ou la détection de motifs.
        """
        if not self.honey_zone:
            logger.error("Zone de miel non définie.")
            return self.previous_honey

        try:
            # Convertir l'observation en image PIL
            img = Image.fromarray((observation.transpose(1, 2, 0) * 255).astype(np.uint8))
            # Définir la région où le miel est affiché
            honey_region = img.crop(self.honey_zone)  # (x1, y1, x2, y2)
            # Prétraitement pour améliorer la précision de l'OCR
            honey_region = honey_region.convert('L')  # Convertir en niveaux de gris
            honey_region = honey_region.point(lambda x: 0 if x < 140 else 255, '1')  # Seuil binaire
            honey_text = pytesseract.image_to_string(honey_region, config='--psm 6 digits')
            honey = int(honey_text.strip()) if honey_text.strip().isdigit() else 0
            return honey
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction du miel : {e}")
            return self.previous_honey  # Retourner la valeur précédente en cas d'erreur

    def _get_observation(self):
        """
        Méthode auxiliaire pour obtenir l'observation actuelle.
        """
        return capture_screen()

    def save_screenshot(self, event):
        """
        Enregistre une capture d'écran avec un nom basé sur l'événement et le timestamp.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.screenshot_dir, f"{event}_{timestamp}.png")
        try:
            observation = capture_screen()
            # Convertir l'observation en image et sauvegarder
            img_array = observation.transpose(1, 2, 0) * 255.0
            img = Image.fromarray(img_array.astype(np.uint8))
            img.save(filename)
            logger.info(f"Capture d'écran enregistrée : {filename}")
        except Exception as e:
            logger.error(f"Erreur lors de l'enregistrement de la capture d'écran : {e}")

    def set_pollen_zone(self, zone):
        """
        Définit la zone de pollen.
        """
        self.pollen_zone = zone
        logger.info(f"Zone de pollen définie à : {zone}")

    def set_honey_zone(self, zone):
        """
        Définit la zone de miel.
        """
        self.honey_zone = zone
        logger.info(f"Zone de miel définie à : {zone}")  
