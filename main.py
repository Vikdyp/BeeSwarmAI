# main.py

import tkinter as tk
from tkinter import messagebox, simpledialog
from tkinter import ttk  # Importer ttk pour les onglets
from PIL import Image, ImageTk
from src.environment import BeeSwarmEnv
from src.agent import train_agent, test_agent
import config
from src.logger import logger
import os
import threading
from stable_baselines3 import PPO  # Assurez-vous que PPO est importé
import src.utils.screen_capture as screen_capture  # Importer capture_screen
from src.game_data import GAME_DATA, save_game_data, load_game_data
import json
import torch  # Importer PyTorch pour la vérification de CUDA
import logging

class TextHandler(logging.Handler):
    """
    Cette classe permet de rediriger les logs vers un widget Text de Tkinter.
    """
    def __init__(self, text_widget):
        logging.Handler.__init__(self)
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record)
        # Tkinter n'est pas thread-safe, utiliser after pour planifier l'ajout du log
        self.text_widget.after(0, self.append, msg)

    def append(self, msg):
        self.text_widget.configure(state='normal')
        self.text_widget.insert(tk.END, msg + '\n')
        self.text_widget.configure(state='disabled')
        self.text_widget.see(tk.END)

class BeeSwarmGUI:
    def __init__(self, master):
        self.master = master
        master.title("BeeSwarmAI Controller")
        master.geometry("1000x700")  # Augmenter la taille de la fenêtre pour plus de confort
        master.resizable(False, False)

        logger.info("Initialisation de BeeSwarmGUI")

        self.env = BeeSwarmEnv()
        self.training_thread = None
        self.training = False

        # Créer les onglets
        self.tab_control = ttk.Notebook(master)
        
        # Onglet Training
        self.training_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.training_tab, text='Training')
        
        # Onglet Settings
        self.settings_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.settings_tab, text='Settings')
        
        # Onglet Logs
        self.logs_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.logs_tab, text='Logs')
        
        # Onglet Screenshots
        self.screenshots_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.screenshots_tab, text='Screenshots')
        
        # Onglet Game Data Management
        self.game_data_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.game_data_tab, text='Game Data Management')
        
        self.tab_control.pack(expand=1, fill='both')

        logger.info("Création des onglets terminée")

        # --------------------
        # Contenu de l'Onglet Training
        # --------------------
        self.start_button = tk.Button(self.training_tab, text="Start Training", command=self.start_training, width=20, height=2)
        self.start_button.pack(pady=10)

        self.stop_button = tk.Button(self.training_tab, text="Stop Training", command=self.stop_training_func, width=20, height=2, state=tk.DISABLED)
        self.stop_button.pack(pady=10)

        self.test_button = tk.Button(self.training_tab, text="Test Agent", command=self.test_agent_func, width=20, height=2, state=tk.DISABLED)
        self.test_button.pack(pady=10)

        # Ajouter une barre de progression déterminée
        self.progress = ttk.Progressbar(self.training_tab, orient='horizontal', mode='determinate', maximum=config.TOTAL_TIMESTEPS)
        self.progress.pack(pady=10, fill='x', padx=20)

        # Ajouter un label pour la progression
        self.progress_label = tk.Label(self.training_tab, text="Progression: 0%")
        self.progress_label.pack(pady=5)

        # --------------------
        # Contenu de l'Onglet Settings
        # --------------------
        self.timesteps_label = tk.Label(self.settings_tab, text="Total Timesteps:")
        self.timesteps_label.grid(row=0, column=0, padx=10, pady=5, sticky='e')

        self.timesteps_entry = tk.Entry(self.settings_tab)
        self.timesteps_entry.insert(0, str(config.TOTAL_TIMESTEPS))
        self.timesteps_entry.grid(row=0, column=1, padx=10, pady=5)

        self.learning_rate_label = tk.Label(self.settings_tab, text="Learning Rate:")
        self.learning_rate_label.grid(row=1, column=0, padx=10, pady=5, sticky='e')

        self.learning_rate_entry = tk.Entry(self.settings_tab)
        self.learning_rate_entry.insert(0, str(config.LEARNING_RATE))
        self.learning_rate_entry.grid(row=1, column=1, padx=10, pady=5)

        self.batch_size_label = tk.Label(self.settings_tab, text="Batch Size:")
        self.batch_size_label.grid(row=2, column=0, padx=10, pady=5, sticky='e')

        self.batch_size_entry = tk.Entry(self.settings_tab)
        self.batch_size_entry.insert(0, str(config.BATCH_SIZE))
        self.batch_size_entry.grid(row=2, column=1, padx=10, pady=5)

        self.policy_type_label = tk.Label(self.settings_tab, text="Policy Type:")
        self.policy_type_label.grid(row=3, column=0, padx=10, pady=5, sticky='e')

        self.policy_type_options = ["CnnPolicy", "MlpPolicy"]
        self.policy_type_var = tk.StringVar()
        self.policy_type_var.set(config.POLICY_TYPE)
        self.policy_type_menu = tk.OptionMenu(self.settings_tab, self.policy_type_var, *self.policy_type_options)
        self.policy_type_menu.grid(row=3, column=1, padx=10, pady=5, sticky='w')

        # Bouton pour appliquer les paramètres
        self.apply_button = tk.Button(self.settings_tab, text="Apply Settings", command=self.apply_settings, width=20)
        self.apply_button.grid(row=4, column=0, columnspan=2, pady=10)

        # Configuration de la grille pour l'onglet Settings
        for i in range(4):
            self.settings_tab.grid_rowconfigure(i, weight=1)
        self.settings_tab.grid_columnconfigure(0, weight=1)
        self.settings_tab.grid_columnconfigure(1, weight=1)

        # --------------------
        # Contenu de l'Onglet Logs
        # --------------------
        self.log_text = tk.Text(self.logs_tab, state='disabled', wrap='word')
        self.log_text.pack(expand=1, fill='both', padx=10, pady=10)

        # Configurer le logger pour ajouter le TextHandler
        log_handler = TextHandler(self.log_text)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        log_handler.setFormatter(formatter)
        logger.addHandler(log_handler)

        # --------------------
        # Contenu de l'Onglet Screenshots
        # --------------------
        self.screenshot_button = tk.Button(self.screenshots_tab, text="Take Screenshot", command=self.take_screenshot, width=20, height=2)
        self.screenshot_button.pack(pady=10)

        self.screenshot_label = tk.Label(self.screenshots_tab)
        self.screenshot_label.pack(pady=10)

        # --------------------
        # Contenu de l'Onglet Game Data Management
        # --------------------
        self.game_data_frame = ttk.Frame(self.game_data_tab)
        self.game_data_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Sections pour chaque type de game data
        self.create_game_data_section('Equipments', 'equipments')
        self.create_game_data_section('Consumables', 'consumables')
        self.create_game_data_section('Amulets', 'amulets')
        self.create_game_data_section('Bees', 'bees')

        # Boutons pour définir les zones de pollen et de miel
        self.zone_frame = ttk.LabelFrame(self.game_data_tab, text="Délimitation des Zones d'Observation")
        self.zone_frame.pack(fill='x', padx=10, pady=10)

        self.set_pollen_zone_button = tk.Button(self.zone_frame, text="Set Pollen Zone", command=lambda: self.set_zone('pollen'))
        self.set_pollen_zone_button.pack(side='left', padx=10, pady=10)

        self.set_honey_zone_button = tk.Button(self.zone_frame, text="Set Honey Zone", command=lambda: self.set_zone('honey'))
        self.set_honey_zone_button.pack(side='left', padx=10, pady=10)

        # Charger les zones depuis le fichier de configuration
        self.load_zones()

        # Vérifier CUDA au démarrage
        self.check_cuda()

        logger.info("BeeSwarmGUI initialisé avec succès")

    def create_game_data_section(self, title, category):
        """
        Crée une section pour gérer un type de game data.
        """
        frame = ttk.LabelFrame(self.game_data_frame, text=title)
        frame.pack(fill='both', expand=True, padx=5, pady=5)

        # Liste des items
        listbox = tk.Listbox(frame, height=6)
        listbox.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        listbox.bind('<<ListboxSelect>>', lambda e, c=category, lb=listbox: self.on_select(c, lb))

        # Boutons de gestion
        button_frame = ttk.Frame(frame)
        button_frame.pack(side='right', fill='y', padx=5, pady=5)

        add_button = tk.Button(button_frame, text="Add", command=lambda c=category: self.add_item(c))
        add_button.pack(fill='x', pady=2)

        edit_button = tk.Button(button_frame, text="Edit", command=lambda c=category, lb=listbox: self.edit_item(c, lb))
        edit_button.pack(fill='x', pady=2)

        delete_button = tk.Button(button_frame, text="Delete", command=lambda c=category, lb=listbox: self.delete_item(c, lb))
        delete_button.pack(fill='x', pady=2)

        # Remplir la liste
        self.populate_listbox(category, listbox)

    def populate_listbox(self, category, listbox):
        """
        Remplit le listbox avec les items de la catégorie donnée.
        """
        listbox.delete(0, tk.END)
        for key, item in GAME_DATA.get(category, {}).items():
            listbox.insert(tk.END, item.get('name', key))

    def on_select(self, category, listbox):
        """
        Gère la sélection d'un item dans le listbox.
        """
        selected_indices = listbox.curselection()
        if not selected_indices:
            return
        index = selected_indices[0]
        item_name = listbox.get(index)
        # Afficher les détails si nécessaire
        # Pour simplifier, on ne fait rien ici
        pass

    def add_item(self, category):
        """
        Ajoute un nouvel item à la catégorie donnée.
        """
        item_data = {}
        fields = self.get_fields_for_category(category)
        for field in fields:
            value = simpledialog.askstring("Input", f"Entrez {field.replace('_', ' ').capitalize()}:")
            if value is None:
                return  # Annulé
            try:
                if field in ['pollen_multiplier', 'honey_multiplier', 'value']:
                    value = float(value)
                elif field in ['cost', 'duration', 'pollen_rate', 'honey_rate']:
                    value = int(value)
            except ValueError:
                messagebox.showerror("Erreur", f"Valeur invalide pour {field.replace('_', ' ').capitalize()}.")
                return
            item_data[field] = value

        # Générer une clé unique
        key = self.generate_unique_key(category)
        GAME_DATA[category][key] = item_data
        GAME_DATA[category][key]['name'] = item_data.get('name', f"New {category[:-1].capitalize()}")

        save_game_data(GAME_DATA)
        logger.info(f"Ajouté {item_data['name']} à {category}.")

        # Rafraîchir les listes
        self.refresh_all_listboxes()

    def edit_item(self, category, listbox):
        """
        Modifie l'item sélectionné dans la catégorie donnée.
        """
        selected_indices = listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Avertissement", "Veuillez sélectionner un élément à modifier.")
            return
        index = selected_indices[0]
        item_name = listbox.get(index)

        # Trouver la clé correspondante
        key = None
        for k, v in GAME_DATA[category].items():
            if v.get('name') == item_name:
                key = k
                break
        if not key:
            messagebox.showerror("Erreur", "Élément non trouvé.")
            return

        item = GAME_DATA[category][key]
        fields = self.get_fields_for_category(category)
        for field in fields:
            current_value = item.get(field, '')
            new_value = simpledialog.askstring("Input", f"Entrez {field.replace('_', ' ').capitalize()} (Actuel: {current_value}):")
            if new_value is None:
                return  # Annulé
            try:
                if field in ['pollen_multiplier', 'honey_multiplier', 'value']:
                    new_value = float(new_value)
                elif field in ['cost', 'duration', 'pollen_rate', 'honey_rate']:
                    new_value = int(new_value)
            except ValueError:
                messagebox.showerror("Erreur", f"Valeur invalide pour {field.replace('_', ' ').capitalize()}.")
                return
            item[field] = new_value

        GAME_DATA[category][key] = item
        save_game_data(GAME_DATA)
        logger.info(f"Modifié {item['name']} dans {category}.")

        # Rafraîchir les listes
        self.refresh_all_listboxes()

    def delete_item(self, category, listbox):
        """
        Supprime l'item sélectionné dans la catégorie donnée.
        """
        selected_indices = listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Avertissement", "Veuillez sélectionner un élément à supprimer.")
            return
        index = selected_indices[0]
        item_name = listbox.get(index)

        # Confirmer la suppression
        if not messagebox.askyesno("Confirmation", f"Voulez-vous vraiment supprimer {item_name} de {category} ?"):
            return

        # Trouver la clé correspondante
        key = None
        for k, v in GAME_DATA[category].items():
            if v.get('name') == item_name:
                key = k
                break
        if not key:
            messagebox.showerror("Erreur", "Élément non trouvé.")
            return

        del GAME_DATA[category][key]
        save_game_data(GAME_DATA)
        logger.info(f"Supprimé {item_name} de {category}.")

        # Rafraîchir les listes
        self.refresh_all_listboxes()

    def get_fields_for_category(self, category):
        """
        Retourne les champs nécessaires pour une catégorie donnée.
        """
        if category == 'equipments':
            return ['name', 'type', 'pollen_multiplier', 'honey_multiplier', 'cost']
        elif category == 'consumables':
            return ['name', 'type', 'effect', 'value', 'duration', 'cost']
        elif category == 'amulets':
            return ['name', 'type', 'effect', 'value', 'cost']
        elif category == 'bees':
            return ['name', 'type', 'pollen_rate', 'honey_rate', 'cost']
        else:
            return []

    def generate_unique_key(self, category):
        """
        Génère une clé unique pour un nouvel item dans la catégorie donnée.
        """
        existing_keys = GAME_DATA[category].keys()
        index = 1
        while True:
            key = f"{category[:-1]}_{index}"
            if key not in existing_keys:
                return key
            index += 1

    def refresh_all_listboxes(self):
        """
        Rafraîchit toutes les listes des onglets de gestion des game data.
        """
        for child in self.game_data_frame.winfo_children():
            if isinstance(child, ttk.LabelFrame):
                category = child.cget("text").lower().replace(' ', '')
                listbox = child.winfo_children()[0]
                self.populate_listbox(category, listbox)

    def set_zone(self, zone_type):
        """
        Permet à l'utilisateur de délimiter une zone sur l'écran.
        """
        messagebox.showinfo("Définir la Zone", f"Après avoir cliqué sur OK, sélectionnez la zone pour {zone_type} en cliquant et en faisant glisser la souris.")
        self.master.withdraw()  # Masquer la fenêtre principale

        # Créer une fenêtre transparente pour la sélection
        selection_window = tk.Toplevel()
        selection_window.attributes('-alpha', 0.3)  # Transparence
        selection_window.attributes('-fullscreen', True)
        selection_window.attributes('-topmost', True)
        selection_window.configure(bg='gray')

        canvas = tk.Canvas(selection_window, cursor="cross", bg='grey')
        canvas.pack(fill='both', expand=True)

        start_x, start_y = None, None
        rect = None

        def on_mouse_down(event):
            nonlocal start_x, start_y, rect
            start_x, start_y = event.x, event.y
            rect = canvas.create_rectangle(start_x, start_y, start_x, start_y, outline='red', width=2)

        def on_mouse_move(event):
            nonlocal rect
            if rect:
                canvas.coords(rect, start_x, start_y, event.x, event.y)

        def on_mouse_up(event):
            nonlocal rect
            end_x, end_y = event.x, event.y
            canvas.coords(rect, start_x, start_y, end_x, end_y)
            selection_window.destroy()
            self.master.deiconify()  # Rendre la fenêtre principale visible

            # Enregistrer les coordonnées
            x1, y1 = min(start_x, end_x), min(start_y, end_y)
            x2, y2 = max(start_x, end_x), max(start_y, end_y)
            zone = (x1, y1, x2, y2)
            if zone_type == 'pollen':
                self.env.set_pollen_zone(zone)
                config.POLLEN_ZONE = zone
            elif zone_type == 'honey':
                self.env.set_honey_zone(zone)
                config.HONEY_ZONE = zone
            logger.info(f"Zone de {zone_type} définie à : {zone}")
            messagebox.showinfo("Zone Définie", f"Zone de {zone_type} définie à : {zone}")

            # Sauvegarder les zones dans le fichier config_zones.json
            self.save_zones()

        canvas.bind("<ButtonPress-1>", on_mouse_down)
        canvas.bind("<B1-Motion>", on_mouse_move)
        canvas.bind("<ButtonRelease-1>", on_mouse_up)

        selection_window.mainloop()

    def save_zones(self):
        """
        Sauvegarde les zones de pollen et de miel dans un fichier de configuration.
        """
        zones = {
            "pollen_zone": config.POLLEN_ZONE if hasattr(config, 'POLLEN_ZONE') else None,
            "honey_zone": config.HONEY_ZONE if hasattr(config, 'HONEY_ZONE') else None
        }
        config_file = os.path.join(os.path.dirname(__file__), 'config_zones.json')
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(zones, f, indent=4)
            logger.info(f"Zones sauvegardées dans {config_file}.")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des zones : {e}")

    def load_zones(self):
        """
        Charge les zones de pollen et de miel depuis un fichier de configuration.
        """
        config_file = os.path.join(os.path.dirname(__file__), 'config_zones.json')
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    zones = json.load(f)
                    if zones.get("pollen_zone"):
                        self.env.set_pollen_zone(tuple(zones["pollen_zone"]))
                    if zones.get("honey_zone"):
                        self.env.set_honey_zone(tuple(zones["honey_zone"]))
                logger.info(f"Zones chargées depuis {config_file}.")
            except Exception as e:
                logger.error(f"Erreur lors du chargement des zones : {e}")
        else:
            logger.warning(f"Fichier de configuration des zones {config_file} non trouvé.")

    def apply_settings(self):
        try:
            timesteps = int(self.timesteps_entry.get())
            learning_rate = float(self.learning_rate_entry.get())
            batch_size = int(self.batch_size_entry.get())
            policy_type = self.policy_type_var.get()

            config.TOTAL_TIMESTEPS = timesteps
            config.LEARNING_RATE = learning_rate
            config.BATCH_SIZE = batch_size
            config.POLICY_TYPE = policy_type

            # Mettre à jour la barre de progression
            self.progress.config(maximum=timesteps)

            logger.info(f"Paramètres mis à jour: TOTAL_TIMESTEPS = {timesteps}, LEARNING_RATE = {learning_rate}, BATCH_SIZE = {batch_size}, POLICY_TYPE = {policy_type}")
            messagebox.showinfo("Paramètres", "Les paramètres ont été mis à jour avec succès.")
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer des valeurs valides pour les paramètres.")

    def start_training(self):
        if not self.training:
            self.training = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.test_button.config(state=tk.DISABLED)
            self.progress['value'] = 0  # Réinitialiser la barre de progression
            self.progress_label.config(text="Progression: 0%")
            # Lancer l'entraînement dans un thread séparé
            self.training_thread = threading.Thread(target=self.train, daemon=True)
            self.training_thread.start()
            logger.info("Entraînement démarré via l'interface graphique.")

    def stop_training_func(self):
        if self.training:
            logger.info("Arrêt de l'entraînement demandé via l'interface graphique.")
            # Pour l'instant, sans interruption, simplement désactiver les boutons
            self.stop_button.config(state=tk.DISABLED)
            messagebox.showinfo("Information", "L'arrêt de l'entraînement n'est pas encore implémenté.")

    def train(self):
        try:
            logger.info("Début de l'entraînement du modèle PPO.")
            # Pour tester, réduire temporairement les timesteps
            agent = train_agent(self.env, total_timesteps=config.TOTAL_TIMESTEPS, device=self.device)

            # Sauvegarder le modèle une fois l'entraînement terminé
            model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, 'ppo_bee_swarm')
            agent.save(model_path)
            logger.info(f"Modèle sauvegardé à {model_path}.")

            # Activer le bouton Test après l'entraînement
            self.master.after(0, lambda: self.test_button.config(state=tk.NORMAL))
            messagebox.showinfo("Entraînement Terminé", "L'entraînement du modèle est terminé.")
        except Exception as e:
            logger.error(f"Erreur durant l'entraînement : {e}")
            messagebox.showerror("Erreur", f"Erreur durant l'entraînement : {e}")
        finally:
            # Indiquer la fin de l'entraînement dans l'interface
            self.training = False
            self.progress.stop()  # Arrêter la barre de progression
            self.master.after(0, lambda: self.start_button.config(state=tk.NORMAL))
            self.master.after(0, lambda: self.stop_button.config(state=tk.DISABLED))

    def test_agent_func(self):
        if not self.training:
            self.test_button.config(state=tk.DISABLED)
            threading.Thread(target=self.test, daemon=True).start()

    def test(self):
        try:
            # Charger le modèle sauvegardé
            model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
            model_path = os.path.join(model_dir, 'ppo_bee_swarm')
            if not os.path.exists(model_path + ".zip"):
                messagebox.showerror("Erreur", f"Modèle non trouvé à {model_path}.zip")
                logger.error(f"Modèle non trouvé à {model_path}.zip")
                return

            model = PPO.load(model_path, device=self.device)
            test_agent(model, self.env)
            messagebox.showinfo("Test Terminé", "Le test de l'agent est terminé.")
        except Exception as e:
            logger.error(f"Erreur durant le test de l'agent : {e}")
            messagebox.showerror("Erreur", f"Erreur durant le test de l'agent : {e}")
        finally:
            self.master.after(0, lambda: self.test_button.config(state=tk.NORMAL))

    def take_screenshot(self):
        """
        Exécute la prise de capture d'écran dans un thread séparé pour éviter de bloquer l'interface.
        """
        threading.Thread(target=self._take_screenshot, daemon=True).start()

    def _take_screenshot(self):
        try:
            # Capture une capture d'écran
            screenshot = screen_capture.capture_screen()
            # Convertir en image PIL pour afficher
            # Reformer l'image pour PIL (C, H, W) -> (H, W, C)
            screenshot_image = screenshot.transpose(1, 2, 0) * 255.0
            pil_image = Image.fromarray(screenshot_image.astype('uint8'))
            pil_image = pil_image.resize((400, 300))  # Redimensionner pour l'affichage
            tk_image = ImageTk.PhotoImage(pil_image)
            # Mettre à jour l'interface graphique dans le thread principal
            self.master.after(0, lambda: self.update_screenshot_label(tk_image))
            logger.info("Capture d'écran prise et affichée dans l'interface.")
        except Exception as e:
            logger.error(f"Erreur lors de la prise de la capture d'écran : {e}")
            self.master.after(0, lambda: messagebox.showerror("Erreur", f"Erreur lors de la prise de la capture d'écran : {e}"))

    def update_screenshot_label(self, tk_image):
        self.screenshot_label.configure(image=tk_image)
        self.screenshot_label.image = tk_image  # Garder une référence

    def update_progress_label(self, percentage):
        self.progress_label.config(text=f"Progression: {percentage:.2f}%")
        self.progress['value'] = percentage

    def on_closing(self):
        if self.training:
            if messagebox.askokcancel("Quitter", "L'entraînement est en cours. Voulez-vous vraiment quitter?"):
                self.master.destroy()
        else:
            self.master.destroy()

    def check_cuda(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info(f"CUDA est disponible. Utilisation du GPU : {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            logger.info("CUDA n'est pas disponible. Utilisation du CPU.")

def main():
    try:
        logger.info("Démarrage de l'application BeeSwarmAI.")
        root = tk.Tk()
        gui = BeeSwarmGUI(root)
        root.protocol("WM_DELETE_WINDOW", gui.on_closing)
        logger.info("Entrée dans la boucle principale Tkinter.")
        root.mainloop()
    except Exception as e:
        logger.error(f"Erreur globale: {e}")
        messagebox.showerror("Erreur", f"Erreur globale : {e}")

if __name__ == "__main__":
    main()
