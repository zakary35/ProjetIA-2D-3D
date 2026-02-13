import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Optional, Union
from src.vda.video_depth_anything.video_depth_stream import VideoDepthAnything
from src.dav2.dpt import DepthAnythingV2

class BaseDepthEngine:
    def __init__(self, device: str = None):
        # 1. Détection automatique du GPU
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        self.device = torch.device(self.device)

        print(f"🚀 Initialisation du DepthEngine sur : {self.device.upper()}")
        
        self.model = None

    def infer(self, frame: np.ndarray) -> np.ndarray:
        """Prend une image RGB [H, W, 3] et retourne la profondeur [H, W]."""
        raise NotImplementedError
    

class VDAEngine(BaseDepthEngine):
    """Moteur Video Depth Anything avec gestion du cache temporel (Streaming)."""
    def __init__(self, model_size: str = 'vitl', device: str = None, input_size: int = 518, fp32: bool = False):
        """
        Initialisation de la classe
            :param model_size: Taille du modèle choix=['vits', 'vitb', 'vitl']
            :param device: Mettre sur GPU ou autre
            :param input_size:
            :param fp32: Le model infer avec torch.float32, par defaut c'est torch.float16
        """

        # 1. Initialisation avec la classe mère
        super().__init__(device)
        self.input_size = input_size
        self.fp32 = fp32
        
        # 2. Configuration du modèle (Architecture Encoder-Decoder)
        # Ces paramètres correspondent à ceux définis dans le papier de recherche

        configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        if model_size not in configs:
            raise ValueError(f"Modèle inconnu : {model_size}")
        self.model = VideoDepthAnything(**configs[model_size])

        # 3. Chargement des poids (.pth)
        checkpoint_path = f'checkpoints/vda/video_depth_anything_{model_size}.pth'
        try:
            # map_location='cpu' évite de surcharger la mémoire au chargement
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            self.model.load_state_dict(state_dict)
            self.model.to(self.device).eval() # Mode évaluation (pas d'entraînement)
            print(f"✅ Modèle {model_size} chargé avec succès.")
        except FileNotFoundError:
            raise FileNotFoundError(f"❌ ERREUR : Le fichier {checkpoint_path} manque.")
        
    def infer(self, frame: np.ndarray) -> np.ndarray:
        """
        Calcul la carte de profondeur de l'image d'entrer
            :param frame: Image [H, W, 3] (BGR) [0,255]
            :return depth:  La carte de profondeur [H, W] (Float32/Float16) [0,1]
        """
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)# Conversion BRG en RGB car VideoDepthAnything infère a partir d'image RGB

        # La méthode infer_video_depth_one gère automatiquement un cache interne pour la reinjection
        depth = self.model.infer_video_depth_one(frame, input_size=self.input_size, device=self.device, fp32=self.fp32)
        
        return depth


class DAV2Engine(BaseDepthEngine):
    """Moteur Depth Anything V2 classique (Image-par-image)."""
    def __init__(self, model_size: str = 'vitl', device: str = None, input_size: int =518):
        """
        Initialisation de la classe
            :param model_size: Taille du modèle choix=['vits', 'vitb', 'vitl','vitg']
            :param device: Mettre sur GPU ou autre
            :param input_size:
        """
        super().__init__(device)
        self.input_size = input_size

        # 2. Configuration du modèle (Architecture Encoder-Decoder)
        # Ces paramètres correspondent à ceux définis dans le papier de recherche
        configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        
        if model_size not in configs:
            raise ValueError(f"Modèle inconnu : {model_size}")
        self.model = DepthAnythingV2(**configs[model_size])

        # 3. Chargement des poids (.pth)
        checkpoint_path = f'checkpoints/dav2/depth_anything_v2_{model_size}.pth'
        try:
            # map_location='cpu' évite de surcharger la mémoire au chargement
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            self.model.load_state_dict(state_dict)
            self.model.to(self.device).eval() # Mode évaluation (pas d'entraînement)
            print(f"✅ Modèle {model_size} chargé avec succès.")
        except FileNotFoundError:
            raise FileNotFoundError(f"❌ ERREUR : Le fichier {checkpoint_path} manque.")

    def infer(self, frame: np.ndarray) -> np.ndarray:
        """
        Calcul la carte de profondeur de l'image d'entrer
            :param frame: Image [H, W, 3] (BGR) [0,255]
            :return depth:  La carte de profondeur [H, W] (Float32/Float16) [0,1]
        """

        # DepthAnythingV2 infère à partir d'image  BRG
        return self.model.infer_image(frame, input_size=self.input_size)