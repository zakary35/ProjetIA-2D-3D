import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Optional, Union, Dict, Any

# Import des modèles (Assurez-vous que les chemins d'import sont bons dans votre projet)
from src.vda.video_depth_anything.video_depth_stream import VideoDepthAnything
from src.dav2.dpt import DepthAnythingV2

class BaseDepthEngine:
    """
    Classe abstraite définissant l'interface pour les moteurs d'estimation de profondeur.
    Gère la détection du matériel (CUDA/MPS/CPU).
    """
    def __init__(self, device: str = None):
        """
        Initialise le gestionnaire de périphérique.

        Args:
            device (str, optional): 'cuda', 'mps' ou 'cpu'. Si None, détecte automatiquement.
        """
        # 1. Détection automatique du GPU
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        print(f"🚀 [DepthEngine] Initialisation sur : {str(self.device).upper()}")
        self.model = None

    def infer(self, frame: np.ndarray) -> np.ndarray:
        """
        Méthode abstraite pour l'inférence.
        
        Args:
            frame (np.ndarray): Image source [H, W, 3] (BGR).

        Returns:
            np.ndarray: Carte de profondeur [H, W].
        
        Raises:
            NotImplementedError: Si la classe fille n'implémente pas cette méthode.
        """
        raise NotImplementedError
    

class VDAEngine(BaseDepthEngine):
    """
    Moteur pour 'Video Depth Anything' (VDA).
    Optimisé pour la cohérence temporelle grâce à une gestion d'état (Memory/State).
    """
    
    # Configuration architecturale du modèle (Static)
    CONFIGS = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    def __init__(self, 
                 model_size: str = 'vitl', 
                 device: str = None, 
                 input_size: int = 518, 
                 fp32: bool = False,
                 checkpoint_path: Optional[str] = None):
        """
        Initialise le modèle Video Depth Anything.

        Args:
            model_size (str): Taille du modèle ('vits', 'vitb', 'vitl').
            device (str): Périphérique d'exécution.
            input_size (int): Taille de redimensionnement interne pour l'inférence.
            fp32 (bool): Si True, utilise float32 (plus lent, plus précis). Sinon float16.
            checkpoint_path (str, optional): Chemin spécifique vers le fichier .pth.
        """
        super().__init__(device)
        self.input_size = input_size
        self.fp32 = fp32
        
        if model_size not in self.CONFIGS:
            raise ValueError(f"Modèle inconnu : {model_size}. Choix : {list(self.CONFIGS.keys())}")

        # 1. Instanciation de l'architecture
        print(f"🏗️ [VDA] Construction du modèle {model_size}...")
        self.model = VideoDepthAnything(**self.CONFIGS[model_size])

        # 2. Gestion du chemin des poids
        if checkpoint_path is None:
            # Chemin par défaut
            checkpoint_path = f'checkpoints/vda/video_depth_anything_{model_size}.pth'
        
        # 3. Chargement sécurisé
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"❌ [VDA] ERREUR : Poids introuvables à : {checkpoint_path}")
            
        try:
            # map_location='cpu' évite de saturer la VRAM pendant le chargement initial
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            self.model.load_state_dict(state_dict)
            self.model.to(self.device).eval() # Mode freeze (pas de gradients)
            print(f"✅ [VDA] Modèle chargé et prêt.")
        except Exception as e:
            raise RuntimeError(f"❌ [VDA] Erreur lors du chargement des poids : {e}")
        
    def infer(self, frame: np.ndarray) -> np.ndarray:
        """
        Calcule la profondeur en tenant compte de l'historique vidéo.

        Args:
            frame (np.ndarray): Image actuelle [H, W, 3] en format BGR (Standard OpenCV).

        Returns:
            np.ndarray: Carte de profondeur brute [H, W] (Float32).
                        Note: Ce n'est pas normalisé [0,1], c'est une disparité relative.
        """
        # Conversion BGR -> RGB requise par VDA
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Inférence avec gestion du cache temporel interne
        # Le paramètre 'infer_video_depth_one' suggère un traitement image par image avec mémoire
        depth = self.model.infer_video_depth_one(
            frame_rgb, 
            input_size=self.input_size, 
            device=self.device, 
            fp32=self.fp32
        )
        
        return depth


class DAV2Engine(BaseDepthEngine):
    """
    Moteur pour 'Depth Anything V2' (DAV2).
    Traitement image par image sans cohérence temporelle (Single Image Depth Estimation).
    """
    
    # Configuration architecturale
    CONFIGS = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    def __init__(self, 
                 model_size: str = 'vitl', 
                 device: str = None, 
                 input_size: int = 518,
                 checkpoint_path: Optional[str] = None):
        """
        Initialise le modèle Depth Anything V2.

        Args:
            model_size (str): Taille du modèle ('vits', 'vitb', 'vitl', 'vitg').
            device (str): Périphérique.
            input_size (int): Taille d'entrée (518 est le standard pour DAV2).
            checkpoint_path (str, optional): Chemin vers les poids .pth.
        """
        super().__init__(device)
        self.input_size = input_size

        if model_size not in self.CONFIGS:
            raise ValueError(f"Modèle inconnu : {model_size}. Choix : {list(self.CONFIGS.keys())}")

        # 1. Instanciation
        print(f"🏗️ [DAV2] Construction du modèle {model_size}...")
        self.model = DepthAnythingV2(**self.CONFIGS[model_size])

        # 2. Gestion du chemin
        if checkpoint_path is None:
            checkpoint_path = f'checkpoints/dav2/depth_anything_v2_{model_size}.pth'

        # 3. Chargement
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"❌ [DAV2] ERREUR : Poids introuvables à : {checkpoint_path}")

        try:
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            self.model.load_state_dict(state_dict)
            self.model.to(self.device).eval()
            print(f"✅ [DAV2] Modèle chargé et prêt.")
        except Exception as e:
            raise RuntimeError(f"❌ [DAV2] Erreur lors du chargement : {e}")

    def infer(self, frame: np.ndarray) -> np.ndarray:
        """
        Calcule la profondeur d'une image unique.

        Args:
            frame (np.ndarray): Image [H, W, 3] en BGR.

        Returns:
            np.ndarray: Carte de profondeur brute [H, W].
                        Attention : Les valeurs ne sont pas bornées [0, 1].
        """
        # DepthAnythingV2.infer_image gère généralement la conversion BGR/RGB en interne 
        # ou attend du BGR standard OpenCV. Nous passons la frame brute.
        return self.model.infer_image(frame, input_size=self.input_size)