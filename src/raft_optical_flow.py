import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import argparse
from typing import Optional, Tuple, Union

# Gestion des imports relatifs (Le dossier 'raft' doit être présent)
try:
    from .raft.raft import RAFT
    from .raft.utils.utils import InputPadder
except ImportError:
    # Fallback si lancé depuis la racine sans structure de package
    import sys
    sys.path.append('src/raft') # Ajustez selon votre structure
    try:
        from raft.raft import RAFT
        from raft.utils.utils import InputPadder
    except ImportError:
        raise ImportError("❌ Le module 'raft' (code source officiel) est introuvable. "
                          "Veuillez cloner le repo RAFT dans le dossier src/.")

class RAFTFlowEngine:
    """
    Moteur d'estimation de Flux Optique basé sur RAFT (Recurrent All-Pairs Field Transforms).
    Optimisé pour l'inférence vidéo avec compilation JIT et gestion mémoire.
    
    Pattern Singleton : Assure une seule instance du modèle en mémoire.
    """
    _instance: Optional['RAFTFlowEngine'] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(RAFTFlowEngine, cls).__new__(cls)
        return cls._instance

    def __init__(self, 
                 checkpoint_path: str = "checkpoints/raft/raft-things.pth", 
                 small: bool = False, 
                 device: str = None,
                 iters: int = 20):
        """
        Initialise le moteur RAFT.

        Args:
            checkpoint_path (str): Chemin vers le modèle pré-entraîné (.pth).
            small (bool): Si True, utilise RAFT-Small (plus rapide, moins précis).
            device (str): 'cuda' ou 'cpu'.
            iters (int): Nombre d'itérations du GRU interne (Défaut=20). 
                         Réduire à 10-12 accélère massivement sans trop perdre en qualité.
        """
        # Évite la ré-initialisation si l'instance existe déjà (Singleton)
        if hasattr(self, 'initialized'): return
        
        # 1. Configuration du Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.iters = iters
        self.small = small
        
        print(f"🚀 [RAFT] Initialisation sur : {str(self.device).upper()}")

        # 2. Création de la configuration (Sans EasyDict)
        self.args = argparse.Namespace(
            small=small,
            mixed_precision=True, # Active le FP16 pour les calculs internes
            alternate_corr=False
        )
        
        # 3. Chargement de l'architecture
        try:
            self.model = RAFT(self.args)
        except Exception as e:
            raise RuntimeError(f"❌ [RAFT] Erreur lors de l'instanciation du modèle : {e}")
        
        # 4. Chargement des poids
        self._load_checkpoint(checkpoint_path)
        
        self.model.to(self.device).eval()
        
        # 5. Optimisation : Torch Compile (PyTorch 2.x+)
        # Note : DataParallel a été retiré car inutile pour un batch size de 1 (Vidéo Stream)
        if torch.cuda.is_available():
            try:
                # 'reduce-overhead' est idéal pour les petites boucles d'inférence répétées
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("⚡ [RAFT] Compilation JIT activée (PyTorch 2.0+).")
            except Exception:
                print("ℹ️ [RAFT] Compilation échouée ou indisponible. Passage en mode standard.")
            
        self.initialized = True

    def _load_checkpoint(self, path: str):
        """
        Charge les poids de manière robuste (CPU -> GPU).
        Gère le préfixe 'module.' si le checkpoint vient d'un entraînement Multi-GPU.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ [RAFT] Checkpoint introuvable : {path}")
            
        # Chargement sur CPU pour éviter les OOM immédiats
        state_dict = torch.load(path, map_location='cpu')
        
        # Nettoyage des clés (retrait de 'module.' si présent)
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        try:
            self.model.load_state_dict(new_state_dict)
        except RuntimeError as e:
            print(f"⚠️ [RAFT] Attention : Mismatch de clés. Vérifiez que 'small={self.small}' correspond au checkpoint.")
            raise e

    def _preprocess(self, img: np.ndarray) -> torch.Tensor:
        """
        Prétraitement : BGR [H, W, 3] -> RGB Normalisé [1, 3, H, W] sur GPU.
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float()
        return tensor.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def compute_flow(self, img_prev: np.ndarray, img_curr: np.ndarray) -> np.ndarray:
        """
        Calcule le flux optique dense entre deux frames.

        Args:
            img_prev (np.ndarray): Image à t-1 [H, W, 3] (BGR).
            img_curr (np.ndarray): Image à t [H, W, 3] (BGR).

        Returns:
            np.ndarray: Champ de vecteurs [H, W, 2] (Float32).
                        Canal 0 : Déplacement Horizontal (dx).
                        Canal 1 : Déplacement Vertical (dy).
        """
        # Validation des dimensions
        if img_prev.shape != img_curr.shape:
            raise ValueError(f"Dimensions incompatibles : {img_prev.shape} vs {img_curr.shape}")

        img1 = self._preprocess(img_prev)
        img2 = self._preprocess(img_curr)

        # Padding : RAFT nécessite des dimensions divisibles par 8
        padder = InputPadder(img1.shape)
        img1_pad, img2_pad = padder.pad(img1, img2)

        # Inférence avec précision mixte (Automatic Mixed Precision)
        # Accélère significativement sur les GPU modernes (T4, A100, RTX)
        with torch.amp.autocast(device_type=self.device.type, enabled=True):
            # RAFT retourne une liste de flux (du plus grossier au plus fin)
            # On prend le dernier élément ([-1]) qui est le plus raffiné
            flow_low, flow_up = self.model(img1_pad, img2_pad, iters=self.iters, test_mode=True)
        
        # Post-traitement : Unpad et retour sur CPU
        # flow_up est [1, 2, H, W]
        flow_tensor = padder.unpad(flow_up)[0] 
        flow_numpy = flow_tensor.permute(1, 2, 0).cpu().numpy() # [H, W, 2]
        
        return flow_numpy