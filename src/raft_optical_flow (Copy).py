import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from typing import Optional, Tuple

# Importations relatives aux dossiers RAFT
from .raft.raft import RAFT
from .raft.utils.utils import InputPadder

class RAFTFlowEngine:
    """
    Moteur RAFT optimisé. 
    Gère le cache du modèle, la compilation et le multi-GPU.
    """
    _instance: Optional['RAFTFlowEngine'] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(RAFTFlowEngine, cls).__new__(cls)
        return cls._instance

    def __init__(self, checkpoint_path: str = "checkpoints/raft/raft-things.pth", 
                 small: bool = False, device: str = "cuda:0"):
        # Évite la ré-initialisation si l'instance existe déjà
        if hasattr(self, 'initialized'): return
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.args = self._create_config(small)
        
        # 1. Chargement de l'architecture
        self.model = RAFT(self.args)
        
        # 2. Chargement des poids (Professionnel : CPU puis GPU)
        self._load_checkpoint(checkpoint_path)
        
        # 3. Optimisation Multi-GPU si disponible
        if torch.cuda.device_count() > 1 and "cuda" in str(self.device):
            self.model = nn.DataParallel(self.model)
            
        self.model.to(self.device).eval()
        
        # 4. Compilation Torch (PyTorch 2.x+) pour accélérer l'inférence
        try:
            self.model = torch.compile(self.model, mode="reduce-overhead")
            print("🚀 RAFT compilé avec succès.")
        except Exception:
            print("⚠️ Échec de la compilation, passage en mode standard.")
            
        self.initialized = True

    @staticmethod
    def _create_config(small: bool):
        """Crée un objet config compatible avec les attentes de RAFT."""
        from easydict import EasyDict
        return EasyDict({
            'small': small,
            'mixed_precision': True, # Accélère sur T4/P100
            'alternate_corr': False
        })

    def _load_checkpoint(self, path: str):
        """Charge les poids de manière robuste."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Checkpoint RAFT introuvable : {path}")
            
        state_dict = torch.load(path, map_location='cpu')
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(new_state_dict)

    def _preprocess(self, img: np.ndarray) -> torch.Tensor:
        """Prétraitement : BGR [H, W, 3] -> RGB [1, 3, H, W] sur GPU."""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float()
        return tensor.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def compute_flow(self, img_prev: np.ndarray, img_curr: np.ndarray, iters: int = 20) -> np.ndarray:
        """
        Calcule le flux optique entre deux images.
        Tenseurs : img_pad [1, 3, H_pad, W_pad], flow [1, 2, H, W]
        """
        img1 = self._preprocess(img_prev)
        img2 = self._preprocess(img_curr)

        padder = InputPadder(img1.shape)
        img1_pad, img2_pad = padder.pad(img1, img2)

        # Autocast pour utiliser les Tensor Cores du GPU
        with torch.amp.autocast(device_type="cuda"):
            _, flow_up = self.model(img1_pad, img2_pad, iters=iters, test_mode=True)
        
        # Unpad et conversion NumPy
        flow = padder.unpad(flow_up)[0].permute(1, 2, 0).cpu().numpy()
        return flow # [H, W, 2]