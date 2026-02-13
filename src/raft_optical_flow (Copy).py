import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import argparse
from typing import Optional, Tuple, Union

# Gestion des imports
try:
    from .raft.raft import RAFT
    from .raft.utils.utils import InputPadder
except ImportError:
    import sys
    sys.path.append('src/raft')
    from raft.raft import RAFT
    from raft.utils.utils import InputPadder

class RAFTFlowEngine:
    """Moteur RAFT sans compilation JIT pour économiser la VRAM."""
    _instance: Optional['RAFTFlowEngine'] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(RAFTFlowEngine, cls).__new__(cls)
        return cls._instance

    def __init__(self, 
                 checkpoint_path: str = "checkpoints/raft/raft-things.pth", 
                 small: bool = False, 
                 device: str = None,
                 iters: int = 12): # Réduit à 12 itérations par défaut (suffisant)
        
        if hasattr(self, 'initialized'): return
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.iters = iters
        self.small = small
        
        print(f"🚀 [RAFT] Initialisation sur : {str(self.device).upper()}")

        self.args = argparse.Namespace(
            small=small,
            mixed_precision=True,
            alternate_corr=False
        )
        
        self.model = RAFT(self.args)
        self._load_checkpoint(checkpoint_path)
        self.model.to(self.device).eval()
        
        # SUPPRESSION DE TORCH.COMPILE POUR ÉVITER LE OOM
        # self.model = torch.compile(self.model) 
            
        self.initialized = True

    def _load_checkpoint(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ [RAFT] Checkpoint introuvable : {path}")
        state_dict = torch.load(path, map_location='cpu')
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(new_state_dict)

    def _preprocess(self, img: np.ndarray) -> torch.Tensor:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float()
        return tensor.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def compute_flow(self, img_prev: np.ndarray, img_curr: np.ndarray) -> np.ndarray:
        if img_prev.shape != img_curr.shape:
            raise ValueError("Dimensions incompatibles")

        img1 = self._preprocess(img_prev)
        img2 = self._preprocess(img_curr)

        padder = InputPadder(img1.shape)
        img1_pad, img2_pad = padder.pad(img1, img2)

        # Autocast strict pour économiser la mémoire
        with torch.amp.autocast(device_type=self.device.type, enabled=True):
            flow_low, flow_up = self.model(img1_pad, img2_pad, iters=self.iters, test_mode=True)
        
        flow_numpy = padder.unpad(flow_up)[0].permute(1, 2, 0).cpu().numpy()
        return flow_numpy
