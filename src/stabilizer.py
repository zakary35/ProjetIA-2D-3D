import cv2
import numpy as np
from collections import deque
from typing import Optional, Union, Tuple

class DepthStabilizer:
    """
    Stabilisateur temporel de cartes de profondeur.
    Offre plusieurs stratégies allant du simple lissage au warping guidé par le flux optique.
    """

    def __init__(self, 
                 method: str = 'confidence', 
                 window_size: int = 5, 
                 alpha_ema: float = 0.5, 
                 warp_tolerance: float = 0.05, 
                 edge_sensitivity: float = 50.0):
        """
        Initialise le stabilisateur.

        Args:
            method (str): Stratégie de stabilisation.
                          - 'raw': Pas de stabilisation (Pass-through).
                          - 'median': Filtre médian temporel (Robuste au bruit, ajoute du lag).
                          - 'ema': Moyenne mobile exponentielle (Rapide, traînées possibles).
                          - 'confidence': Propagation temporelle guidée par flux optique + confiance (État de l'art).
            window_size (int): Taille du buffer pour la méthode 'median'.
            alpha_ema (float): Facteur d'oubli pour 'ema' (0.0 = Figé, 1.0 = Brut).
            warp_tolerance (float): Tolérance relative pour la confiance géométrique (0.05 = 5%).
            edge_sensitivity (float): Sensibilité aux bords pour préserver les détails fins.
        """
        self.method = method
        self.window_size = window_size
        self.alpha_ema = alpha_ema
        self.warp_tolerance = warp_tolerance
        self.edge_sensitivity = edge_sensitivity
        
        # Buffers d'état
        self.buffer = deque(maxlen=window_size)
        self.prev_depth = None
        self.prev_gray = None
        
        # Cache pour la grille de warping (Optimisation vitesse)
        self.grid_cache = None
        self.cache_shape = (0, 0)

    def apply(self, 
              current_depth_raw: np.ndarray, 
              current_img_color: np.ndarray, 
              flow: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Applique la stabilisation sur la frame actuelle.

        Args:
            current_depth_raw (np.ndarray): Carte de profondeur brute [H, W].
            current_img_color (np.ndarray): Image RGB/BGR [H, W, 3] (utilisée pour le guidage).
            flow (np.ndarray, optional): Flux optique pré-calculé [H, W, 2]. 
                                         Si None, calcule Farneback en interne (plus lent).

        Returns:
            np.ndarray: Carte de profondeur stabilisée [H, W].
        """
        if self.method == 'raw':
            return current_depth_raw
            
        elif self.method == 'median':
            return self._apply_median(current_depth_raw)
            
        elif self.method == 'ema':
            return self._apply_ema(current_depth_raw)
            
        elif self.method == 'confidence':
            return self._apply_confidence(current_depth_raw, current_img_color, flow)
            
        else:
            raise ValueError(f"Méthode inconnue : {self.method}")

    def _apply_median(self, depth: np.ndarray) -> np.ndarray:
        """Filtre médian temporel."""
        self.buffer.append(depth)
        # On peut calculer la médiane même si le buffer n'est pas plein pour éviter le lag au démarrage
        if len(self.buffer) > 1:
            stack = np.array(self.buffer)
            return np.median(stack, axis=0).astype(depth.dtype)
        return depth

    def _apply_ema(self, depth: np.ndarray) -> np.ndarray:
        """Exponential Moving Average."""
        if self.prev_depth is None:
            stabilized = depth
        else:
            # Formule : Stablized = alpha * Current + (1-alpha) * Prev
            stabilized = (self.alpha_ema * depth) + ((1 - self.alpha_ema) * self.prev_depth)
        
        self.prev_depth = stabilized
        return stabilized

    def _apply_confidence(self, 
                          current_depth: np.ndarray, 
                          img_color: np.ndarray, 
                          external_flow: Optional[np.ndarray]) -> np.ndarray:
        """
        Méthode avancée : Reprojection temporelle avec masque de confiance.
        """
        current_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        
        # Initialisation (Premier passage)
        if self.prev_depth is None or self.prev_gray is None:
            self.prev_depth = current_depth
            self.prev_gray = current_gray
            return current_depth

        # 1. Gestion du Flux Optique
        if external_flow is not None:
            # Utilisation du flux RAFT fourni par le pipeline (Rapide & Précis)
            flow = external_flow
        else:
            # Fallback : Calcul Farneback (CPU, plus lent)
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, current_gray, None, 
                0.5, 3, 15, 3, 5, 1.2, 0
            )

        # 2. Warping (Reprojection de t-1 vers t)
        h, w = current_depth.shape
        
        # Optimisation : On ne régénère la grille que si la taille change
        if self.cache_shape != (h, w):
            grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
            self.grid_cache = (grid_x, grid_y)
            self.cache_shape = (h, w)
        
        grid_x, grid_y = self.grid_cache
        
        # Application du flux
        map_x = grid_x + flow[..., 0]
        map_y = grid_y + flow[..., 1]
        
        # Remap : BORDER_REPLICATE évite les zones noires sur les bords caméra
        prev_depth_warped = cv2.remap(self.prev_depth, map_x, map_y, 
                                      interpolation=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REPLICATE)

        # 3. Calcul des Masques de Confiance
        
        # A. Confiance Géométrique (Photometric Consistency pour la profondeur)
        # Si la profondeur warpée est très différente de la nouvelle, c'est probablement une occlusion
        diff_geo = np.abs(prev_depth_warped - current_depth)
        # On évite la division par zéro avec 1e-6
        threshold_geo = self.warp_tolerance * np.maximum(current_depth, 1e-3)
        conf_geo = np.clip(1.0 - (diff_geo / (threshold_geo + 1e-6)), 0, 1)

        # B. Confiance des Bords (Spatial Consistency)
        # On se fie moins à l'historique sur les bords nets (discontinuités)
        laplacian = cv2.Laplacian(current_depth, cv2.CV_32F)
        edge_intensity = np.abs(laplacian)
        conf_edge = np.exp(-edge_intensity / self.edge_sensitivity)
        
        # C. Fusion des confiances
        confidence_map = conf_geo * conf_edge
        
        # 4. Fusion Temporelle (Alpha Blending Adaptatif)
        
        # Logique : 
        # - Confiance Haute (1.0) -> Alpha dynamique bas -> On garde l'historique warpé (Stable)
        # - Confiance Basse (0.0) -> Alpha dynamique haut -> On prend la nouvelle mesure (Réactif)
        # Le terme '+ 0.05' assure qu'on intègre toujours un peu de la nouvelle mesure
        alpha_dynamic = (1.0 - confidence_map) + 0.05
        alpha_dynamic = np.clip(alpha_dynamic, 0, 1)

        stabilized = (1 - alpha_dynamic) * prev_depth_warped + alpha_dynamic * current_depth

        # Mise à jour des états
        self.prev_depth = stabilized
        self.prev_gray = current_gray
        
        return stabilized
    
    def reset(self):
        """Réinitialise les buffers (utile lors d'un changement de scène)."""
        self.buffer.clear()
        self.prev_depth = None
        self.prev_gray = None