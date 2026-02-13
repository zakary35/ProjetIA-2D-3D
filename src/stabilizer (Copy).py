#src.stabilizer.py
import cv2
import numpy as np
from collections import deque

class DepthStabilizer:
    def __init__(self, method='confidence', window_size=5, alpha_ema=0.5, 
                 warp_tolerance=0.05, edge_sensitivity=50.0):
        """
        Outil multi-méthodes.
        method: 'raw', 'median', 'ema', 'confidence'
        """
        self.method = method
        self.window_size = window_size
        self.alpha_ema = alpha_ema
        self.warp_tolerance = warp_tolerance
        self.edge_sensitivity = edge_sensitivity
        
        # Buffers
        self.buffer = deque(maxlen=window_size)
        self.prev_depth = None
        self.prev_gray = None
        
    def apply(self, current_depth_raw, current_img_color):
        current_gray = cv2.cvtColor(current_img_color, cv2.COLOR_BGR2GRAY)

        # 1. RAW
        if self.method == 'raw':
            return current_depth_raw

        # 2. MEDIAN
        elif self.method == 'median':
            self.buffer.append(current_depth_raw)
            if len(self.buffer) >= 3:
                stack = np.array(self.buffer)
                return np.median(stack, axis=0)
            return current_depth_raw

        # 3. EMA
        elif self.method == 'ema':
            if self.prev_depth is None:
                stabilized = current_depth_raw
            else:
                stabilized = (self.alpha_ema * current_depth_raw) + ((1 - self.alpha_ema) * self.prev_depth)
            self.prev_depth = stabilized
            return stabilized

        # 4. CONFIDENCE PROPAGATION (Votre méthode avancée)
        elif self.method == 'confidence':
            if self.prev_depth is None or self.prev_gray is None:
                self.prev_depth = current_depth_raw
                self.prev_gray = current_gray
                return current_depth_raw

            # A. Flux Optique
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, current_gray, None, 
                0.5, 3, 20, 3, 7, 1.2, 0
            )

            # B. Warping (Avec mgrid pour éviter les artefacts verticaux)
            h, w = current_depth_raw.shape
            grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
            map_x = grid_x + flow[..., 0]
            map_y = grid_y + flow[..., 1]
            
            # Remap avec BORDER_REPLICATE pour éviter les bords noirs
            prev_depth_warped = cv2.remap(self.prev_depth, map_x, map_y, 
                                          interpolation=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_REPLICATE)

            # C. Carte de Confiance Géométrique
            diff_geo = np.abs(prev_depth_warped - current_depth_raw)
            threshold_geo = self.warp_tolerance * np.maximum(current_depth_raw, 1e-3)
            conf_geo = np.clip(1.0 - (diff_geo / (threshold_geo + 1e-6)), 0, 1)

            # D. Carte de Confiance des Bords (Laplacien)
            # Utilisation de CV_32F pour compatibilité types
            laplacian = cv2.Laplacian(current_depth_raw, cv2.CV_32F)
            edge_intensity = np.abs(laplacian)
            conf_edge = np.exp(-edge_intensity / self.edge_sensitivity)
            
            # E. Fusion
            confidence_map = conf_geo * conf_edge
            
            # Alpha dynamique : Si confiance haute, alpha bas (on garde warp)
            alpha_dynamic = (1.0 - confidence_map) + 0.05
            alpha_dynamic = np.clip(alpha_dynamic, 0, 1)

            stabilized = (1 - alpha_dynamic) * prev_depth_warped + alpha_dynamic * current_depth_raw

            self.prev_depth = stabilized
            self.prev_gray = current_gray
            
            return stabilized

        return current_depth_raw