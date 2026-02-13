import torch
import torch.nn as nn
import numpy as np
import cv2
from scipy import signal
from typing import Tuple, List, Optional
import lpips 

# Import de ton moteur RAFT
from src.raft_optical_flow import RAFTFlowEngine


class StabilityMetrics:
    """
    Suite de métriques professionnelles pour l'évaluation de la stabilité temporelle.
    Singleton pour éviter de recharger LPIPS/RAFT à chaque instanciation.
    Toutes les méthodes acceptent des tableaux NumPy ou des Tenseurs PyTorch.
    """

    _instance: Optional['StabilityMetrics'] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(StabilityMetrics, cls).__new__(cls)
        return cls._instance

    def __init__(self, device='cpu', use_raft = True , raft_ch_path: str = None):
        # Évite la ré-initialisation si l'instance existe déjà
        if hasattr(self, 'initialized'): return
        self.devive = device
        self.lips = lpips.LPIPS(net='vgg').to(device).eval()
        self.raft_engine = None
        self.use_raft = use_raft
        if self.use_raft:
            self.raft_engine = RAFTFlowEngine(checkpoint_path=raft_ch_path)

        self.initialized = True
    
    def _calculate_temporal_lpips(self, depth_curr: np.ndarray, depth_prev_warped: np.ndarray, device: str = "cuda") -> float:
        """
        T-LPIPS : Distance perceptuelle entre la frame actuelle et la précédente recalée.
        Référence : Lei et al. (2020) "Deep Video Prior".
        
        :param depth_curr: [H, W] Profondeur à l'instant t [0, 1].
        :param depth_prev_warped: [H, W] Profondeur t-1 warpée vers t [0, 1].
        """
        
        def to_lpips_tensor(d: np.ndarray) -> torch.Tensor:
            # Normalisation [0, 1] -> [-1, 1] requis par LPIPS
            d = (d - d.min()) / (d.max() - d.min() + 1e-6)
            d = (d * 2.0) - 1.0 

            # Passage de [H, W] à [1, 3, H, W]
            t = torch.from_numpy(d).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
            return t.to(device)

        with torch.no_grad():
            d1 = to_lpips_tensor(depth_curr)
            d2 = to_lpips_tensor(depth_prev_warped)
            dist = self.lips(d1, d2)
        return float(dist.item())
    
    @staticmethod
    def is_scene_change(frame_prev: np.ndarray, frame_curr: np.ndarray, threshold: float = 0.6) -> bool:
        """
        Détecte un changement de plan (Cut) via corrélation d'histogrammes HSV.
        Si la corrélation < threshold, c'est une nouvelle scène.
        """
        # Conversion en HSV pour être moins sensible aux changements de luminosité
        hsv_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2HSV)
        hsv_curr = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2HSV)

        # Calcul de l'histogramme sur la teinte (Hue) et la saturation (Sat)
        # On ignore la Value (Luminosité) pour éviter les faux positifs sur les flashs
        hist_prev = cv2.calcHist([hsv_prev], [0, 1], None, [50, 60], [0, 180, 0, 256])
        hist_curr = cv2.calcHist([hsv_curr], [0, 1], None, [50, 60], [0, 180, 0, 256])

        # Normalisation
        cv2.normalize(hist_prev, hist_prev, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_curr, hist_curr, 0, 1, cv2.NORM_MINMAX)

        # Comparaison (Corrélation : 1 = Identique, 0 = Différent)
        score = cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_CORREL)

        return score < threshold
    
    def calculate_flikering_error(self,
                                  depth_curr: np.ndarray, 
                                  depth_prev: np.ndarray, 
                                  frame_t: np.ndarray,
                                  frame_prev: np.ndarray) -> Tuple[float, float]:
        """
        Mesure l'incohérence temporelle (Scintillement) en utilisant le Flux Optique WarpingError L1 et Temporal-LPIPS
        WarpingError L1 : distance L1 entre la depth map t et la depth map t-1 wrapé (réalignée par flux optique).
        Temporal-LPIPS : distance LPIPS entre la depth map t et la depth map t-1 wrapé (réalignée par flux optique).
        Score Bas = Stable. Score Haut = Scintillement.*
        :param depth_t: depth map actuelle (t)
        :param depth_prev: depth map précédente (t-1)
        :param fram_t: frame actuelle en BGR (t)
        :param frame_prev: frame précédente en BGR (t-1)
        :return: (WarpingError L1, Temporal-LPIPS)
        """

        # 0. CHECK SCENE CHANGE
        # Si c'est un nouveau plan, le warping n'a aucun sens.
        if self.is_scene_change(frame_prev, frame_curr):
            # On retourne 0.0 ou -1.0 pour signaler qu'il faut ignorer cette frame
            print("Il y a changement de scènes")
            return 0.0, 0.0

        # 1. Calcul du Flux Optique
        if self.use_raft and self.raft_engine:
            # RAFT attend du BGR et renvoie [H, W, 2]
            flow = self.raft_engine.compute_flow(frame_prev, frame_t)
        else:
            prev_gray = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(frame_t, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # 2. Warping (Recalage géométrique)
        h, w = depth_prev.shape
        # Création de la grille de coordonnées
        # Attention : flow est en (x, y), donc on ajoute flow[..., 0] à grid_x
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + flow[..., 0]).astype(np.float32)
        map_y = (grid_y + flow[..., 1]).astype(np.float32)

        # On déforme l'image précédente pour qu'elle matche l'actuelle
        warped_prev_depth = cv2.remap(depth_prev, map_x, map_y, interpolation=cv2.INTER_LINEAR)

        # 3. Masque de validité (pour ignorer les bords qui rentrent dans l'image)
        mask = (map_x > 2) & (map_x < w-2) & (map_y > 2) & (map_y < h-2)
        
        # 4. Calcul des erreurs
        # E_warp (L1)
        if np.sum(mask) == 0: return 0.0, 0.0 # Sécurité si mouvement trop violent
        l1_err = float(np.mean(np.abs(depth_curr[mask] - warped_prev_depth[mask])))
        
        # T-LPIPS (Perceptuel)
        lpips_error = self._calculate_temporal_lpips(depth_curr, warped_prev_depth)
        
        return l1_err, lpips_error
    
    @staticmethod
    def calculate_sharpness(depth_map: np.ndarray) -> float:
        """
        Mesure la netteté via la variance du Laplacien.
        Score Haut = Net (Bords précis). Score Bas = Flou.
        :param depth_map: [H, W] (idéalement normalisée en uint8 pour la variance du Laplacien)
        :return: le score de netteté
        """
        # On s'assure d'avoir une plage de valeurs significative
        if depth_map.dtype != np.uint8:
            depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            depth_vis = depth_map
            
        laplacian = cv2.Laplacian(depth_vis, cv2.CV_64F)
        return laplacian.var()
    
    @staticmethod
    def calculate_fidelity_rmse(depth_stabilized: np.ndarray, depth_raw: np.ndarray) -> float:
        """
        Mesure la différence avec le signal brut.
        Score Haut = Lag (Retard) ou Ghosting. Score Bas = Fidèle.
        """
        diff = (depth_stabilized - depth_raw) ** 2
        rmse = np.sqrt(np.mean(diff))
        return rmse

    @staticmethod
    def calculate_edge_alignment(frame: np.ndarray, depth_map: np.ndarray) -> float:
        """
        Corrélation entre les gradients de l'image RGB et de la profondeur.
        Référence : Yang et al. (2024) "Depth Anything V2".
        Interprétation du score:
            Score > 0.5 (Excellent): Les contours des objets dans la carte de profondeur collent parfaitement aux objets de la vidéo.
            Score entre 0.1 et 0.4 (Moyen / Correct): Les objets principaux sont là, mais les bords sont flous ou légèrement décalés.
            Score proche de 0 (Mauvais): Aucune corrélation. Le modèle "hallucine" des formes qui n'existent pas ou rate complètement
                                        les objets fins (comme des câbles ou des feuilles)
            Score Négatif (< 0): C'est une anomalie grave (surtout avec la méthode des magnitudes). Cela signifie que là où l'image 
                                        est complexe, la profondeur est plate, et inversement. C'est souvent signe d'un échec total 
                                        du modèle sur cette frame.
            ⚠️ un score de 1.0 n'est pas forcément souhaitable on a du Texture Copying

        :param frame: [H, W, 3] Image (BGR OpenCV).
        :param depth_map: [H, W] Carte de profondeur.
        :return: score d'alignement
        """
        
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Gradients séparés X et Y
        sobel_x_img = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y_img = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
        mag_img = cv2.magnitude(sobel_x_img, sobel_y_img)
        
        sobel_x_depth = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y_depth = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)
        mag_depth = cv2.magnitude(sobel_x_depth, sobel_y_depth)
        
        flat_img = mag_img.flatten()
        flat_depth = mag_depth.flatten()
        
        if np.std(flat_img) == 0 or np.std(flat_depth) == 0:
            return 0.0
            
        return float(np.corrcoef(flat_img, flat_depth)[0, 1])
    
    @staticmethod
    def calculate_avg_psd(depth_stack: np.ndarray, fs: float = 30.0, 
                      grid_size: int = 10, margin_ratio: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        AJOUTÉ : Calcule la PSD moyenne sur une grille pour quantifier le flickering global.
        :param depth_stack: [T, H, W]
        :param margin_ratio: Pourcentage de l'image à ignorer sur les bords (0.1 = 10%).
        :return:
        """
        T, H, W = depth_stack.shape
    
        # Calcul des marges en pixels
        margin_h = int(H * margin_ratio)
        margin_w = int(W * margin_ratio)
        
        # Sécurité : Si l'image est trop petite, on réduit la marge
        if margin_h * 2 >= H or margin_w * 2 >= W:
            margin_h, margin_w = 0, 0
        
        # Création de la grille dans la zone centrale (Safe Zone)
        y_coords = np.linspace(margin_h, H - margin_h, grid_size, dtype=int)
        x_coords = np.linspace(margin_w, W - margin_w, grid_size, dtype=int)
        
        all_psds = []
        freq_axis = None
        
        for y in y_coords:
            for x in x_coords:
                sig = depth_stack[:, y, x]
                f, pxx = signal.welch(sig, fs=fs, nperseg=min(len(sig), 64))
                if len(pxx) > 0:
                    all_psds.append(pxx)
                    if freq_axis is None: freq_axis = f
                    
        if not all_psds: return np.array([]), np.array([])
        return freq_axis, np.mean(all_psds, axis=0)
    
    @staticmethod
    def extract_kymogram_slice(depth_stack: np.ndarray, axis: int = 0, coord: int = 360) -> np.ndarray:
        """
        Génère une coupe spatio-temporelle (Kymogramme).
        
        :param depth_stack: [T, H, W] Pile de cartes de profondeur.
        :param axis: 0 pour une coupe horizontale (Y fixé), 1 pour verticale (X fixé).
        :param coord: La coordonnée de la coupe.
        :return: [T, W] ou [T, H] Image spatio-temporelle.
        """
        if axis == 0:
            return depth_stack[:, coord, :] # [T, W]
        return depth_stack[:, :, coord] # [T, H]



