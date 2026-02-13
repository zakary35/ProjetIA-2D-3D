import torch
import torch.nn as nn
import numpy as np
import cv2
from scipy import signal
from typing import Tuple, List, Optional
import lpips 

# Gestion robuste de l'import RAFT
try:
    from src.raft_optical_flow import RAFTFlowEngine
    RAFT_AVAILABLE = True
except ImportError:
    print("⚠️ Attention : Module RAFT introuvable. Fallback sur Farneback.")
    RAFT_AVAILABLE = False


class StabilityMetrics:
    """
    Suite de métriques professionnelles pour l'évaluation de la stabilité temporelle et de la qualité géométrique.
    Implémente le pattern Singleton pour optimiser le chargement des réseaux neuronaux (LPIPS, RAFT).
    """

    _instance: Optional['StabilityMetrics'] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(StabilityMetrics, cls).__new__(cls)
        return cls._instance

    def __init__(self, device='cuda', use_raft=True, raft_ch_path: str = "checkpoints/raft/raft-things.pth"):
        """
        Initialise les modèles de métriques sur le GPU.
        
        Args:
            device (str): 'cuda' ou 'cpu'.
            use_raft (bool): Si True, utilise RAFT pour le flux optique (plus précis mais plus lourd).
            raft_ch_path (str): Chemin vers les poids du modèle RAFT.
        """
        # Évite la ré-initialisation si l'instance existe déjà
        if hasattr(self, 'initialized'): return
        
        self.device = torch.device(device)
        
        # LPIPS : Chargé une seule fois, mis en mode eval
        self.loss_fn_lpips = lpips.LPIPS(net='vgg').to(self.device).eval()
        
        self.use_raft = use_raft and RAFT_AVAILABLE
        self.raft_engine = None
        
        if self.use_raft:
            self.raft_engine = RAFTFlowEngine(checkpoint_path=raft_ch_path, device=device)
            print(f"✅ Metrics Engine : RAFT chargé sur {self.device}")
        else:
            print(f"ℹ️ Metrics Engine : Utilisation de Farneback (CPU/OpenCV)")

        self.initialized = True
    
    def _calculate_temporal_lpips(self, depth_curr: np.ndarray, depth_prev_warped: np.ndarray) -> float:
        """
        Calcule la distance perceptuelle (T-LPIPS) entre deux frames de profondeur.
        
        Args:
            depth_curr (np.ndarray): Profondeur frame t [H, W].
            depth_prev_warped (np.ndarray): Profondeur frame t-1 warpée vers t [H, W].
            
        Returns:
            float: Score de distance (0.0 = Identique, >0.5 = Très différent).
        """
        def to_lpips_tensor(d: np.ndarray) -> torch.Tensor:
            # Normalisation robuste min-max vers [0, 1] puis [-1, 1] pour LPIPS
            val_min, val_max = d.min(), d.max()
            if val_max - val_min > 1e-6:
                d = (d - val_min) / (val_max - val_min)
            d = (d * 2.0) - 1.0 

            # [H, W] -> [1, 3, H, W] (Duplication sur 3 canaux pour simuler RGB)
            t = torch.from_numpy(d).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
            return t.to(self.device, dtype=torch.float32)

        with torch.no_grad():
            d1 = to_lpips_tensor(depth_curr)
            d2 = to_lpips_tensor(depth_prev_warped)
            dist = self.loss_fn_lpips(d1, d2)
        return float(dist.item())
    
    @staticmethod
    def is_scene_change(frame_prev: np.ndarray, frame_curr: np.ndarray, threshold: float = 0.6) -> bool:
        """
        Détecte un changement de plan (Cut) via corrélation d'histogrammes HSV.
        
        Args:
            frame_prev (np.ndarray): Image t-1 BGR [H, W, 3].
            frame_curr (np.ndarray): Image t BGR [H, W, 3].
            threshold (float): Seuil de corrélation (Défaut 0.6).
            
        Returns:
            bool: True si c'est une nouvelle scène (Cut détecté).
        """
        # Conversion HSV pour robustesse luminosité
        hsv_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2HSV)
        hsv_curr = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2HSV)

        # Histogramme sur Hue (Teinte) et Saturation
        hist_prev = cv2.calcHist([hsv_prev], [0, 1], None, [50, 60], [0, 180, 0, 256])
        hist_curr = cv2.calcHist([hsv_curr], [0, 1], None, [50, 60], [0, 180, 0, 256])

        cv2.normalize(hist_prev, hist_prev, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_curr, hist_curr, 0, 1, cv2.NORM_MINMAX)

        score = cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_CORREL)
        return score < threshold
    
    def calculate_flickering_error(self,
                                  depth_curr: np.ndarray, 
                                  depth_prev: np.ndarray, 
                                  frame_curr: np.ndarray,
                                  frame_prev: np.ndarray) -> Tuple[float, float]:
        """
        Mesure l'incohérence temporelle (Scintillement/Flickering).
        Combine une erreur pixel (L1) et une erreur perceptuelle (LPIPS).
        
        Args:
            depth_curr (np.ndarray): Carte de profondeur à t [H, W].
            depth_prev (np.ndarray): Carte de profondeur à t-1 [H, W].
            frame_curr (np.ndarray): Image couleur à t [H, W, 3].
            frame_prev (np.ndarray): Image couleur à t-1 [H, W, 3].
            
        Returns:
            Tuple[float, float]: (WarpingError_L1, Temporal_LPIPS).
            Retourne (0.0, 0.0) si un changement de scène est détecté.
        """

        # 0. Sécurité Changement de Scène
        if self.is_scene_change(frame_prev, frame_curr):
            # On ignore cette frame pour ne pas polluer les statistiques
            return 0.0, 0.0

        # 1. Calcul du Flux Optique (Motion Vectors)
        if self.use_raft and self.raft_engine:
            flow = self.raft_engine.compute_flow(frame_prev, frame_curr)
        else:
            prev_gray = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # 2. Warping (Recalage t-1 vers t)
        h, w = depth_prev.shape
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + flow[..., 0]).astype(np.float32)
        map_y = (grid_y + flow[..., 1]).astype(np.float32)

        warped_prev_depth = cv2.remap(depth_prev, map_x, map_y, interpolation=cv2.INTER_LINEAR)

        # 3. Masque de validité (Exclusion des bords et occlusions)
        # Marge de 2 pixels pour éviter les artefacts d'interpolation
        mask = (map_x > 2) & (map_x < w-2) & (map_y > 2) & (map_y < h-2)
        
        if np.sum(mask) == 0: 
            return 0.0, 0.0

        # 4. Calcul des erreurs (Sur zone valide uniquement)
        
        # A. Warping Error (L1)
        # Moyenne des différences absolues sur les pixels valides
        l1_err = float(np.mean(np.abs(depth_curr[mask] - warped_prev_depth[mask])))
        
        # B. Temporal LPIPS (Perceptuel)
        # CORRECTION : On applique le masque AVANT d'envoyer au réseau de neurones
        # Sinon, LPIPS voit les bords noirs et renvoie une erreur énorme.
        d_curr_masked = depth_curr.copy()
        d_warp_masked = warped_prev_depth.copy()
        d_curr_masked[~mask] = 0
        d_warp_masked[~mask] = 0
        
        lpips_err = self._calculate_temporal_lpips(d_curr_masked, d_warp_masked)
        
        return l1_err, lpips_err
    
    @staticmethod
    def calculate_sharpness(depth_map: np.ndarray) -> float:
        """
        Mesure la netteté (Sharpness) de la carte de profondeur.
        Utilise la variance du Laplacien.
        
        Args:
            depth_map (np.ndarray): Carte de profondeur [H, W].
            
        Returns:
            float: Score de netteté.
            - Haut (> 500) : Bords très nets (Typique de Depth Anything V2).
            - Bas (< 100) : Image floue ou lisse (Typique de MiDaS ou modèles vidéo lissés).
        """
        if depth_map.dtype != np.uint8:
            depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            depth_vis = depth_map
            
        laplacian = cv2.Laplacian(depth_vis, cv2.CV_64F)
        return laplacian.var()
    
    @staticmethod
    def calculate_fidelity_rmse(depth_stabilized: np.ndarray, depth_raw: np.ndarray) -> float:
        """
        Mesure la fidélité par rapport au signal brut (RMSE).
        Permet de vérifier si la stabilisation ne s'éloigne pas trop de la réalité.
        
        Args:
            depth_stabilized (np.ndarray): Profondeur post-traitée.
            depth_raw (np.ndarray): Profondeur brute sortie du modèle.
            
        Returns:
            float: Erreur quadratique moyenne.
            - Haut : Lag important ou Ghosting (La stabilisation est trop agressive).
            - Bas : La stabilisation respecte la géométrie originale.
        """
        diff = (depth_stabilized - depth_raw) ** 2
        rmse = np.sqrt(np.mean(diff))
        return rmse

    @staticmethod
    def calculate_edge_alignment(frame: np.ndarray, depth_map: np.ndarray) -> float:
        """
        Mesure l'alignement des bords (Edge Alignment) entre RGB et Profondeur.
        Utilise la Magnitude du Gradient pour être invariant à l'orientation.
        
        Args:
            frame (np.ndarray): Image RGB [H, W, 3].
            depth_map (np.ndarray): Carte de profondeur [H, W].
            
        Returns:
            float: Coefficient de corrélation de Pearson [-1, 1].
            - > 0.5 (Excellent) : Les objets sont parfaitement détourés.
            - 0.1 - 0.4 (Moyen) : Bords flous ou légers décalages.
            - < 0 (Anomalie) : Hallucination de structure ou inversion de profondeur.
        """
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Gradients X et Y séparés (Crucial pour détecter toutes les orientations)
        sobel_x_img = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y_img = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
        mag_img = cv2.magnitude(sobel_x_img, sobel_y_img)
        
        sobel_x_depth = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y_depth = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)
        mag_depth = cv2.magnitude(sobel_x_depth, sobel_y_depth)
        
        flat_img = mag_img.flatten()
        flat_depth = mag_depth.flatten()
        
        # Sécurité division par zéro (Image unie)
        if np.std(flat_img) == 0 or np.std(flat_depth) == 0:
            return 0.0
            
        return float(np.corrcoef(flat_img, flat_depth)[0, 1])
    
    @staticmethod
    def calculate_avg_psd(depth_stack: np.ndarray, fs: float = 30.0, 
                      grid_size: int = 10, margin_ratio: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcule la Densité Spectrale de Puissance (PSD) moyenne sur une grille.
        C'est LA métrique fréquentielle pour quantifier le "Flickering Global".
        
        Args:
            depth_stack (np.ndarray): Pile temporelle de profondeur [Frames, H, W].
            fs (float): Fréquence d'échantillonnage (FPS de la vidéo).
            grid_size (int): Nombre de points de grille (ex: 10 => 10x10 = 100 points).
            margin_ratio (float): Pourcentage de bordure à ignorer (ex: 0.1 = 10%).
                                  Évite d'analyser les bords souvent bruités par le padding des CNN.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (Axe des Fréquences, PSD Moyenne).
            - Un pic d'énergie dans les hautes fréquences (>5Hz) indique un fort flickering.
        """
        T, H, W = depth_stack.shape
    
        margin_h = int(H * margin_ratio)
        margin_w = int(W * margin_ratio)
        
        # Sécurité dimensions
        if margin_h * 2 >= H or margin_w * 2 >= W:
            margin_h, margin_w = 0, 0
        
        # Grille d'échantillonnage (Zone centrale "Safe")
        y_coords = np.linspace(margin_h, H - margin_h, grid_size, dtype=int)
        x_coords = np.linspace(margin_w, W - margin_w, grid_size, dtype=int)
        
        all_psds = []
        freq_axis = None
        
        for y in y_coords:
            for x in x_coords:
                sig = depth_stack[:, y, x] # Extraction du signal temporel 1D
                # Welch est plus propre que FFT brute pour le bruit
                f, pxx = signal.welch(sig, fs=fs, nperseg=min(len(sig), 64))
                if len(pxx) > 0:
                    all_psds.append(pxx)
                    if freq_axis is None: freq_axis = f
                    
        if not all_psds: return np.array([]), np.array([])
        return freq_axis, np.mean(all_psds, axis=0)
    
    @staticmethod
    def extract_kymogram_slice(depth_stack: np.ndarray, axis: int = 0, coord: int = 360) -> np.ndarray:
        """
        Extrait une coupe spatio-temporelle (Kymogramme) pour visualisation.
        Permet de voir le "Jitter" temporel sous forme de dents de scie.
        
        Args:
            depth_stack (np.ndarray): Pile de profondeur [T, H, W].
            axis (int): 0 = Coupe Horizontale (fixe Y), 1 = Coupe Verticale (fixe X).
            coord (int): La coordonnée du pixel où couper.
            
        Returns:
            np.ndarray: Image 2D [Temps, Espace].
            - Lignes lisses = Stable.
            - Lignes hachées = Flickering.
        """
        if axis == 0:
            # Coupe horizontale : on regarde une ligne de pixels évoluer dans le temps
            return depth_stack[:, coord, :] # [T, W]
        # Coupe verticale
        return depth_stack[:, :, coord] # [T, H]