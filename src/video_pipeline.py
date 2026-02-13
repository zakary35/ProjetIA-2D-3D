import cv2
import numpy as np
import torch
import time
from tqdm import tqdm
from typing import Dict, Any, Optional

# Imports de vos modules
from .depth_engine import VDAEngine, DAV2Engine
from .metrics import StabilityMetrics
from .stabilizer import DepthStabilizer
from .raft_optical_flow import RAFTFlowEngine

class VideoPipeline:
    """
    Pipeline de traitement vidéo complet : Lecture -> Inférence -> Stabilisation -> Métriques -> Écriture.
    """

    def __init__(self, 
                 input_path: str, 
                 output_path: str, 
                 model_type: str = 'dav2', 
                 model_size: str = 'vitl',
                 device: str = 'cuda',
                 stabilize_method: str = 'raw',
                 limit_frames: Optional[int] = None):
        
        self.input_path = input_path
        self.output_path = output_path
        self.limit_frames = limit_frames
        self.device = device
        
        print(f"🔧 Initialisation du Pipeline : {model_type.upper()}-{model_size} | Stab: {stabilize_method}")

        # 1. Chargement du Moteur de Profondeur
        if model_type == 'vda':
            self.engine = VDAEngine(model_size=model_size, device=device)
        elif model_type == 'dav2':
            self.engine = DAV2Engine(model_size=model_size, device=device)
        else:
            raise ValueError(f"Modèle inconnu : {model_type}")

        # 2. Chargement des Outils
        self.metrics = StabilityMetrics(device=device, use_raft=True)
        self.stabilizer = DepthStabilizer(method=stabilize_method)
        
        # 3. État interne (State)
        self.prev_frame = None
        self.prev_depth = None
        
        # 4. Stockage Métriques (Optimisé mémoire)
        self.stats = {
            'processing_time': [],
            'warping_error': [],
            'temporal_lpips': [],
            'edge_alignment': [],
            'probes_psd': [] # Stockera seulement une grille de pixels [Frames, 100]
        }
        self.probe_coords = None # Sera défini à la frame 1

    def _normalize_depth(self, depth: np.ndarray) -> np.ndarray:
        """Normalise la profondeur en [0, 1] pour la comparabilité."""
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-6:
            return (depth - d_min) / (d_max - d_min)
        return np.zeros_like(depth)

    def _colorize_depth(self, depth: np.ndarray) -> np.ndarray:
        """Convertit la depth [0, 1] en image couleur (JET) pour la vidéo."""
        depth_uint8 = (depth * 255).astype(np.uint8)
        return cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)

    def run(self):
        """Exécute le traitement vidéo."""
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Impossible d'ouvrir {self.input_path}")

        # Infos Vidéo
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Gestion de la limite
        if self.limit_frames is not None:
            total_frames = min(total_frames, self.limit_frames)
            print(f"⚠️ Mode Test : Limite fixée à {total_frames} frames.")

        # Writer (Vidéo côte à côte : RGB | Depth)
        out = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (orig_w * 2, orig_h))

        pbar = tqdm(total=total_frames, desc="Traitement", unit="frame")
        frame_idx = 0

        while cap.isOpened():
            if self.limit_frames and frame_idx >= self.limit_frames:
                break

            ret, frame = cap.read()
            if not ret: break

            t0 = time.time()

            # 1. Inférence
            raw_depth = self.engine.infer(frame) # [H, W] float32 non borné
            
            # 2. Calcul du Flux Optique (Si dispo, pour guider stab et métriques)
            flow = None
            if self.prev_frame is not None and self.metrics.raft_engine:
                # On utilise RAFT pour tout le monde (efficacité)
                flow = self.metrics.raft_engine.compute_flow(self.prev_frame, frame)

            # 3. Stabilisation (Optionnelle)
            # On passe le 'flow' pour éviter que le stabilisateur ne recalcule Farneback
            depth = self.stabilizer.apply(raw_depth, frame, flow=flow)
            
            # 4. Normalisation [0, 1] (Critique pour métriques et visu)
            depth_norm = self._normalize_depth(depth)

            # 5. Calcul Métriques (Streaming)
            if self.prev_depth is not None and self.prev_frame is not None:
                # A. Flickering (Warping + LPIPS)
                l1, lpips_score = self.metrics.calculate_flickering_error(
                    depth_norm, self.prev_depth, frame, self.prev_frame
                )
                if l1 is not None: # Si pas de cut
                    self.stats['warping_error'].append(l1)
                    self.stats['temporal_lpips'].append(lpips_score)

            # B. Edge Alignment (Intra-frame)
            edge_score = self.metrics.calculate_edge_alignment(frame, depth_norm)
            self.stats['edge_alignment'].append(edge_score)

            # C. Sondes pour PSD (Mémoire légère)
            if self.probe_coords is None:
                # Grille 10x10 au centre
                h, w = depth.shape
                ys = np.linspace(h//10, h-h//10, 10).astype(int)
                xs = np.linspace(w//10, w-w//10, 10).astype(int)
                grid_y, grid_x = np.meshgrid(ys, xs)
                self.probe_coords = (grid_y.flatten(), grid_x.flatten())
            
            py, px = self.probe_coords
            self.stats['probes_psd'].append(depth_norm[py, px])

            # 6. Écriture Vidéo
            depth_color = self._colorize_depth(depth_norm)
            combined = np.hstack((frame, depth_color))
            out.write(combined)

            # Mise à jour états
            self.prev_frame = frame.copy()
            self.prev_depth = depth_norm.copy()
            self.stats['processing_time'].append(time.time() - t0)

            frame_idx += 1
            pbar.update(1)

        cap.release()
        out.release()
        pbar.close()

        print(f"✅ Traitement terminé. Vidéo sauvegardée : {self.output_path}")
        return self._finalize_stats(fps)

    def _finalize_stats(self, fps) -> Dict[str, Any]:
        """Post-traitement des statistiques (PSD, Moyennes)."""
        print("📊 Calcul des métriques globales (PSD)...")
        
        # Calcul PSD sur les sondes
        # probes_stack: [Frames, 100]
        probes_stack = np.array(self.stats['probes_psd']) 
        # On transpose pour avoir [100, Frames] compatible avec Welch
        freqs, avg_psd = self.metrics.calculate_avg_psd(
            probes_stack.T.reshape(-1, len(probes_stack), 1), # Hack dimensions pour votre fonction
            fs=fps, 
            grid_size=10
        )
        
        # Pour le rapport final
        results = {
            'fps_process': 1.0 / np.mean(self.stats['processing_time']),
            'warping_error_mean': np.mean(self.stats['warping_error']),
            'lpips_mean': np.mean(self.stats['temporal_lpips']),
            'edge_alignment_mean': np.mean(self.stats['edge_alignment']),
            'psd_data': {'freqs': freqs.tolist(), 'power': avg_psd.tolist()}
        }
        
        return results