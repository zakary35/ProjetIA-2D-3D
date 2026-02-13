import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Any

# Import du pipeline unitaire
from src.video_pipeline import VideoPipeline

class BenchmarkRunner:
    """
    Orchestrateur de tests pour comparer différents modèles et méthodes de stabilisation.
    Génère un rapport complet (CSV + Graphiques).
    """

    # Configurations à tester
    MODELS: List[str] = ['dav2', 'vda']
    METHODS: List[str] = ['raw', 'median', 'ema', 'confidence']
    
    def __init__(self, input_video: str, output_dir: str = "benchmark_results", limit_frames: int = None, device: str = 'cuda'):
        """
        Initialise le banc de test.

        Args:
            input_video (str): Chemin de la vidéo source.
            output_dir (str): Dossier où tout sera sauvegardé.
            limit_frames (int, optional): Nombre de frames pour un test rapide.
            device (str): 'cuda' ou 'cpu'.
        """
        self.input_video = input_video
        self.output_dir = output_dir
        self.limit_frames = limit_frames
        self.device = device
        self.results_data: List[Dict[str, Any]] = []
        
        # Création du dossier
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Configuration graphique pro
        sns.set_theme(style="whitegrid")

    def run_all(self):
        """Lance la boucle d'expérimentation sur toutes les combinaisons."""
        print(f"🚀 Démarrage du Benchmark Complet sur {self.device.upper()}...")
        print(f"📂 Résultats dans : {self.output_dir}")

        for model in self.MODELS:
            for method in self.METHODS:
                
                # Cas particulier : VDA a déjà une cohérence temporelle. 
                # On peut vouloir tester 'raw' seulement, mais pour la science, testons tout.
                experiment_name = f"{model.upper()}_{method}"
                print(f"\n🧪 Test en cours : {experiment_name}")
                
                # Définition du fichier de sortie vidéo
                video_out = os.path.join(self.output_dir, f"{experiment_name}.mp4")
                
                # Instanciation et exécution du pipeline
                # On capture les erreurs pour ne pas arrêter le benchmark si un modèle plante
                try:
                    pipeline = VideoPipeline(
                        input_path=self.input_video,
                        output_path=video_out,
                        model_type=model,
                        model_size='vitl', # On garde le meilleur modèle pour le comparatif
                        device=self.device,
                        stabilize_method=method,
                        limit_frames=self.limit_frames
                    )
                    
                    stats = pipeline.run()
                    
                    # Enregistrement des données brutes
                    entry = {
                        'Model': model.upper(),
                        'Method': method.capitalize(),
                        'Configuration': experiment_name,
                        'Warp Error (L1)': stats['warping_error_mean'],
                        'LPIPS (Perceptual)': stats['lpips_mean'],
                        'Edge Alignment': stats['edge_alignment_mean'],
                        'FPS': stats['fps_process'],
                        # On garde les données PSD brutes pour le graph global
                        'psd_freqs': stats['psd_data']['freqs'],
                        'psd_power': stats['psd_data']['power']
                    }
                    self.results_data.append(entry)
                    
                except Exception as e:
                    print(f"❌ Erreur sur {experiment_name} : {e}")
                    continue

    def generate_report(self):
        """Génère le fichier CSV et les graphiques comparatifs."""
        if not self.results_data:
            print("⚠️ Aucun résultat à analyser.")
            return

        # 1. Création du DataFrame Pandas
        df = pd.DataFrame(self.results_data)
        csv_path = os.path.join(self.output_dir, "final_scores.csv")
        df.to_csv(csv_path, index=False)
        print(f"📄 Tableau des scores sauvegardé : {csv_path}")
        
        # On retire les colonnes PSD pour l'affichage console
        print("\n=== RÉSUMÉ DES SCORES ===")
        print(df.drop(columns=['psd_freqs', 'psd_power']))

        # 2. Graphiques en Barres (Scores)
        self._plot_metric_comparison(df, 'Warp Error (L1)', 'Stabilité Géométrique (Plus bas = Mieux)')
        self._plot_metric_comparison(df, 'LPIPS (Perceptual)', 'Stabilité Perceptuelle (Plus bas = Mieux)')
        self._plot_metric_comparison(df, 'Edge Alignment', 'Respect des Bords (Plus haut = Mieux)')
        
        # 3. Graphique PSD Comparatif (Multi-Line)
        self._plot_combined_psd()

    def _plot_metric_comparison(self, df: pd.DataFrame, metric: str, title: str):
        """Génère un bar chart groupé."""
        plt.figure(figsize=(10, 6))
        
        # Bar chart : X=Method, Y=Metric, Hue=Model
        sns.barplot(data=df, x='Method', y=metric, hue='Model', palette="viridis")
        
        plt.title(title)
        plt.ylabel(metric)
        plt.xlabel("Méthode de Post-Traitement")
        plt.legend(title='Modèle')
        
        filename = f"comparison_{metric.split()[0].lower()}.png"
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()

    def _plot_combined_psd(self):
        """Génère un graphique spectral superposant toutes les courbes."""
        plt.figure(figsize=(12, 8))
        
        for entry in self.results_data:
            freqs = np.array(entry['psd_freqs'])
            power = np.array(entry['psd_power'])
            label = entry['Configuration']
            
            # Style de ligne : Pointillé pour VDA, Plein pour DAV2 (pour distinguer)
            linestyle = '--' if 'VDA' in label else '-'
            
            plt.plot(freqs, power, label=label, linestyle=linestyle, alpha=0.8, linewidth=1.5)

        plt.title("Analyse Spectrale du Flickering (PSD Global)")
        plt.xlabel("Fréquence (Hz)")
        plt.ylabel("Puissance du Bruit (Log)")
        plt.yscale('log') # Indispensable pour voir les différences
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # Légende à l'extérieur
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.output_dir, "comparison_psd_all.png"))
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Automatique DAV2 vs VDA")
    parser.add_argument('--input', type=str, required=True, help="Vidéo d'entrée")
    parser.add_argument('--limit', type=int, default=None, help="Limite de frames (ex: 100 pour test)")
    
    args = parser.parse_args()
    
    # Lancement
    bench = BenchmarkRunner(input_video=args.input, limit_frames=args.limit)
    bench.run_all()
    bench.generate_report()