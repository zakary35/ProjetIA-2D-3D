import argparse
import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Import du pipeline
from src.video_pipeline import VideoPipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmarking Depth Temporal Stability")
    
    # Entrées / Sorties
    parser.add_argument('--input', type=str, required=True, help="Chemin vidéo entrée")
    parser.add_argument('--output_dir', type=str, default="results", help="Dossier sortie")
    
    # Configuration Modèle
    parser.add_argument('--model', type=str, default='dav2', choices=['dav2', 'vda'], help="Type de modèle")
    parser.add_argument('--size', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'], help="Taille modèle")
    
    # Options Test
    parser.add_argument('--limit', type=int, default=None, help="Limiter le nombre de frames (Test rapide)")
    parser.add_argument('--stabilize', type=str, default='raw', choices=['raw', 'median', 'ema', 'confidence'], help="Méthode de stabilisation")
    parser.add_argument('--device', type=str, default='cuda', help="Device (cuda/cpu)")

    return parser.parse_args()

def save_psd_plot(freqs, power, output_path, title="Flickering Analysis (PSD)"):
    """Génère le graphique PSD."""
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, power, label='Temporal Noise', color='blue', linewidth=2)
    plt.xlabel('Fréquence (Hz)')
    plt.ylabel('Puissance Spectrale (Log)')
    plt.yscale('log') # Échelle log indispensable pour voir le flickering
    plt.title(title)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.savefig(output_path)
    plt.close()
    print(f"📈 Graphique PSD sauvegardé : {output_path}")

def main():
    args = parse_args()
    
    # Création dossier
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Nom du fichier de sortie
    video_name = os.path.splitext(os.path.basename(args.input))[0]
    output_filename = f"{video_name}_{args.model}_{args.size}_{args.stabilize}.mp4"
    output_path = os.path.join(args.output_dir, output_filename)
    
    # Lancement du Pipeline
    pipeline = VideoPipeline(
        input_path=args.input,
        output_path=output_path,
        model_type=args.model,
        model_size=args.size,
        device=args.device,
        stabilize_method=args.stabilize,
        limit_frames=args.limit
    )
    
    stats = pipeline.run()
    
    # Sauvegarde des stats (JSON)
    json_path = output_path.replace('.mp4', '_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=4)
    
    # Génération Graphique
    psd_path = output_path.replace('.mp4', '_psd.png')
    save_psd_plot(
        np.array(stats['psd_data']['freqs']), 
        np.array(stats['psd_data']['power']), 
        psd_path,
        title=f"PSD - {args.model.upper()} ({args.stabilize})"
    )
    
    print("\n" + "="*40)
    print(f"RESULTATS FINAUX ({args.model.upper()})")
    print("="*40)
    print(f"Stability (Warp Error) : {stats['warping_error_mean']:.5f} (Plus bas = Mieux)")
    print(f"Perceptual (LPIPS)     : {stats['lpips_mean']:.5f} (Plus bas = Mieux)")
    print(f"Edges (Alignment)      : {stats['edge_alignment_mean']:.3f} (Plus haut = Mieux)")
    print("="*40)

if __name__ == "__main__":
    main()