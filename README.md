
---

# 🕶️ Monocular Video to Stereoscopic VR (Depth Stability Benchmark)

**Academic Project - Master AI & Applications (ISTIC - Univ. Rennes)**

## 📝 Project Overview

This project implements a computer vision pipeline to convert standard 2D videos into **stereoscopic 3D content** suitable for VR headsets.

Before generating the final 3D view, ensuring the **temporal consistency** of the depth maps is critical to prevent VR motion sickness. Therefore, **Phase 1** of this project focuses on a rigorous **Benchmarking Framework** comparing state-of-the-art foundation models (**Depth Anything V2** vs **Video Depth Anything**) and implementing advanced stabilization techniques.

## 🚀 Key Features

### Phase 1: Depth Estimation & Stability Analysis (Current)

* **Foundation Models Comparison:** Benchmarking **Depth Anything V2** (Image-based) vs **Video Depth Anything** (Video-native).
* **Scientific Metrics:**
* **Warping Error ():** Geometric consistency measurement using **RAFT** Optical Flow.
* **Flickering Analysis (PSD):** Power Spectral Density analysis (Welch's method) to detect high-frequency noise.
* **Temporal LPIPS:** Perceptual stability measurement.


* **Advanced Stabilization:** Implementation of a **Confidence-Based Stabilizer** fusing temporal reprojection and spatial edge consistency.

### Phase 2: 3D Generation (In Progress)

* **Depth-to-Disparity Transformation:** Converting metric depth to stereo disparity for VR.
* **Occlusion Handling:** Inpainting regions revealed by the stereoscopic shift.
* **Stereo Rendering:** Generating Side-by-Side (SBS) output.

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/zakary35/ProjetIA-2D-3D
cd ProjetIA-2D-3D

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies (ensure GPU support for RAFT/DAV2)
pip install -r requirements.txt

```

## 📂 Project Structure

```text
├── checkpoints/       # Model weights (DAV2, VDA, RAFT)
├── data/inputs/       # Input videos
├── benchmark_results/ # Generated reports (CSV, Plots, Videos)
├── src/
│   ├── depth_engine.py       # Wrapper for VDA and DAV2
│   ├── stabilizer.py         # Confidence-based stabilization logic
│   ├── metrics.py            # Scientific metrics (PSD, LPIPS, Warping)
│   ├── raft_optical_flow.py  # Optimized Optical Flow engine
│   └── warping.py            # (Upcoming) 3D Transformation logic
├── main.py            # CLI for single video processing
├── benchmark.py       # CLI for automated benchmarking
└── notebook/          # Experiments and visualizations

```

Vous avez raison, pour un README technique complet, il est important de lister **toutes** les options disponibles dans le script pour que l'utilisateur (ou le jury) sache exactement ce qu'il peut configurer.

Voici la section **"Usage"** mise à jour avec le tableau complet des arguments :

---

### 💻 Usage

#### 1. Depth Estimation & Stabilization (`main.py`)

Process a single video to generate a stabilized depth map and analyze its stability.

**Standard Run (GPU):**

```bash
python main.py --input data/inputs/video.mp4 --model dav2 --stabilize confidence

```

**Quick CPU Test (Debug):**

```bash
python main.py --input data/inputs/video.mp4 --model dav2 --size vits --limit 10 --device cpu

```

**Full Argument List:**

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--input` | `str` | **Required** | Path to the input video file (e.g., `data/video.mp4`). |
| `--output_dir` | `str` | `results` | Directory where the output video and metrics will be saved. |
| `--model` | `str` | `dav2` | Foundation model to use: `dav2` (Depth Anything V2) or `vda` (Video Depth Anything). |
| `--size` | `str` | `vitl` | Model size/complexity: `vits` (Small), `vitb` (Base), `vitl` (Large). |
| `--stabilize` | `str` | `raw` | Post-processing method: `raw` (None), `median`, `ema`, `confidence`. |
| `--limit` | `int` | `None` | Limit the number of frames to process (useful for quick debugging). |
| `--device` | `str` | `cuda` | Hardware accelerator: `cuda` (NVIDIA), `mps` (Mac), or `cpu`. |

#### 2. Automated Benchmark (`benchmark.py`)

Run a full comparative study (All models x All stabilization methods) and generate a report.

```bash
python benchmark.py --input data/inputs/video.mp4 --limit 200

```

**Outputs:**

* 📄 `final_scores.csv`: Detailed metrics table.
* 📊 `comparison_psd_all.png`: Spectral analysis of flickering.
* 🎥 Rendered videos for visual comparison.

## 📊 Understanding the Metrics

This project uses industrial-grade metrics to quantify stability:

1. **Average PSD (Power Spectral Density):** Detects **flickering**. A flat curve is ideal; peaks in high frequencies indicate noise.
2. **Warping Error:** Measures **geometric stability**. We warp frame  to  using Optical Flow and compare the depth maps. Lower is better.
3. **Edge Alignment:** Measures **contour precision**. Higher score (closer to 1.0) means depth edges align perfectly with RGB objects.

## 👥 Authors

This project is part of the Master IA & Applications curriculum at **ISTIC (Univ. Rennes 1)**.

* **Daniel Quenum** - *AI Engineer & Pipeline Lead* - [GitHub](https://github.com/danielquenum)
* **Elie Danigue** - *AI Engineer & Pipeline Lead*
* **Zakary Sadok** - *AI Engineer & Pipeline Lead*

*Made with ❤️ using PyTorch, Depth Anything V2, and RAFT.*