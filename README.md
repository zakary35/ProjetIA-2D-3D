# Monocular Video to Stereoscopic VR (Depth Anything V2)

![Python](https://img.shields.io/badge/Python-3.10-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red) ![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

**Academic Project - Master AI & Applications (ISTIC - Univ. Rennes)**

## 📝 Project Overview
This project implements a computer vision pipeline to convert standard 2D videos into **stereoscopic 3D content** suitable for VR headsets.
It leverages **Depth Anything V2** (Foundation Model) to estimate temporal-consistent depth maps and applies geometric warping to generate the second eye view.

## 🚀 Key Features
- **State-of-the-art Depth Estimation** using Vision Transformers (DINOv2 backbone).
- **Temporal Consistency Analysis** on dynamic scenes (Anime, Cartoons, Real-world).
- **Depth-to-Disparity Transformation** for VR rendering.
- **Occlusion Handling** (Inpainting) for artifact reduction.

## 🛠️ Installation

```bash
# Clone the repository
git clone [URL_DE_VOTRE_REPO]
cd Monocular-Video-to-VR

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
