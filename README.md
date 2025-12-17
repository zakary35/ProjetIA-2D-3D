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
git clone https://github.com/zakary35/ProjetIA-2D-3D
cd ProjetIA-2D-3D

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

## 📂 Project Structure
This repository is organized as follows:

- `src/`: Contains the source code for the pipeline.
  - `depth_engine.py`: Loads the Depth Anything V2 model.
  - `warping.py`: Handles the 2D-to-3D geometric transformation.
- `notebooks/`: Jupyter notebooks for experiments and visualizations.
- `data/`: Folder to store input videos (ignored by Git).
- `assets/`: Images and GIFs used in this README.

## 👥 Authors
This project is part of the Master IA & Applications curriculum at **ISTIC (Univ. Rennes 1)**.

- **[VOTRE NOM]** - *AI Engineer & Pipeline Lead* - [Lien vers votre GitHub]
- **[NOM DU CAMARADE]** - *Rôle (ex: Data Processing)*
- **[NOM DU CAMARADE]** - *Rôle (ex: VR Integration)*

---
*Made with ❤️ using PyTorch and Depth Anything V2.*
