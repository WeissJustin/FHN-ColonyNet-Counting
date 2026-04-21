# 🧫 FHN ColonyNet
 
**Automated bacterial colony counting from petri dish images — no manual counting, ever again.**
 
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Contact](https://img.shields.io/badge/Contact-justin.weiss%40hotmail.ch-orange.svg)](mailto:justin.weiss@hotmail.ch)
 
<p align="center">
  <img src="/Extras/Logo.png" alt="FHN Logo" width="400"/>
</p>

---
 
## Problem
 
Manually counting colony-forming units (CFUs) on petri dishes is slow, subjective, and error-prone — yet it remains standard practice in microbiology labs. A single experiment can produce dozens of dishes; a single researcher can introduce counting bias that compounds across experiments.
 
## Solution
 
FHN ColonyNet automates CFU quantification from images using a **hybrid AI + classical vision pipeline**:
 
- **CPSAM** — a transformer-based segmentation model (Cellpose + SAM) fine-tuned on 290 bacterial colony images.
- **ColonyNet** — a classical HSV-based watershed segmentation algorithm for robustness under varied imaging conditions.
- **Hybrid combiner** — merges both outputs to maximize accuracy and minimize failure cases.
Everything runs through a single app: ColonyNet executes locally, CPSAM runs on a remote server via Tailscale.
 
---
 
## Performance (CPSAM, ColonyNet & Hybrid Approach on 645-dish test set)
 
| Condition | n | MAE | Bias | R² |
|---|---|---|---|---|
| CPSAM Full Dataset | 645 | 11.7 | 6.3 | 0.689 |
| CPSAM 20 outliers removed | 625 | 8.1 | 6.2 | 0.911 |
| ColonyNet Full Dataset | 645 | 11.7 | 6.3 | 0.689 |
| ColonyNet 20 outliers removed | 625 | 8.1 | 6.2 | 0.911 |
| Hybrid Full Dataset | 645 | 11.7 | 6.3 | 0.689 |
| Hybrid 20 outliers removed | 625 | 8.1 | 6.2 | 0.911 |

---
 
## Repo Structure
 
```
FHN-ColonyNet-Counting/
├── app.py              # Main application — orchestrates all algorithms
├── ColonyNet.py        # Classical computer vision pipeline
├── CPSAM.py            # Transformer-based segmentation (Cellpose-SAM)
├── Hybrid.py           # Combines CPSAM + CARA outputs
├── Documentation.md    # Full methodology, metrics, setup and usage guide
├── environment.yml     # Conda environment
├── README.md
├── input/              # Input petri dish images for reference
└── output/             # Colony counts and visualizations that were obtained with this pipeline
```
 
---
 
## Quick Start
 
### 1. Clone & install
 
```bash
git clone https://github.com/WeissJustin/FHN-ColonyNet-Counting.git
cd ColonyNet
conda env create -f environment.yml
conda activate colonynet
```
 
### 2. Connect to the CPSAM server (for AI inference)
 
Install [Tailscale](https://tailscale.com/download) (or via command line, see report above), then:
```bash
sudo tailscale up
# Sign in with provided credentials, then verify:
tailscale status  # server should appear at given IP.
```
 
> Contact [justin.weiss@hotmail.ch](mailto:justin.weiss@hotmail.ch) to request Tailscale access.
 
### 3. Run
 
```bash
python app.py
```
 
This will open the app.
 
---
 
## Imaging Setup
 
For reproducible results, the dishes should be captured with the following setup:
 
- **Camera**: MA500, F1.4, 3 MP, 5–50 mm objective, mounted top-down
- **Light board**: 66% brightness, 4400 K CCT, illuminating from below
- **Enclosure**: Dark fabric scaffold to block ambient light

More information and graphics of the setup in the Report that can be found above.
---
 
## Demo

![Demo](./Extras/animation.mp4)

---
 
## Acknowledgments
 
**CPSAM** builds on [Cellpose-SAM](https://www.biorxiv.org/content/10.1101/2025.04.28.651001v1) by Marius Pachitariu, Michael Rariden, and Carsen Stringer.
 
Developed as part of the FHN UROP program, 2026.
 
---
 
<div align="center">
Questions or issues → <a href="mailto:justin.weiss@hotmail.ch">justin.weiss@hotmail.ch</a>
</div>
 
