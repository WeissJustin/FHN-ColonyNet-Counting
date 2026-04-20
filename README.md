<div align="center">
# 🧫 FHN ColonyNet
 
**Automated bacterial colony counting from petri dish images — no manual counting, ever again.**
 
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Contact](https://img.shields.io/badge/Contact-justin.weiss%40hotmail.ch-orange.svg)](mailto:justin.weiss@hotmail.ch)
 
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/20/FHN_Nordwestschweiz_Logo.svg/320px-FHN_Nordwestschweiz_Logo.svg.png" alt="FHN Logo" width="200"/>
</div>
---
 
## Problem
 
Manually counting colony-forming units (CFUs) on petri dishes is slow, subjective, and error-prone — yet it remains standard practice in microbiology labs. A single experiment can produce dozens of dishes; a single researcher can introduce counting bias that compounds across experiments.
 
## Solution
 
FHN ColonyNet automates CFU quantification from images using a **hybrid AI + classical vision pipeline**:
 
- **CPSAM** — a transformer-based segmentation model (Cellpose + SAM) fine-tuned on 290 bacterial colony images, achieving R²=0.91 on well-behaved samples after outlier removal
- **CARA** — a classical HSV-based watershed segmentation algorithm for robustness under varied imaging conditions
- **Hybrid combiner** — merges both outputs to maximize accuracy and minimize failure cases
Everything runs through a single app: CARA executes locally, CPSAM runs on a remote server via Tailscale.
 
---
 
## Performance (CPSAM on 645-dish test set)
 
| Condition | n | MAE | Bias | R² |
|---|---|---|---|---|
| Full dataset | 645 | 11.7 | 6.3 | 0.689 |
| 20 outliers removed | 625 | 8.1 | 6.2 | 0.911 |
 
---
 
## Repo Structure
 
```
FHN-ColonyNet-Counting/
├── app.py              # Main application — orchestrates all algorithms
├── CARA.py             # Classical computer vision pipeline
├── CPSAM.py            # Transformer-based segmentation (Cellpose-SAM)
├── Hybrid.py           # Combines CPSAM + CARA outputs
├── Documentation.md    # Full methodology, metrics, and usage guide
├── environment.yml     # Conda environment
├── README.md
├── input/              # Input petri dish images
└── output/             # Colony counts and visualizations
```
 
---
 
## Quick Start
 
### 1. Clone & install
 
```bash
git clone https://github.com/FHN/ColonyNet.git
cd ColonyNet
conda env create -f environment.yml
conda activate colonynet
```
 
### 2. Connect to the CPSAM server (for AI inference)
 
Install [Tailscale](https://tailscale.com/download), then:
 
```bash
sudo tailscale up
# Sign in with provided credentials, then verify:
tailscale status  # server should appear at 100.111.18.10
```
 
> Contact [justin.weiss@hotmail.ch](mailto:justin.weiss@hotmail.ch) to request Tailscale access.
 
### 3. Run
 
```bash
python app.py --input input/ --output output/
```
 
Results are saved to `output/` as overlay images and per-colony CSV files.
 
---
 
## Imaging Setup
 
For reproducible results, use the documented imaging rig:
 
- **Camera**: MA500, F1.4, 3 MP, 5–50 mm objective, mounted top-down
- **Light board**: 66% brightness, 4400 K CCT, illuminating from below
- **Enclosure**: Dark fabric scaffold to block ambient light
---
 
## Demo
 
> 🎥 *Demo video coming soon — will show full pipeline from raw image to colony count overlay.*
 
---
 
## Roadmap
 
- [ ] CARA section documentation (in progress)
- [ ] Hybrid algorithm documentation (in progress)
- [ ] Reduce systematic undercounting bias (current bias ≈ 6 colonies)
- [ ] Expand fine-tuning dataset beyond 290 images
- [ ] Offline CPSAM mode (no server dependency)
- [ ] iPad annotation interface documentation
---
 
## Acknowledgments
 
**CPSAM** builds on [Cellpose-SAM](https://www.biorxiv.org/content/10.1101/2025.04.28.651001v1) by Marius Pachitariu, Michael Rariden, and Carsen Stringer.
 
Developed as part of the FHN UROP program, 2026.
 
---
 
<div align="center">
Questions or issues → <a href="mailto:justin.weiss@hotmail.ch">justin.weiss@hotmail.ch</a>
</div>
 
