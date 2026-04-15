# 🧫 FHN-ColonyNet-Counting

A hybrid framework for automated bacterial colony detection and counting from petri dish imagery.

## Overview

FHN-ColonyNet-Counting enables automatic quantification of colony-forming units (CFUs) from bacterial imaging data through a hybrid framework combining two complementary algorithms. This approach maximizes segmentation performance while mitigating the limitations inherent to each individual method.

The complete application integrates both local and server-based processing via the FHN ColonyNet application, which manages CPSAM inference on a remote server and CARA execution locally.

**For technical issues or server access, contact:** justin.weiss@hotmail.ch

## Methodology

The framework combines two complementary segmentation approaches:

- **CPSAM**: A transformer-based segmentation model combining Cellpose and Segment Anything Model (SAM) architectures. Uses a SAM-based Vision Transformer encoder coupled with Cellpose-style flow field prediction for robust mask reconstruction.

- **CARA**: A classical computer vision approach consisting of HSV-based color filtering, watershed segmentation, and adaptive postprocessing. Provides robustness across varying imaging conditions.

Both methods operate in parallel with outputs combined via a hybrid strategy to achieve optimal results.

## Features

- ✅ Automated colony detection and counting
- ✅ Hybrid deep learning + classical CV pipeline
- ✅ Robust across varying image conditions
- ✅ Modular, extensible architecture
- ✅ Parallel processing for improved accuracy
- ✅ Server-based inference management

## Documentation

FHN-ColonyNet-Counting/ <br>
├── app.py                          # Main application - executes all algorithms <br>
├── CARA.py                         # Classical computer vision (CARA) implementation <br>
├── CPSAM.py                        # Transformer-based segmentation algorithm <br>
├── Hybrid.py                       # Hybrid algorithm combining CPSAM + CARA outputs <br>
├── Documentation.md                # Complete documentation: <br>
│                                   #   - Algorithm methodology <br>
│                                   #   - Performance metrics & analysis <br>
│                                   #   - Usage instructions <br>
│                                   #   - Implementation details  <br>
├── environment.yml                 # Conda Environment Setup <br>
├── README.md                       # Project overview <br>
├── input/                          # Input petri dish images <br>
└── output/                         # Generated counts & visualizations <br>

## Installation

TODO
