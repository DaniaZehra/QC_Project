# QC_Project

Adapting Quantum Gated Recurrent GAN with Gaussian Uncertainty to Image Anomaly Detection

This repository contains the codebase and research paper details for adapting a state-of-the-art Network Anomaly Detection framework (QGRU-WGAN with Gaussian Uncertainty originally proposed by Hammami et al., 2025) into the visual domain for Video Anomaly Detection.

Historically tested on predictability-heavy, low-dimensional tabular telemetry datasets governed by thermodynamic laws (e.g., HAI dataset), this project studies the architectural bottlenecks, spatial-temporal feature scaling mechanics, and potential structural scalability drops encountered when transferring quantum machine learning pipelines to complex pixel-level anomaly recognition tasks.
📌 Project Overview & Intent

Anomaly detection in video frames involves computing high-dimensional spatial representations alongside temporal mechanics (motion variations). While classic tabular data can map straight into quantum circuits, video streaming data requires a deep two-pronged hybrid classical-quantum approach:

    Classical Routing: Leveraging deep feature extraction backbones (Pre-trained ResNet-18 and ResNet-50 models initialized with ImageNet) to compress raw target frames into low-dimensional latent space.

    Quantum Feature Optimization: Injecting feature mappings directly into a Hybrid Quantum Layer (HQL) containing a Variational Quantum Circuit (VQC) embedded into the update, reset, and candidate gates of an enhanced Quantum Gated Recurrent Unit (QGRU).

This sequence is then trained with a Wasserstein Generative Adversarial Network (WGAN) to utilize critic/generator uncertainty scores for precise frame-level anomaly detection.
🏗 Pipeline Architecture

The end-to-end framework downsamples, frames, encodes, and tracks video patterns via the pipeline depicted below:

UCSD Ped2 Video Frames
       │
       ▼
 ┌───────────┐
 │ ResNet-18 │  ──> GPU-accelerated spatial feature backbone
 └───────────┘
       │
       ▼
  [16-Dim Space] ──> Frame pixel data compressed to a 16-dimensional latent vector
       │
       ▼
 ┌───────────┐
 │ Windowing │  ──> Groups sequences into temporal windows (Size W=3)
 └───────────┘
       │
       ▼
 ┌───────────┐
 │  K-Means  │  ──> Cluster downsampling to keep representative normal samples
 └───────────┘
       │
       ▼
 ┌───────────┐
 │ QGRU-WGAN │  ──> VQC Quantum Layer updates + Critic Wasserstein loss loop
 └───────────┘
       │
       ▼
 ┌───────────┐
 │ Inference │  ──> Two-stage anomaly validation scoring (Interval gate + uncertainty)
 └───────────┘

📊 Dataset Detail

Experiments are systematically benchmarked using the UCSD Pedestrian 2 (Ped2) dataset:

    Description: Pedestrian footage captured via an elevated stationary camera viewpoint.

    Normal Behavior: Pedestrians moving along designated walkways.

    Anomalous Signatures: Sudden non-pedestrian objects (e.g., bicycles, skateboards, vehicles entering walkways) or erratic pedestrian motion violations.

The experiment downloads your target vectors dynamically through the Kaggle API right within the workspace setup.
🛠 Prerequisites & Installation

To run the pipeline notebooks locally or within cloud instances (Colab/Kaggle environments), configure the foundational Python environments and dependencies.
Dependencies

Install the quantum processing toolkits and lightning simulator environments:
Bash

pip install pennylane pennylane-lightning -q
pip install torch torchvision
pip install scikit-learn numpy pandas matplotlib

Dataset Access Setup

To fetch the training clips, make sure your Kaggle environment credentials are authenticated within your runtime script:
Python

import os
os.environ['KAGGLE_USERNAME'] = "your_kaggle_username"
os.environ['KAGGLE_KEY'] = "your_kaggle_api_key"

💻 Core Implementation Layout

The companion notebook structure handles processing natively via the following phases:

    Preprocessing Block: Downloads the structured UCSD dataset, loads pixel imagery, normalizes bounds using MinMax scaling, and reformats the tensors into continuous sequence blocks of shape (N, 3, 16).

    Computational Downsampling: Applies MiniBatchKMeans over frame centroids to dramatically lower quantum training footprint while maintaining crucial distribution parameters.

    Quantum HQL Structure: Initializes the pennylane simulator loop using default.qubit or lightning.qubit. Gates use quantum encodings like Angle, Double Angle, Phase, and Amplitude encoding.

    Adversarial Training Iteration: Executes the WGAN loops where the generator synthesizes nominal trajectories while the critic computes structural earth mover approximations.

📈 Summary of Research Findings

While the architecture demonstrates strong validity on simple regular structures, empirical results from the associated research paper indicate distinct challenges when applying it to vision tasks:

    Performance Drop: The adapted video model experienced a substantial drop in standard accuracy scores compared to its tabular network telemetry baselines.

    The Complexity Bottleneck: Network sensor telemetries follow rigid laws of thermodynamics, making data variations highly predictable. On the other hand, vision spaces introduce non-linear spatial dependencies and complex variations that pose a heavy challenge for modern NISQ-era variational circuits.
