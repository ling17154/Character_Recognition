# Rotation-free Online Handwritten Character Recognition Using Linear Recurrent Units

**Accepted by ICPR 2026**  
Paper: <a href="https://arxiv.org/abs/2602.01533" title="https://arxiv.org/abs/2602.01533" target="_blank"><img src="/images/ext/file.png" alt="" style="width: 32px; height: 32px; vertical-align: middle;"></a>

This repository contains an implementation of an online handwritten character recognition framework designed for rotation-robust online handwritten input.

## Model Overview

The framework consists of two parts:

- **Preprocessing Module**: used for trajectory preprocessing and sliding window path signature feature extraction.
- **Classification Module**: used for sequence classification and final character recognition.

The preprocessing pipeline mainly includes hanging normalization, dynamic feature construction, and sliding window path signature extraction for downstream classification.

## Dataset

Dataset: https://doi.org/10.6084/m9.figshare.30759515

Experiments were conducted on three subsets of CASIA-OLHWDB1.1: digits (10 classes), uppercase English letters (26 classes), and Chinese radicals (52 classes). Each class contains approximately 300 samples, with 240 used for training and 60 for testing. During training, random global rotations, affine perturbations, and elastic distortions were applied for data augmentation. During testing, each sample was rotated at 12° intervals, generating 30 rotated variants per sample.

## Environment

- JAX 0.4.30 / jaxlib 0.4.30
- PyTorch 2.5.1 + CUDA 12.1

Core libraries include:

- JAX / Flax / Optax / Equinox / Diffrax
- Torch / Torchvision
- Path signature tools: signax / esig / iisignature

## Note

LRU, S5, NCDE, and Log-NCDE are implemented in the JAX environment, while the other models are implemented in the PyTorch environment.
