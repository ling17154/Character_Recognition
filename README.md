# Rotation-free Online Handwritten Character Recognition Using Linear Recurrent Units

This repository contains an implementation of an online handwritten character recognition framework specifically designed for rotation-based online handwritten input.

## Model Overview

The framework consists of two parts:

- **Preprocessing Module**: Used for trajectory preprocessing and sliding window path feature extraction.

- **Classification Module**: Used for sequence classification and final character recognition.

Preprocessing mainly includes feature extraction via dangling normalization, feature construction, and sliding window path symbolization for downstream classification tasks.

## Dataset

Experiments were conducted on three subsets of CASIA-OLHWDB1.1: digits (10 classes), uppercase English letters (26 classes), and Chinese radicals (52 classes). Each class contains approximately 300 samples, with 240 used for training and 60 for testing. During training, we applied random global rotations, affine perturbations, and elastic deformation for data augmentation. During testing, each sample was rotated at 12° intervals, generating 30 rotation variants per sample.

## Environment

- JAX 0.4.30 / jaxlib 0.4.30

- PyTorch 2.5.1 + CUDA 12.1

Core libraries include:

- JAX / Flax / Optax / Equinox / Diffrax

- Torch / Torchvision

- Path signature tools: signax / esig / iisignature

## Note

LRU, S5, NCDE, and log-NCDE run in the JAX environment; other models run in the PyTorch environment.
