# Online Handwritten Character Recognition with Rotation

This repository contains the implementation of an online handwritten character recognition framework designed for rotated handwriting inputs.

## Model Overview

The framework is divided into two parts:

- **JAX-based module**: used for trajectory preprocessing and sliding window path signature feature extraction.
- **PyTorch-based module**: used for sequence classification and final character recognition.

By combining JAX for efficient feature computation and PyTorch for flexible model training, the method improves robustness to rotation while preserving local stroke structure and sequential dependencies.

## Supported Tasks

- Digit recognition
- Uppercase English letter recognition
- Chinese radical recognition

## Environment

- JAX 0.4.30 / jaxlib 0.4.30
- PyTorch 2.5.1 + CUDA 12.1

Core libraries include:

- JAX / Flax / Optax / Equinox / Diffrax
- Torch / Torchvision / Accelerate
- Path signature tools: signax / esig / iisignature

## Note

Some components run in a JAX environment, while others run in a PyTorch environment.
