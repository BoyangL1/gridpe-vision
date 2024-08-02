# GridPE: Unifying Positional Encoding in Transformers with a Grid Cell-Inspired Framework

This repository contains the implementation of the GridPE framework, which aims to unify positional encoding in transformers using a grid cell-inspired approach. This work is a continuation and extension of existing research in the field of vision transformers and positional encoding.

## Introduction

GridPE is a novel framework designed to unify positional encoding in transformers by drawing inspiration from the biological grid cells in mammalian brains. This repository contains the implementation and experiments that demonstrate the efficacy of GridPE in various transformer architectures.

## Acknowledgments

Our work builds on the following key repositories:

### [ViT-PyTorch](https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py)

The ViT-PyTorch repository provides a robust and flexible implementation of Vision Transformers (ViT) and its variants, including PiT and DeepViT. This codebase has been instrumental in enabling us to experiment with and extend positional encoding mechanisms within transformer architectures.

### [Twins](https://github.com/Meituan-AutoML/Twins)

The Twins repository offers state-of-the-art transformer architectures with hierarchical structures, enhancing the ability of transformers to model vision data effectively. The Twins model has provided a strong baseline for comparing and integrating our GridPE framework.