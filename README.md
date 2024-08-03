# GridPE: Unifying Positional Encoding in Transformers with a Grid Cell-Inspired Framework

This repository contains the implementation of the GridPE framework, which aims to unify positional encoding in transformers using a grid cell-inspired approach. This work is a continuation and extension of existing research in the field of vision transformers and positional encoding.

## Introduction

GridPE is a novel framework designed to unify positional encoding in transformers by drawing inspiration from the biological grid cells in mammalian brains. This repository contains the implementation and experiments that demonstrate the efficacy of GridPE in various transformer architectures.

## Main Components

- **Twins_gridPE**: Contains the implementation of GridPE applied to the Twins transformer architecture. This includes data processing, training scripts, and specific utility functions.
- **deepvit_gridPE**: Contains the implementation of GridPE applied to the DeepViT architecture.
- **gridPE**: Core implementation of the GridPE framework, including grid-based positional encoding and related utilities.
- **pit_gridPE**: Contains the implementation of GridPE applied to the PiT transformer architecture.
- **vit_gridPE**: Contains the implementation of GridPE applied to the Vision Transformer (ViT) architecture.
- **test_*.py**: Test scripts for evaluating the GridPE framework across different transformer models.
- **main_*.py**: Main scripts to run experiments with different transformer architectures.
- **utils.py**: Utility functions used across different scripts.

## Data Preparation

For our experiments, we use the ImageNet100 dataset, a subset of the larger ImageNet dataset, containing 100 classes. This dataset is well-suited for testing and prototyping due to its reduced size while still maintaining a diverse set of images.

### Preparing the Dataset

1. **Download ImageNet100**: If you do not have access to ImageNet100 directly, you can create it by selecting a subset of 100 classes from the full ImageNet dataset. Ensure that you have the necessary permissions and follow the dataset's usage guidelines.

2. **Organize the Dataset**: Place the dataset in a directory structure that matches the expected format for the dataloaders in our codebase:

```
imagenet100/
├── class1/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── …
├── class2/
└── …
```

## Example Command

To train a model using the ImageNet100 dataset with specific hyperparameters, you can use the following command:

```bash
python main_model.py --model_type model_name --image_size 256 --patch_size 16 --num_classes 100 --dim 256 --depth 6 --heads 8 --mlp_dim 2048 --dropout 0.1 --emb_dropout 0.1 --batch_size 32 --epochs 100 --lr 0.0001 --data_path "/path/to/dataset"
```

## Acknowledgments

Our work builds on the following key repositories, and we extend our sincere thanks to the authors for their valuable contributions:

### [ViT-PyTorch](https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py)

The ViT-PyTorch repository provides a robust and flexible implementation of Vision Transformers (ViT) and its variants, including PiT and DeepViT. This codebase has been instrumental in enabling us to experiment with and extend positional encoding mechanisms within transformer architectures.

### [Twins](https://github.com/Meituan-AutoML/Twins)

The Twins repository offers state-of-the-art transformer architectures with hierarchical structures, enhancing the ability of transformers to model vision data effectively. The Twins model has provided a strong baseline for comparing and integrating our GridPE framework.