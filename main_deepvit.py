import argparse
from deepvit_gridPE.deepvit import DeepViT, DeepViTComplex, DeepViTMerge, DeepViTRotate, DeepViTDeep
from utils import *

def main_imagenet(args):
    model_dict = {
        'DeepViT': DeepViT,
        'DeepViTComplex': DeepViTComplex,
        'DeepViTMerge': DeepViTMerge,
        'DeepViTRotate': DeepViTRotate,
        'DeepViTDeep': DeepViTDeep
    }

    ModelClass = model_dict[args.model_type]

    # Initialize model with command-line parameters
    v = ModelClass(
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_classes=args.num_classes,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout,
        emb_dropout=args.emb_dropout
    )

    # Load data and split into training and validation sets
    train_loader, val_loader = load_data(args.data_path, args.image_size, batch_size=args.batch_size)

    # Train and validate model
    train_and_validate(v, train_loader, val_loader, epochs=args.epochs, lr=args.lr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DeepViT model on ImageNet100")

    # Add arguments for model hyperparameters without defaults
    parser.add_argument("--model_type", type=str, choices=['DeepViT', 'DeepViTComplex', 'DeepViTMerge', 'DeepViTRotate', 'DeepViTDeep'], required=True, help="Type of DeepViT model to use")
    parser.add_argument("--image_size", type=int, required=True, help="Input image size")
    parser.add_argument("--patch_size", type=int, required=True, help="Patch size")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of classes")
    parser.add_argument("--dim", type=int, required=True, help="Dimension of the model")
    parser.add_argument("--depth", type=int, required=True, help="Depth of the transformer")
    parser.add_argument("--heads", type=int, required=True, help="Number of attention heads")
    parser.add_argument("--mlp_dim", type=int, required=True, help="Dimension of the MLP layer")
    parser.add_argument("--dropout", type=float, required=True, help="Dropout rate")
    parser.add_argument("--emb_dropout", type=float, required=True, help="Embedding dropout rate")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size")
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset")

    args = parser.parse_args()

    main_imagenet(args)