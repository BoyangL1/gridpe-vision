import torch
from pit_gridPE.pit import PiT,PiTRotate,PiTComplex,PiTDeep,PiTMerge


def main():
    v = PiTMerge(
        image_size=256,
        patch_size=32,
        dim=256,
        num_classes=1000,
        depth=(
            3,
            3,
            3,
        ),
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1,
    )

    img = torch.randn(1, 3, 256, 256)

    preds = v(img)
    assert preds.shape == (1, 1000), "correct logits outputted"


if __name__ == "__main__":
    main()
