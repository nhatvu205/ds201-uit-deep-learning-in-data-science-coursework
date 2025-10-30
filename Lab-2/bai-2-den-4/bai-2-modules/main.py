import argparse
import os, sys
from data_module import build_dataloaders
from googlenet import build_googlenet
from train_eval import fit

DATA_DIR = "/content/vinafood21"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_dir",
        type=str,
        default=DATA_DIR
    )
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    train_loader, test_loader, num_classes, class_names = build_dataloaders(
        args.data_dir, batch_size=args.batch_size, image_size=args.img_size
    )

    model = build_googlenet(num_classes=num_classes)

    results = fit(model, train_loader, test_loader, epochs=args.epochs, lr=args.lr, device=args.device)

    print("Kết quả cuối cùng (macro):")
    print(
        f"Precision: {results['test_precision']:.4f} | Recall: {results['test_recall']:.4f} | F1: {results['test_f1']:.4f}"
    )


if __name__ == "__main__":
    main()


