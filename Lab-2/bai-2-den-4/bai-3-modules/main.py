import argparse
import torch
from data_module import build_dataloaders
from resnet18 import build_resnet18
from train_eval import fit
from visualize import plot_training_history, show_predictions

DATA_DIR = "/content/vinafood21"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default=DATA_DIR)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--no_viz", action="store_true", help="Tắt visualization")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    train_loader, test_loader, num_classes, class_names = build_dataloaders(
        args.data_dir, batch_size=args.batch_size, image_size=args.img_size
    )
    print(f"Số lớp: {num_classes}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    model = build_resnet18(num_classes=num_classes)

    results, history = fit(model, train_loader, test_loader, epochs=args.epochs, lr=args.lr, device=args.device)

    print("Kết quả cuối cùng (macro):")
    print(
        f"Precision: {results['test_precision']:.4f} | Recall: {results['test_recall']:.4f} | F1: {results['test_f1']:.4f}"
    )

    if not args.no_viz:
        print("\n" + "=" * 60)
        print("Vẽ biểu đồ training history...")
        plot_training_history(history)

        print("\n" + "=" * 60)
        print("Hiển thị mẫu dự đoán...")
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        show_predictions(model, test_loader, class_names, device, num_correct=2, num_wrong=2)


if __name__ == "__main__":
    main()

