import os
from typing import Optional
import matplotlib.pyplot as plt
from torchvision import datasets
from data_module import count_images


DATA_DIR = "vinafood21"


def show_overview(root_dir: str, max_cols: int = 6):
    print("=" * 60)
    print("Khám phá dữ liệu:")

    train_dir = os.path.join(root_dir, "train")
    test_dir = os.path.join(root_dir, "test")

    train_count, test_count = count_images(root_dir)
    train_set = datasets.ImageFolder(train_dir)
    classes = train_set.classes
    num_classes = len(classes)

    print(f"Số lớp: {num_classes}")
    print(f"Số ảnh train/test: {train_count}/{test_count}")
    print(f"Ví dụ nhãn: {classes[:5]}")

    if num_classes == 0:
        print("Không có lớp nào để hiển thị.")
        return

    # Lấy 1 ảnh của mỗi lớp, lấy 3 lớp đầu tiên
    num_show_classes = min(3, num_classes)
    class_samples = {}

    # Truy cập trực tiếp theo samples để tìm nhanh ảnh đầu tiên của mỗi lớp
    for img_path, label in train_set.samples:
        if label < num_show_classes and label not in class_samples:
            # Load ảnh trực tiếp
            from PIL import Image
            img = Image.open(img_path).convert('RGB')
            class_samples[label] = img
        if len(class_samples) == num_show_classes:
            break

    # Vẽ lưới: 1 hàng x num_show_classes cột
    fig, axes = plt.subplots(1, num_show_classes, figsize=(4 * num_show_classes, 4))

    if num_show_classes == 1:
        axes = [axes]

    for i in range(num_show_classes):
        ax = axes[i]
        if i in class_samples:
            ax.imshow(class_samples[i])
            ax.set_title(classes[i], fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DATA_DIR
    )
    args = parser.parse_args()

    show_overview(args.data_dir)


