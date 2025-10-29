import torch
from torch.utils.data import Dataset
import idx2numpy
import numpy as np
import gzip
import os

def load_idx_file(gz_path: str) -> np.ndarray:
    print(f"\nReading file: {gz_path}")

    # Đọc file gz
    with gzip.open(gz_path, 'rb') as f:
        data = f.read()

    # Lưu file giải nén tạm thời
    temp_path = gz_path[:-3]  # Bỏ đuôi .gz để tạo file tạm
    try:
        # Ghi file tạm
        with open(temp_path, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())

        # Đọc file IDX
        result = idx2numpy.convert_from_file(temp_path)
        print(f"Reading files complete, shape: {result.shape}")
        return result

    except Exception as e:
        print(f"Error: {str(e)}")
        raise

class MnistDataset(Dataset):
    def __init__(self, image_path: str, label_path: str):
        images = load_idx_file(image_path)
        labels = load_idx_file(label_path)

        # Chuẩn hóa ảnh về [0, 1]
        images = images.astype('float32') / 255.0

        # Chuyển đổi dữ liệu thành list[dict]
        self._data = [
            {
                'image': np.array(image),
                'label': label
            }
            for image, label in zip(images, labels)
        ]
        print(f"Dataset loaded with {len(self._data)} samples")

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

def collate_fn(items: list[dict]):
    # Thêm chiều channel cho ảnh và gom thành batch
    items = [{
        'image': item['image'][None, ...],  # Thêm chiều channel (1, 28, 28)
        'label': np.array(item['label'])
    } for item in items]

    # Stack các ảnh và nhãn thành batch
    items = {
        'image': np.stack([item['image'] for item in items], axis=0),
        'label': np.stack([item['label'] for item in items], axis=0)
    }

    # Chuyển sang tensor
    items = {
        'image': torch.tensor(items['image']),
        'label': torch.tensor(items['label'])
    }

    return items
