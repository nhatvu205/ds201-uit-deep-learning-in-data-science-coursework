import torch
from torch.utils.data import Dataset
import numpy as np
import os
import idx2numpy

def _read_uncompressed_idx(idx_path: str) -> np.ndarray:
    return idx2numpy.convert_from_file(idx_path)

def load_idx_file(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find {path}")
    # Đọc trực tiếp file không đuôi .gz
    return _read_uncompressed_idx(path)

class MnistDataset(Dataset):
    def __init__(self, image_path: str, label_path: str):
        images = load_idx_file(image_path)
        labels = load_idx_file(label_path)

        # Chuẩn hóa về [0,1]
        images = images.astype('float32') / 255.0

        # Chuyển đổi dữ liệu thành list[dict]
        self._data = [
            {
                'image': np.array(image),
                'label': label
            }
            for image, label in zip(images, labels)
        ]

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    @staticmethod
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