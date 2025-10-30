import torch
import time
from data_module import build_dataloaders

DATA_DIR = "/content/vinafood21"

print("=" * 60)
print("1. Kiểm tra CUDA:")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print("\n" + "=" * 60)
print("2. Đang load dataset...")
start = time.time()

try:
    train_loader, test_loader, num_classes, class_names = build_dataloaders(
        DATA_DIR, batch_size=32, num_workers=0, image_size=224
    )
    load_time = time.time() - start
    
    print(f"   ✓ Load thành công trong {load_time:.2f}s")
    print(f"   Số lớp: {num_classes}")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    print(f"   Train samples: {len(train_loader.dataset)}")
    print(f"   Test samples: {len(test_loader.dataset)}")
    
    print("\n" + "=" * 60)
    print("3. Test tốc độ load 1 batch:")
    start = time.time()
    images, labels = next(iter(train_loader))
    batch_time = time.time() - start
    print(f"   ✓ Load 1 batch trong {batch_time:.3f}s")
    print(f"   Batch shape: {images.shape}")
    
    print("\n" + "=" * 60)
    print("4. Test forward pass trên GPU:")
    if torch.cuda.is_available():
        from googlenet import build_googlenet
        model = build_googlenet(num_classes=num_classes).cuda()
        images = images.cuda()
        
        # Warm up
        with torch.no_grad():
            _ = model(images)
        
        start = time.time()
        with torch.no_grad():
            outputs = model(images)
        forward_time = time.time() - start
        print(f"   ✓ Forward pass trong {forward_time:.3f}s")
        print(f"   Output shape: {outputs.shape}")
        
        # Ước lượng thời gian (ĐÚNG)
        train_batches = len(train_loader)
        gpu_time_per_batch = forward_time * 3  # forward + backward
        total_time_per_batch = batch_time + gpu_time_per_batch  # load + GPU
        estimated_epoch_time = train_batches * total_time_per_batch
        
        print(f"\n   Phân tích thời gian/batch:")
        print(f"   - Load data: {batch_time:.3f}s")
        print(f"   - GPU (forward+backward): {gpu_time_per_batch:.3f}s")
        print(f"   - Tổng: {total_time_per_batch:.3f}s/batch")
        print(f"\n   Ước tính thời gian 1 epoch train: {estimated_epoch_time:.1f}s (~{estimated_epoch_time/60:.1f} phút)")
        
except Exception as e:
    print(f"   ✗ LỖI: {e}")
    import traceback
    traceback.print_exc()

print("=" * 60)

