#!/bin/bash

files=(
    "train-images-idx3-ubyte"
    "train-labels-idx1-ubyte"
    "t10k-images-idx3-ubyte"
    "t10k-labels-idx1-ubyte"
)

base_url="http://yann.lecun.com/exdb/mnist"

for file in "${files[@]}"; do
    echo "Đang tải ${file}..."
    
    rm -f "$file"
    
    curl -L -o "$file" "$base_url/${file}.gz" --progress-bar
    
    if [ -f "$file" ]; then
        echo "$file đã được tải thành công"
    else
        echo "Lỗi: Không thể tải $file"
        exit 1
    fi
done

echo "Tất cả file đã được tải thành công!"
ls -l
