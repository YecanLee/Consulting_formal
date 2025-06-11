#!/usr/bin/env python3
"""
Script to reconvert masks from grayscale values to proper class indices
"""
import numpy as np
from pathlib import Path
import sys

def reconvert_masks(data_dir, num_classes=4):
    """Reconvert masks to use proper class indices"""
    data_path = Path(data_dir)
    npy_files = sorted(data_path.glob("*.npy"))
    
    if not npy_files:
        print(f"No .npy files found in {data_dir}")
        return
    
    print(f"Found {len(npy_files)} mask files to reconvert")
    
    # First, analyze the unique values across all masks
    all_unique_values = set()
    for npy_file in npy_files:
        mask = np.load(str(npy_file))
        unique_values = np.unique(mask)
        all_unique_values.update(unique_values.tolist())
    
    all_unique_values = sorted(list(all_unique_values))
    print(f"\nUnique values found across all masks: {all_unique_values}")
    
    if len(all_unique_values) > num_classes:
        print(f"WARNING: Found {len(all_unique_values)} unique values but expected {num_classes} classes")
        print(f"Will map the first {num_classes} values to class indices 0-{num_classes-1}")
    
    # Create global mapping
    value_to_class = {val: idx for idx, val in enumerate(all_unique_values[:num_classes])}
    print(f"\nMapping: {value_to_class}")
    
    # Reconvert each mask
    for i, npy_file in enumerate(npy_files):
        mask = np.load(str(npy_file))
        
        # Convert to class indices
        class_mask = np.zeros_like(mask, dtype=np.uint8)
        for pixel_val, class_idx in value_to_class.items():
            class_mask[mask == pixel_val] = class_idx
        
        # Handle any remaining values (clamp to max class)
        for val in all_unique_values[num_classes:]:
            class_mask[mask == val] = num_classes - 1
        
        # Save the converted mask
        np.save(str(npy_file), class_mask)
        
        if i % 10 == 0:
            print(f"Processed {i+1}/{len(npy_files)} masks...")
    
    print(f"\nConversion complete! All masks now use class indices 0-{num_classes-1}")

if __name__ == "__main__":
    # Process all data directories
    data_dirs = [
        "data/ceiling_easy_train_with_masks/train/",
        "data/ceiling_easy_train_with_masks/val/",
        "data/ceiling_easy_train_with_masks/test/",
    ]
    
    num_classes = 4  # background, mural, brief, relief
    
    for data_dir in data_dirs:
        if Path(data_dir).exists():
            print(f"\n{'='*50}")
            print(f"Processing: {data_dir}")
            print(f"{'='*50}")
            reconvert_masks(data_dir, num_classes)
        else:
            print(f"Skipping {data_dir} (not found)") 