import os
import cv2
import numpy as np
import json
from PIL import Image
import shutil
from pathlib import Path

def convert_mmcv_format_to_slime(input_dir, output_dir, split_name="train"):
    """
    Convert mmcv format to `SLiMe` format
    input_dir/
        img_dir/
            img1.jpg
            img2.jpg
        ann_dir/
            img1_mask.png
            img2_mask.png
    """
    # Create output directory
    output_path = Path(output_dir) / split_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    images_dir = Path(input_dir) / "img_dir"
    masks_dir = Path(input_dir) / "ann_dir"
    
    # Get all image files
    image_files = sorted(images_dir.glob("*.jpg"))
    
    # Track unique mask values to understand the data
    all_unique_values = set()
    
    for img_path in image_files:
        # Get corresponding mask
        mask_name = img_path.stem + "_mask.png"
        mask_path = masks_dir / mask_name
        
        if not mask_path.exists():
            print(f"Warning: Mask not found for {img_path.name}")
            continue
            
        # Read and convert image to PNG
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        output_img_path = output_path / f"{img_path.stem}.png"
        cv2.imwrite(str(output_img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        # Read mask and save as numpy array
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Convert mask values to class indices
        # Common mappings:
        # - Binary masks: 0 (background), 255 (foreground) -> 0, 1
        # - Multi-class: 0, 85, 170, 255 -> 0, 1, 2, 3
        unique_values = np.unique(mask)
        all_unique_values.update(unique_values.tolist())
        
        # Create a mapping from pixel values to class indices
        value_to_class = {val: idx for idx, val in enumerate(sorted(unique_values))}
        
        # Convert mask to class indices
        class_mask = np.zeros_like(mask, dtype=np.uint8)
        for pixel_val, class_idx in value_to_class.items():
            class_mask[mask == pixel_val] = class_idx
        
        output_mask_path = output_path / f"{img_path.stem}.npy"
        np.save(str(output_mask_path), class_mask)
        
        print(f"Converted: {img_path.name} -> {output_img_path.name} + {output_mask_path.name}")
        print(f"  Mask values {unique_values} -> classes {list(range(len(unique_values)))}")
    
    print(f"\nAll unique mask values found in dataset: {sorted(all_unique_values)}")
    print(f"This suggests {len(all_unique_values)} classes (including background)")
    return len(all_unique_values)

def convert_detectron2_format_to_slime(images_dir, json_path, output_dir, split_name="train"):
    """
    Convert detectron2 format to `SLiMe` format
    images_dir/
        img1.jpg
        img2.jpg
    metadata.json with segmentation info
    """
    # Create output directory
    output_path = Path(output_dir) / split_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load JSON metadata
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    
    images_path = Path(images_dir)
    
    for img_name, seg_info in metadata.items():
        img_path = images_path / img_name
        
        if not img_path.exists():
            print(f"Warning: Image not found: {img_name}")
            continue
            
        # Read and convert image to PNG
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        output_img_path = output_path / f"{img_path.stem}.png"
        cv2.imwrite(str(output_img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        # Create mask from segmentation info
        # Assuming seg_info contains mask data or annotations
        # You'll need to adapt this based on your JSON structure
        if isinstance(seg_info, dict) and 'mask' in seg_info:
            # If mask is stored as array in JSON
            mask = np.array(seg_info['mask'])
        elif isinstance(seg_info, dict) and 'annotations' in seg_info:
            # If you have polygon/bbox annotations, convert to mask
            # This is a placeholder - adapt based on your format
            h, w = img.shape[:2]
            mask = create_mask_from_annotations(seg_info['annotations'], h, w)
        else:
            print(f"Warning: Cannot parse segmentation info for {img_name}")
            continue
            
        output_mask_path = output_path / f"{img_path.stem}.npy"
        np.save(str(output_mask_path), mask)
        
        print(f"Converted: {img_name} -> {output_img_path.name} + {output_mask_path.name}")

def create_mask_from_annotations(annotations, height, width):
    """
    Create mask from annotations (polygons, bboxes, etc.)
    This is a placeholder - implement based on your annotation format
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    
    for ann in annotations:
        if 'polygon' in ann:
            # Convert polygon to mask
            pts = np.array(ann['polygon'], np.int32)
            cv2.fillPoly(mask, [pts], ann.get('class_id', 1))
        elif 'bbox' in ann:
            # Convert bbox to mask
            x, y, w, h = ann['bbox']
            mask[y:y+h, x:x+w] = ann.get('class_id', 1)
    
    return mask

def verify_dataset(dataset_dir):
    """Verify the converted dataset"""
    dataset_path = Path(dataset_dir)
    
    png_files = sorted(dataset_path.glob("*.png"))
    npy_files = sorted(dataset_path.glob("*.npy"))
    
    print(f"\nDataset verification for {dataset_dir}:")
    print(f"Found {len(png_files)} PNG images")
    print(f"Found {len(npy_files)} NPY masks")
    
    # Check if each image has a corresponding mask
    for png_file in png_files:
        npy_file = dataset_path / f"{png_file.stem}.npy"
        if not npy_file.exists():
            print(f"Warning: Missing mask for {png_file.name}")
    
    # Print unique class labels
    if npy_files:
        sample_mask = np.load(str(npy_files[0]))
        unique_classes = np.unique(sample_mask)
        print(f"Unique class labels in first mask: {unique_classes}")
        print(f"Mask shape: {sample_mask.shape}")

def create_train_val_test_split(input_dir, output_base_dir, train_ratio=0.7, val_ratio=0.15):
    """Split dataset into train/val/test sets"""
    input_path = Path(input_dir)
    png_files = sorted(input_path.glob("*.png"))
    
    n_total = len(png_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Shuffle files
    import random
    random.shuffle(png_files)
    
    # Split files
    train_files = png_files[:n_train]
    val_files = png_files[n_train:n_train+n_val]
    test_files = png_files[n_train+n_val:]
    
    # Create directories and copy files
    for split_name, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
        split_dir = Path(output_base_dir) / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        for png_file in files:
            # Copy image
            shutil.copy2(png_file, split_dir / png_file.name)
            
            # Copy mask
            npy_file = input_path / f"{png_file.stem}.npy"
            if npy_file.exists():
                shutil.copy2(npy_file, split_dir / npy_file.name)
        
        print(f"Created {split_name} set with {len(files)} samples")

# Example usage
if __name__ == "__main__":
    # For format 1 (separate folders)
    convert_mmcv_format_to_slime(
        input_dir="../ceiling_easy_train_with_masks",
        output_dir="./data/ceiling_easy_train_with_masks",
        split_name="train"
    )
    
    # For format 2 (JSON metadata)
    # convert_detectron2_format_to_slime(
    #     images_dir="path/to/images",
    #     json_path="path/to/metadata.json",
    #     output_dir="slime/data/custom_dataset",
    #     split_name="all_data"
    # )
    
    # # Verify conversion
    # verify_dataset("/data/ceiling_easy_train_with_masks/train")
    
    # # Create train/val/test splits
    # create_train_val_test_split(
    #     input_dir="/data/ceiling_easy_train_with_masks/train",
    #     output_base_dir="/data/ceiling_easy_train_with_masks",
    #     train_ratio=0.7,
    #     val_ratio=0.15
    # )
