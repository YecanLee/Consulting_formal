import json
import numpy as np
from PIL import Image
import os
from pycocotools import mask as mask_util
from tqdm import tqdm

def convert_coco_to_semantic_masks(json_file, image_dir, output_dir, class_mapping=None):
    """
    Convert COCO instance annotations to semantic segmentation masks.
    
    Args:
        json_file: Path to COCO JSON annotation file
        image_dir: Directory containing the images
        output_dir: Directory to save semantic segmentation masks
        class_mapping: Optional dictionary mapping category names to IDs for semantic masks
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load COCO annotations
    with open(json_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create category mapping
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # If no class mapping provided, create one based on unique names
    if class_mapping is None:
        unique_names = sorted(set(categories.values()))
        class_mapping = {name: idx + 1 for idx, name in enumerate(unique_names)}  # 0 is background
    
    print(f"Class mapping: {class_mapping}")
    
    # Create image id to annotations mapping
    img_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
    
    # Process each image
    for img_info in tqdm(coco_data['images'], desc="Converting images"):
        img_id = img_info['id']
        img_filename = img_info['file_name']
        height = img_info['height']
        width = img_info['width']
        
        # Initialize semantic mask (0 is background)
        semantic_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Get annotations for this image
        if img_id in img_to_anns:
            for ann in img_to_anns[img_id]:
                cat_id = ann['category_id']
                cat_name = categories[cat_id]
                semantic_id = class_mapping[cat_name]
                
                # Convert segmentation to binary mask
                if 'segmentation' in ann:
                    if isinstance(ann['segmentation'], list):
                        # Polygon format
                        from pycocotools.coco import COCO
                        rles = mask_util.frPyObjects(ann['segmentation'], height, width)
                        rle = mask_util.merge(rles)
                        binary_mask = mask_util.decode(rle)
                    else:
                        # RLE format
                        binary_mask = mask_util.decode(ann['segmentation'])
                    
                    # Update semantic mask
                    semantic_mask[binary_mask > 0] = semantic_id
        
        # Save semantic mask
        output_filename = os.path.splitext(img_filename)[0] + '.png'
        output_path = os.path.join(output_dir, output_filename)
        Image.fromarray(semantic_mask).save(output_path)
    
    print(f"Converted {len(coco_data['images'])} images")
    return class_mapping

# Convert ceiling painting datasets
if __name__ == "__main__":
    # Define class mapping (consistent across train/val/test)
    class_mapping = {
        "mural": 1,
        "brief": 2,
        "relief": 3
    }
    
    # Convert training set
    print("Converting training set...")
    convert_coco_to_semantic_masks(
        json_file="/home/ra78lof/consulting_pro/SAN/san/data/ceiling_painting_segmentation/train/json_annotation_train.json",
        image_dir="/home/ra78lof/consulting_pro/SAN/san/data/ceiling_painting_segmentation/train",
        output_dir="/home/ra78lof/consulting_pro/ZegFormer/datasets/ceiling_painting/annotations_train",
        class_mapping=class_mapping
    )
    
    # Convert validation set
    print("Converting validation set...")
    convert_coco_to_semantic_masks(
        json_file="/home/ra78lof/consulting_pro/SAN/san/data/ceiling_painting_segmentation/valid/json_annotation_val.json",
        image_dir="/home/ra78lof/consulting_pro/SAN/san/data/ceiling_painting_segmentation/valid",
        output_dir="/home/ra78lof/consulting_pro/ZegFormer/datasets/ceiling_painting/annotations_val",
        class_mapping=class_mapping
    )
    
    # Convert test set
    print("Converting test set...")
    convert_coco_to_semantic_masks(
        json_file="/home/ra78lof/consulting_pro/SAN/san/data/ceiling_painting_segmentation/test/json_annotation_test.json",
        image_dir="/home/ra78lof/consulting_pro/SAN/san/data/ceiling_painting_segmentation/test",
        output_dir="/home/ra78lof/consulting_pro/ZegFormer/datasets/ceiling_painting/annotations_test",
        class_mapping=class_mapping
    ) 