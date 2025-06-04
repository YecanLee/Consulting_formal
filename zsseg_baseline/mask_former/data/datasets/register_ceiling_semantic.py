import os
from detectron2.data import DatasetCatalog, MetadataCatalog
import logging

# set up the logger to log the mask checking process
logger = logging.getLogger(__name__)

def get_ceiling_semantic_dataset_dicts(image_dir, ann_dir):
    """
    Create dataset dicts for semantic segmentation from separate image and annotation directories.
    
    Args:
        image_dir: Directory containing images
        ann_dir: Directory containing semantic segmentation masks
        
    Returns:
        list[dict]: List of dataset dicts in detectron2 format
    """
    dataset_dicts = []
    
    # Get all image files in the img_dir subfolder
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    
    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(image_dir, image_file)
        # The mask file has _mask.png suffix with the same name as the image name
        mask_file = image_file.replace('.jpg', '_mask.png')
        mask_path = os.path.join(ann_dir, mask_file)
        
        # Check if mask exists
        if not os.path.exists(mask_path):
            logger.warning(f"Warning: No mask found for {image_file}, skipping...")
            continue
            
        # Create dataset dict
        record = {
            "file_name": image_path,
            "sem_seg_file_name": mask_path,
            "image_id": idx,
        }
        dataset_dicts.append(record)
    
    return dataset_dicts

def register_ceiling_semantic_datasets(root_dir="/home/ra78lof/consulting_pro"):
    """
    Register ceiling painting semantic segmentation datasets.

    Args:
        root_dir: The root directory of the dataset

    Returns:
        None
    """
    logger.warning(f'You may need to adjust the dataset paths to the correct ones! ⚠️')
    # Define dataset paths
    datasets = {
        "ceiling_sem_seg_train": {
            "image_dir": os.path.join(root_dir, "ceiling_easy_train_with_masks/imd_dir"),
            "ann_dir": os.path.join(root_dir, "ceiling_easy_train_with_masks/ann_dir"),
        },
        "ceiling_sem_seg_val": {
            "image_dir": os.path.join(root_dir, "ceiling_easy_valid_with_masks/imd_dir"),
            "ann_dir": os.path.join(root_dir, "ceiling_easy_valid_with_masks/ann_dir"),
        },
        "ceiling_sem_seg_test": {
            "image_dir": os.path.join(root_dir, "ceiling_easy_test_with_masks/imd_dir"),
            "ann_dir": os.path.join(root_dir, "ceiling_easy_test_with_masks/ann_dir"),
        },
    }
    
    # Define the classes 
    logger.warning(f"You may need to adjust the class names to the correct ones! ⚠️")
    stuff_classes = ["mural", "brief", "mural", "relief"]  
    # Same as the `mmcv-full` format of color palette
    stuff_colors = [[220, 20, 60], [119, 11, 32], [0, 0, 142], [0, 60, 100]] 
    
    for dataset_name, paths in datasets.items():
        # Register dataset
        DatasetCatalog.register(
            dataset_name,
            lambda paths=paths: get_ceiling_semantic_dataset_dicts(paths["image_dir"], paths["ann_dir"])
        )
        
        # Set metadata
        MetadataCatalog.get(dataset_name).set(
            stuff_classes=stuff_classes,
            stuff_colors=stuff_colors,
            evaluator_type="sem_seg",
            ignore_label=255,  # Label for unlabeled pixels
        )
        
        print(f"Registered {dataset_name}")
    
    # Print dataset info
    print(f"\nDataset registered with {len(stuff_classes)} classes: {stuff_classes}")

# Register the datasets when this module is imported
register_ceiling_semantic_datasets()

# Test the registration
if __name__ == "__main__":
    # Verify registration
    dataset_name = "ceiling_sem_seg_train"
    dataset_dicts = DatasetCatalog.get(dataset_name)
    print(f"\nLoaded {len(dataset_dicts)} images from {dataset_name}")
    
    # Check first few entries
    for i, d in enumerate(dataset_dicts[:3]):
        print(f"\nEntry {i}:")
        print(f"  Image: {os.path.basename(d['file_name'])}")
        print(f"  Mask: {os.path.basename(d['sem_seg_file_name'])}")
        
    # Check metadata
    metadata = MetadataCatalog.get(dataset_name)
    print(f"\nMetadata:")
    print(f"  Classes: {metadata.stuff_classes}")
    print(f"  Evaluator type: {metadata.evaluator_type}")
    print(f"  Ignore label: {metadata.ignore_label}") 