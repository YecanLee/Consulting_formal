TRAIN_JSON = "/home/ra78lof/consulting_pro/SAN/san/data/ceiling_painting_segmentation/train/json_annotation_train.json"
TRAIN_DIR = "/home/ra78lof/consulting_pro/SAN/san/data/ceiling_painting_segmentation/train"

VAL_JSON = "/home/ra78lof/consulting_pro/SAN/san/data/ceiling_painting_segmentation/valid/json_annotation_val.json"
VAL_DIR = "/home/ra78lof/consulting_pro/SAN/san/data/ceiling_painting_segmentation/valid"

TEST_JSON = "/home/ra78lof/consulting_pro/SAN/san/data/ceiling_painting_segmentation/test/json_annotation_test.json"
TEST_DIR = "/home/ra78lof/consulting_pro/SAN/san/data/ceiling_painting_segmentation/test"
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# register the dataset
from detectron2.data.datasets import register_coco_instances, load_coco_json

# visualize the dataset for test
import matplotlib
from matplotlib import pyplot as plt
import cv2
import random

from PIL import Image
import numpy as np
import tempfile

def load_ceiling_val_data_with_sem_seg():
    """
    Loads COCO annotations for the validation set and generates temporary
    semantic segmentation mask files from instance annotations for use by SemSegEvaluator.
    """
    # Load the standard COCO annotations
    dataset_dicts = load_coco_json(VAL_JSON, VAL_DIR, dataset_name="ceiling_easy_val")
    
    processed_dicts = []
    for record in dataset_dicts:
        # Get image dimensions
        height, width = record["height"], record["width"]
        
        # Initialize a semantic segmentation mask (all background initially)
        sem_seg_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Check if there are annotations for this image
        if "annotations" in record and record["annotations"]:
            for ann in record["annotations"]:
                # Get the class ID (assuming 'category_id' starts from 0 or 1)
                # Adjust if your category IDs don't map directly to mask values
                class_id = ann["category_id"]
                if class_id >= len(MetadataCatalog.get("ceiling_easy_val").stuff_classes):
                    print(f"Warning: Class ID {class_id} out of range for stuff_classes. Skipping.")
                    continue
                
                # Convert polygon to mask
                if "segmentation" in ann and ann["segmentation"]:
                    polygons = ann["segmentation"]
                    # Handle both single polygon and list of polygons
                    if isinstance(polygons[0], list):
                        polys = polygons
                    else:
                        polys = [polygons]
                    
                    # Draw each polygon as the class_id in the mask
                    for poly in polys:
                        # Reshape polygon to format expected by cv2.fillPoly
                        poly = np.array(poly, dtype=np.int32).reshape(-1, 2)
                        cv2.fillPoly(sem_seg_mask, [poly], class_id + 1)  # +1 if background is 0
                
        # Save the semantic segmentation mask to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            cv2.imwrite(tmp_file.name, sem_seg_mask)
            tmp_file_path = tmp_file.name
        
        # Add the temporary file path to the record
        record["sem_seg_file_name"] = tmp_file_path
        processed_dicts.append(record)
    
    print(f"Processed {len(processed_dicts)} validation records with temporary semantic segmentation masks.")
    return processed_dicts

def register_ceiling_dataset():
    # Register training data as before
    register_coco_instances("ceiling_easy_train", {}, TRAIN_JSON, TRAIN_DIR)
    print("Registered ceiling_easy_train")
    
    # Register validation data with custom loader
    DatasetCatalog.register("ceiling_easy_val", load_ceiling_val_data_with_sem_seg)
    
    # Set metadata for validation set
    MetadataCatalog.get("ceiling_easy_val").set(
        evaluator_type="sem_seg",
        stuff_classes=["background", "mural", "brief", "relief"],  # Adjust based on your categories
        ignore_label=255,  # Common for semantic segmentation
    )
    print("Registered ceiling_easy_val with evaluator_type='sem_seg'")
    
    # Register test data as before (adjust if needed for semantic seg evaluation)
    register_coco_instances("ceiling_easy_test", {}, TEST_JSON, TEST_DIR)
    print("Registered ceiling_easy_test")


# def register_ceiling_dataset():
#     register_coco_instances("ceiling_easy_train", {}, TRAIN_JSON, TRAIN_DIR)
#     register_coco_instances("ceiling_easy_val", {}, VAL_JSON, VAL_DIR)
#     register_coco_instances("ceiling_easy_test", {}, TEST_JSON, TEST_DIR)

#     # add some debug print
#     eval_dataset_metadata = MetadataCatalog.get("ceiling_easy_val")
#     del eval_dataset_metadata.evaluator_type
#     eval_dataset_metadata.evaluator_type = "sem_seg"
#     print("ceiling_easy_val evaluator_type 😨😨😨", eval_dataset_metadata.evaluator_type)

#     test_dataset_metadata = MetadataCatalog.get("ceiling_easy_test")
#     del test_dataset_metadata.evaluator_type
#     test_dataset_metadata.evaluator_type = "sem_seg"
#     print("ceiling_easy_test evaluator_type 😨😨😨", test_dataset_metadata.evaluator_type)

register_ceiling_dataset()
