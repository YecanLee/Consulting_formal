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
import torch

def load_ceiling_data_with_sem_seg(json_file, image_dir, dataset_name):
    """
    Loads COCO annotations and generates semantic segmentation masks as tensors.
    """
    dataset_dicts = load_coco_json(json_file, image_dir, dataset_name)
    
    processed_dicts = []
    for record in dataset_dicts:
        height, width = record["height"], record["width"]
        sem_seg_mask = np.zeros((height, width), dtype=np.uint8)
        ignore_label = MetadataCatalog.get(dataset_name).get("ignore_label", 255)
        sem_seg_mask.fill(ignore_label)
        
        if "annotations" in record and record["annotations"]:
            for ann in record["annotations"]:
                class_id = ann["category_id"]
                if "segmentation" in ann and ann["segmentation"]:
                    polygons = ann["segmentation"]
                    if isinstance(polygons[0], list):
                        polys = polygons
                    else:
                        polys = [polygons]
                    for poly in polys:
                        poly = np.array(poly, dtype=np.int32).reshape(-1, 2)
                        cv2.fillPoly(sem_seg_mask, [poly], class_id)
        
        record["sem_seg"] = torch.as_tensor(sem_seg_mask)
        processed_dicts.append(record)
    
    print(f"Processed {len(processed_dicts)} records with semantic segmentation masks for {dataset_name}.")
    return processed_dicts

def register_ceiling_dataset():
    # Use the default register_coco_instances for training data
    register_coco_instances("ceiling_easy_train", {}, TRAIN_JSON, TRAIN_DIR)
    print("Registered ceiling_easy_train")
    
    # Use the default register_coco_instances for test data
    register_coco_instances("ceiling_easy_test", {}, TEST_JSON, TEST_DIR)
    print("Registered ceiling_easy_test")
    
    # Use the default register_coco_instances for valid data
    register_coco_instances("ceiling_easy_val", {}, VAL_JSON, VAL_DIR)
    print("Registered ceiling_easy_val")
    
    # Use the default register_coco_instances for test data
    register_coco_instances("ceiling_easy_test", {}, TEST_JSON, TEST_DIR)
    print("Registered ceiling_easy_test")

register_ceiling_dataset()
