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

from detectron2.data import MetadataCatalog



def register_ceiling_dataset():

    register_coco_instances("ceiling_easy_train", {}, TRAIN_JSON, TRAIN_DIR)
    print("Registered ceiling_easy_train")
    
    # Use the default register_coco_instances for valid data
    register_coco_instances("ceiling_easy_val", {}, VAL_JSON, VAL_DIR)
    print("Registered ceiling_easy_val")
    
    # Use the default register_coco_instances for test data
    register_coco_instances("ceiling_easy_test", {}, TEST_JSON, TEST_DIR)
    print("Registered ceiling_easy_test")

register_ceiling_dataset()

from detectron2.data import MetadataCatalog

### ---Debug Print--- ###
# Debug print shows that the dataset is registered
metadata = MetadataCatalog.get("ceiling_easy_train")
print(metadata, "this is the metadata 🔥🔥🔥")

dataset_train = DatasetCatalog.get("ceiling_easy_train")
print(dataset_train[0].keys(), "this is the dataset train 🔥🔥🔥")
### ---Debug Print Passed--- ###

"""
python train_maskclippp.py \
    --config-file configs/coco-stuff/eva-clip-vit-l-14-336/maskclippp_coco-stuff_eva-clip-vit-l-14-336_ceiling.yaml \
    --num-gpus 1 \
    --dist-url "auto" \
    WANDB.ENABLED True
"""