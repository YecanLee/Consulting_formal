import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

CLASS_NAMES = (
    "mural",
    "non-mural",
)


def _get_ceiling_meta(cat_list):
    ret = {
        "stuff_classes": cat_list,
    }
    return ret


def register_all_ceiling_paintings(root):
    """
    Define a function to register the ceiling paintings dataset to detectron2.

    Args:
        root: The root directory of the ceiling paintings segmentation dataset.
    """
    root = os.path.join(root, "ceiling_painting_segmentation")
    meta = _get_ceiling_meta(CLASS_NAMES)

    # TODO: make sure all the images are in JPEG format

    for name, image_dirname, sem_seg_dirname in [
        ("train", "train", "annotations_detectron2/train"),
        ("valid", "valid", "annotations_detectron2/valid"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        # Example name of the image_dir path: CEILING_PAINTINGS_SEGMENTATION/JPEGImages/000000000000.jpg
        # Example name of the gt_dir path: CEILING_PAINTINGS_SEGMENTATION/annotations_detectron2/train/000000000000.png
        all_name = f"ceiling_paintings_sem_seg_{name}"
        DatasetCatalog.register(
            all_name,
            lambda x=image_dir, y=gt_dir: load_sem_seg(
                y, x, gt_ext="png", image_ext="jpg"
            ),
        )
        MetadataCatalog.get(all_name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **meta,
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_ceiling_paintings(_root)
