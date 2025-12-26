We modified all the `register_ceiling*` scripts to have one important function: `load_ceiling_data_with_sem_seg`.

## `load_ceiling_data_with_sem_seg` function
The core issue with working with `Detectron2` is that it requires the semantic segmentation masks to be in the format of `PNG` files. However, for most of the annotation tools, the semantic segmentation masks are in the format of `COCO` format to be more flexible. This means that there is no independent semantic segmentation masksed version file for each image. They are all stored in the same `COCO` format file.

```bash
┌──────────────────────────────────────────────────────────────────┐
│                      COCO JSON                                   │
│  {                                                               │
│    "annotations": [                                              │
│      {"segmentation": [[x1,y1,x2,y2,...]], "category_id": 0},    │
│      {"segmentation": [[x1,y1,x2,y2,...]], "category_id": 3}     │
│    ]                                                             │
│  }                                                               │
└──────────────────────────────────────────────────────────────────┘
```

The source codes of the `SemSegEvaluator` in `Detectron2` expects `sem_seg_file_name` to be part of the dataset record.

```python 
...
        self.input_file_to_gt_file = {
            dataset_record["file_name"]: dataset_record["sem_seg_file_name"]
            for dataset_record in DatasetCatalog.get(dataset_name)
        }
...
```

It also expects to load the masked images directly by using self-defined function.

```python
...
"""
sem_seg_loading_fn: function to read sem seg file and load into numpy array.
"""
...
```

How the function works is more intuitive and self-explained, you can refer to the codes for more details, basically:
1.The function first loads the `COCO` config json file and gets the `file_name`, `height`, `width` and `annotations`.
2.Then it creates a blank semantic segmentation mask with the same size as the image by using the `np.full` function.
3.Then it iterates over the `annotations` and fills the semantic segmentation mask with `OpenCV`'s `fillPoly` function.
4.Then it saves the semantic segmentation mask as a `PNG` file in the `sem_seg_masks` directory.
5.Finally, it returns the dataset record with the added `sem_seg` and `sem_seg_file_name`. The `sem_seg` is for Training and the `sem_seg_file_name` is for Evaluation.

The whole pipeline can be summarized as follows:
```bash
┌──────────────────────────────────────────────────────────────────┐
│                      COCO JSON                                   │
│  {                                                               │
│    "annotations": [                                              │
│      {"segmentation": [[x1,y1,x2,y2,...]], "category_id": 0},    │
│      {"segmentation": [[x1,y1,x2,y2,...]], "category_id": 3}     │
│    ]                                                             │
│  }                                                               │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────┐
        │     load_ceiling_data_with_sem_seg   │
        │                                      │
        │  1. Create blank mask (all 255)      │
        │  2. For each polygon:                │
        │     cv2.fillPoly(mask, poly, class)  │
        │  3. Save as PNG                      │
        └──────────────────────────────────────┘
                           │
         ┌─────────────────┴─────────────────┐
         ▼                                   ▼
┌─────────────────────┐           ┌─────────────────────┐
│   sem_seg_masks/    │           │   Dataset Record    │
│   image_001.png     │           │                     │
│   (grayscale PNG)   │           │ "sem_seg": tensor   │
│                     │           │ "sem_seg_file_name" │
│   Pixel values:     │           │ "annotations": [...]│
│   0,1,2,3 = classes │           │                     │
│   255 = ignore      │           │                     │
└─────────────────────┘           └─────────────────────┘
         │                                   │
         ▼                                   ▼
  SemSegEvaluator                     Training Loop
  (reads PNG from disk)              (uses tensor directly)
```

## `register_ceiling_dataset` function
We also modified the `register_ceiling_dataset` function to register the dataset with the `load_ceiling_data_with_sem_seg` function.

`Detectron2` doesn't know about our customized dataset. So we need to tell it about the dataset:

1.Where to find the dataset, this is done by the `DatasetCatalog.register` function.
2.What is included in the dataset, this is done by the `MetadataCatalog` function, it includes the many things like `stuff_classes`, `thing_classes`, `ignore_label`, `evaluator_type` etc.

The `DatasetCatalog.register` function can be visualized like this way:
```bash
┌─────────────────────────────────────────────────────────────┐
│                     DatasetCatalog                          │
│                                                             │
│  "coco_2017_train"    → func() returns COCO train data      │
│  "cityscapes_train"   → func() returns Cityscapes data      │
│  "ceiling_easy_train" → func() returns ceiling data         │  ← NEW!
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

The `MetadataCatalog` function can be visualized like this way:
```bash
┌─────────────────────────────────────────────────────────────┐
│                    MetadataCatalog                          │
│                                                             │
│  "ceiling_easy_train":                                      │
│    stuff_classes = ["mural", "brief", "mural", "relief"]    │
│    thing_classes = ["mural", "brief", "mural", "relief"]    │
│    ignore_label  = 255                                      │
│    evaluator_type = "sem_seg"                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

This means that when we called the `from sed.data.datasets import register_ceiling`, The datasets are **automatically registered** without needing to call the function manually.

The whole pipeline can be summarized as follows:
```bash
┌──────────────────────────────────────────────────────────────────┐
│                     register_ceiling_dataset()                   │
└──────────────────────────────────────────────────────────────────┘
                               │
           ┌───────────────────┼───────────────────┐
           ▼                   ▼                   ▼
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │   TRAIN     │     │     VAL     │     │    TEST     │
    └─────────────┘     └─────────────┘     └─────────────┘
           │                   │                   │
           ▼                   ▼                   ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                     DatasetCatalog                          │
    │                                                             │
    │  "ceiling_easy_train" → lambda: load_ceiling_data_with_..   │
    │  "ceiling_easy_val"   → lambda: load_ceiling_data_with_..   │
    │  "ceiling_easy_test"  → lambda: load_ceiling_data_with_..   │
    └─────────────────────────────────────────────────────────────┘
           │                   │                   │
           ▼                   ▼                   ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                     MetadataCatalog                         │
    │                                                             │
    │  Each dataset gets:                                         │
    │    stuff_classes  = ["mural", "brief", "mural", "relief"]   │
    │    thing_classes  = ["mural", "brief", "mural", "relief"]   │
    │    ignore_label   = 255                                     │
    │    evaluator_type = "sem_seg"                               │
    └─────────────────────────────────────────────────────────────┘
```