# datasets/custom_dataset.py
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from mmdet.registry import DATASETS
from mmdet.datasets.base_det_dataset import BaseDetDataset
import cv2


@DATASETS.register_module()
class CeilingPaintingSegmentationDataset(BaseDetDataset):
    """Custom Segmentation Dataset for training and zero-shot inference.
    
    Args:
        data_root (str): Root directory of the dataset.
        ann_file (str): Path to annotation file.
        data_prefix (dict): Prefix for data paths.
        filter_cfg (dict, optional): Config for filtering annotations.
        pipeline (list[dict]): Processing pipeline.
        **kwargs: Other arguments for BaseDetDataset.
    """
    
    METAINFO = {
        'classes': ("#","brief", "mural", "relief"),
        'palette': ((220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230))
    }
    
    def __init__(self,
                 data_root,
                 ann_file='',
                 data_prefix=dict(img='images/', seg='annotations/'),
                 filter_cfg=None,
                 pipeline=None,
                 serialize_data=True,
                 lazy_init=False,
                 max_refetch=1000,
                 **kwargs):
        
        self.data_prefix = data_prefix
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            data_prefix=data_prefix,
            filter_cfg=filter_cfg,
            pipeline=pipeline,
            serialize_data=serialize_data,
            lazy_init=lazy_init,
            max_refetch=max_refetch,
            **kwargs)
    
    def load_data_list(self):
        """Load annotations from directory structure or annotation file."""
        data_list = []
        
        # Using annotation file
        if self.ann_file:
            import json
            with open(self.ann_file, 'r') as f:
                annotations = json.load(f)
            
            # Set classes if available
            if 'categories' in annotations:
                self.METAINFO['classes'] = tuple([cat['name'] for cat in annotations['categories']])
            
            # Process each image
            for img_info in annotations['images']:
                data_info = {
                    'img_id': img_info['id'],
                    'img_path': os.path.join(self.data_prefix['img'], img_info['file_name']),
                    'height': img_info['height'],
                    'width': img_info['width'],
                    'instances': []
                }
                
                # Get annotations for this image
                img_anns = [ann for ann in annotations['annotations'] if ann['image_id'] == img_info['id']]
                
                for ann in img_anns:
                    instance = {
                        'bbox': ann['bbox'],  # [x, y, w, h]
                        'bbox_label': ann['category_id'],
                        'ignore_flag': ann.get('ignore', False)
                    }
                    
                    # Add segmentation if available
                    if 'segmentation' in ann:
                        instance['mask'] = ann['segmentation']
                    
                    data_info['instances'].append(instance)
                
                data_list.append(data_info)
        
        else:
            # Load from directory structure
            img_dir = os.path.join(self.data_root, self.data_prefix['img'])
            seg_dir = os.path.join(self.data_root, self.data_prefix.get('seg', ''))
            
            # Get all images
            img_files = sorted([f for f in os.listdir(img_dir) 
                               if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            
            for idx, img_file in enumerate(img_files):
                img_path = os.path.join(img_dir, img_file)
                
                # Get image info
                img = Image.open(img_path)
                width, height = img.size
                
                data_info = {
                    'img_id': idx,
                    'img_path': img_path,
                    'height': height,
                    'width': width,
                    'instances': []
                }
                
                # Check for corresponding segmentation mask
                if seg_dir and os.path.exists(seg_dir):
                    # Get mask file name - append _mask to the base name
                    base_name = os.path.splitext(img_file)[0]
                    seg_file = base_name + '_mask.png'
                    seg_path = os.path.join(seg_dir, seg_file)
                    
                    if os.path.exists(seg_path):
                        data_info['seg_map_path'] = seg_path
                        
                        # Load mask to get instances
                        seg_map = np.array(Image.open(seg_path))
                        
                        # Handle special mask format where 255 is background
                        # Map 255 to 0 (background)
                        seg_map_processed = seg_map.copy()
                        seg_map_processed[seg_map == 255] = 0
                        
                        # Get unique class IDs from mask (excluding background 0)
                        unique_labels = np.unique(seg_map_processed)
                        unique_labels = unique_labels[unique_labels > 0]
                        
                        # Create instance for each unique label
                        for label in unique_labels:
                            mask = (seg_map_processed == label)
                            
                            # Get bounding box from mask
                            y_indices, x_indices = np.where(mask)
                            if len(x_indices) > 0 and len(y_indices) > 0:
                                x_min, x_max = x_indices.min(), x_indices.max()
                                y_min, y_max = y_indices.min(), y_indices.max()
                                
                                # Convert mask to polygon format
                                mask_uint8 = mask.astype(np.uint8) * 255
                                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                
                                # Convert contours to polygon format
                                polygons = []
                                for contour in contours:
                                    if contour.shape[0] >= 3:  # Valid polygon must have at least 3 points
                                        polygon = contour.flatten().tolist()
                                        if len(polygon) >= 6:  # At least 3 points (6 coordinates)
                                            polygons.append(polygon)
                                
                                if polygons:  # Only add instance if we have valid polygons
                                    instance = {
                                        'bbox': [x_min, y_min, x_max - x_min, y_max - y_min],  # [x, y, w, h] format
                                        'bbox_label': int(label - 1),  # Convert to 0-indexed
                                        'mask': polygons,  # Polygon masks
                                        'ignore_flag': False
                                    }
                                    data_info['instances'].append(instance)
                
                data_list.append(data_info)
        
        return data_list
    
    def get_cat_ids(self, idx):
        """Get category ids by index."""
        data_info = self.get_data_info(idx)
        if 'instances' in data_info:
            return [inst['bbox_label'] for inst in data_info['instances']]
        return []
    
    def parse_data_info(self, raw_data_info):
        """Parse raw data info to standard format."""
        data_info = raw_data_info.copy()
        
        # Ensure instances are properly formatted
        if 'instances' in data_info:
            instances = []
            for inst in data_info['instances']:
                instance = {}
                
                # Convert bbox format if needed
                if 'bbox' in inst:
                    bbox = inst['bbox']
                    if len(bbox) == 4:
                        # Convert [x, y, w, h] to [x1, y1, x2, y2]
                        instance['bbox'] = [bbox[0], bbox[1], 
                                          bbox[0] + bbox[2], bbox[1] + bbox[3]]
                    else:
                        instance['bbox'] = bbox
                
                # Add other fields
                instance['bbox_label'] = inst.get('bbox_label', 0)
                instance['ignore_flag'] = inst.get('ignore_flag', False)
                
                # Add mask if available
                if 'mask' in inst:
                    instance['mask'] = inst['mask']
                
                instances.append(instance)
            
            data_info['instances'] = instances
        
        return data_info


# For zero-shot inference with ceiling painting classes
@DATASETS.register_module()
class CeilingPaintingZeroShotDataset(CeilingPaintingSegmentationDataset):
    """Ceiling painting dataset for zero-shot inference with dynamic classes."""
    
    def __init__(self, 
                 custom_classes=None,
                 class_names_file=None,
                 **kwargs):
        """
        Args:
            custom_classes (list[str], optional): List of class names for zero-shot.
            class_names_file (str, optional): Path to file containing class names.
            **kwargs: Arguments for parent class.
        """
        # Load custom classes
        if custom_classes:
            self.METAINFO['classes'] = tuple(custom_classes)
        elif class_names_file:
            with open(class_names_file, 'r') as f:
                classes = [line.strip() for line in f.readlines()]
            self.METAINFO['classes'] = tuple(classes)
        
        super().__init__(**kwargs)