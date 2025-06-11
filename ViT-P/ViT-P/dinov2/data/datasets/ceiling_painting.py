import torch
import os
from PIL import Image 
import torchvision
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset
import numpy as np
from numpy import random
import glob


class CeilingPaintingDataset(Dataset):
    def __init__(self, 
                split, 
                n_points, 
                image_size, 
                dataset_root='../../../../../ceiling_dataset_for_ViT-P', 
                num_classes=None,
                augmentation=None):
        """
        Args:
            split: 'train' or 'val'
            n_points: number of points to sample
            image_size: size to resize images
            dataset_root: path to ceiling painting dataset
            num_classes: number of classes in ceiling painting dataset
        """
        if split == 'train':
            self.augmentation = transforms.Compose([
                transforms.RandomResizedCrop(size=image_size, scale=(0.5, 1), ratio=(0.75, 1.3333), 
                                           interpolation=torchvision.transforms.InterpolationMode.NEAREST, 
                                           antialias=True),
                transforms.RandomRotation((-60, 60)),
                transforms.RandomHorizontalFlip(p=0.5)
            ])
        else:
            self.augmentation = None
            
        self.split = split
        self.dataset_root = dataset_root
        self.n_points = n_points
        self.num_classes = num_classes
        
        # Work with both jpg and png format
        self.image_paths = sorted(glob.glob(os.path.join(dataset_root, 'images', split, '*.jpg')))
        if not self.image_paths:
            self.image_paths = sorted(glob.glob(os.path.join(dataset_root, 'images', split, '*.png')))
        
        print(f"Found {len(self.image_paths)} images for {split} split!")
        
        self.transform = torchvision.transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Get mask path
        img_basename = os.path.basename(img_path)
        # Remove file extension (.jpg or .png)
        img_name = os.path.splitext(img_basename)[0]
        mask_path = os.path.join(self.dataset_root, 'annotations', self.split, img_name + '_mask.png')
        
        # Load mask
        mask_np = Image.open(mask_path)
        
        # Apply augmentation if training
        if self.augmentation:
            image0, mask_np0 = self.augmentation(image, mask_np)
            mask_np0 = np.array(mask_np0)
            object_numbers0 = np.unique(mask_np0.reshape(-1), axis=0)
            
            # Check if augmentation resulted in empty mask
            if len(object_numbers0) == 1 and (0 in object_numbers0 or 255 in object_numbers0):
                mask_np = np.array(mask_np)
            else:
                image = image0
                mask_np = mask_np0
        else:
            mask_np = np.array(mask_np)
        
        # Transform image
        if self.transform:
            image = self.transform(image)
        
        # Get unique object/class IDs from mask
        object_numbers = np.unique(mask_np.reshape(-1), axis=0)
        
        # Remove background (0) and ignore (255) labels if present
        object_numbers = object_numbers[object_numbers != 0]
        object_numbers = object_numbers[object_numbers != 255]
        
        # If no valid objects, use background
        if len(object_numbers) == 0:
            object_numbers = np.array([0])
        
        # Sample points
        x = np.random.choice(object_numbers, size=self.n_points, replace=True)
        
        points = np.zeros((self.n_points, 2))
        label = np.zeros((self.n_points))
        
        h, w = mask_np.shape
        j = 0
        for i in x:
            if i == 0:  # Handle background
                # Sample random point from background
                bg_mask = mask_np == 0
                if bg_mask.any():
                    ori = np.where(bg_mask)
                    rand = random.randint(ori[0].shape[0])
                    points[j] = (ori[0][rand]/h, ori[1][rand]/w)
                else:
                    # If no background, sample random point
                    points[j] = (random.random(), random.random())
            else:
                ori = np.where(mask_np == i)
                if ori[0].shape[0] > 0:
                    rand = random.randint(ori[0].shape[0])
                    points[j] = (ori[0][rand]/h, ori[1][rand]/w)
                else:
                    # Fallback to random point
                    points[j] = (random.random(), random.random())
            
            # Map label IDs if needed (subtract 1 if labels start from 1)
            label[j] = i if i == 0 else i - 1
            j += 1
        
        # Normalize points to [-1, 1]
        points = 2 * points - 1
        
        return {
            "image": image,
            "points": torch.from_numpy(points).to(dtype=torch.float32),
            "label": torch.from_numpy(label).to(dtype=torch.float32),
        }