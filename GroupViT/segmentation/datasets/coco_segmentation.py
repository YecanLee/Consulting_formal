# Register the CeilingPaintingDataset 
# The customized dataset must be registered to be used.

from mmseg.datasets import CustomDataset
from mmseg.datasets.builder import DATASETS

# we need to register the dataset to the DATASETS registry
@DATASETS.register_module()
class CeilingPaintingDataset(CustomDataset):
    """
    Our Custom COCO format Ceiling Painting Dataset.
    Make sure your annotation files are segmentation masks, they are images in another folder. (This means that we have to create seperate masks files folder by ourselves and save the mask files into this folder)
    where pixel values correspond to class IDs.
    """

    # Classes names in our ceiling painting dataset
    CLASSES = ('#', 'mural', 'relief', 'brief', )
    
    # create a palette for each classes for visualization purposes
    PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],]

    def __init__(self, **kwargs):
        super(CeilingPaintingDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='_mask.png',
            **kwargs
            )

        # Sanity Check for this dataset
        assert self.CLASSES is not None, \
            '`CLASSES` in `CeilingPaintingDataset` can not be None.'
        assert self.PALETTE is not None, \
            '`PALETTE` in `CeilingPaintingDataset` can not be None.'
