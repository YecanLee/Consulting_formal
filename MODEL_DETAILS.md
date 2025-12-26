## Detailed explanation of the source codes, (`ODISE` repo is used as an example here):
The two core components lie in the `ODISE/odise/modeling/meta_arch/odise.py` file, which are the `CategoryODISE` and `CaptionODISE` classes.

### `CategoryODISE` class:
This class is the main class for the category-level semantic segmentation task. This can be seen as a supervised learning task, we manually provide the labels for the semantic segmentation task/object detection task, and the model will learn to predict the labels for the given images.

The training flow can be summarized as follows:
```bash
1. Extract image features via backbone via Stable Diffusion features.
2. Generate mask proposals via Mask2Former head.
3. Compute mask embeddings via PooledMaskEmbed.
4. Match masks to category labels via CLIP.
5. Compute losses (mask classification loss, dice loss, classification loss).
```

### `CaptionODISE` class:
This class is the main class for the caption-level semantic segmentation task. This can be seen as a zero-shot learning task, we do not provide the labels for the semantic segmentation task/object detection task, and the model will learn to predict the labels for the given images based on the captions.

The training flow can be summarized as follows:
```bash
1. Extract image features via backbone via Stable Diffusion features.
2. Extract noun phrases from captions using NLTK (Line 1154-1159)
3. Compute mask-word grounding loss via MaskGroundingCriterion (Line 779-907)
4. Learn to align mask embeddings with word embeddings.
```