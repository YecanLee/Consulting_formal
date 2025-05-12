"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/maskformer_model.py
"""
from typing import Tuple
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom
import torch.utils.checkpoint as cp
from .modeling.meta_arch.mask_adapter_head import build_mask_adapter

from .modeling.transformer_decoder.fcclip_transformer_decoder import MaskPooling, get_classification_logits
VILD_PROMPT = [
    "a photo of a {}.",
    "This is a photo of a {}",
    "There is a {} in the scene",
    "There is the {} in the scene",
    "a photo of a {} in the scene",
    "a photo of a small {}.",
    "a photo of a medium {}.",
    "a photo of a large {}.",
    "This is a photo of a small {}.",
    "This is a photo of a medium {}.",
    "This is a photo of a large {}.",
    "There is a small {} in the scene.",
    "There is a medium {} in the scene.",
    "There is a large {} in the scene.",
]


@META_ARCH_REGISTRY.register()
class FCCLIP(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        mask_adapter: nn.Module,
        weight_dict,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        train_metadata,
        test_metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        # FC-CLIP
        geometric_ensemble_alpha: float,
        geometric_ensemble_beta: float,
        ensemble_on_valid_mask: bool,
        # MASK-ADAPTER
        num_output_maps: int,
        iou_threshold: float,
        mask_threshold: float,
        num_gt_masks: int,
        num_pred_masks: int,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.mask_adapter = mask_adapter
        self.weight_dict = weight_dict
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.train_metadata = train_metadata
        self.test_metadata = test_metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image
        
        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        # FC-CLIP args
        self.mask_pooling = MaskPooling()
        self.geometric_ensemble_alpha = geometric_ensemble_alpha
        self.geometric_ensemble_beta = geometric_ensemble_beta
        self.ensemble_on_valid_mask = ensemble_on_valid_mask

        self.train_text_classifier = None
        self.test_text_classifier = None
        self.void_embedding = nn.Embedding(1, backbone.dim_latent) # use this for void
        self.num_output_maps = num_output_maps
        self.iou_threshold = iou_threshold
        self.mask_threshold = mask_threshold
        self.num_gt_masks = num_gt_masks
        self.num_pred_masks = num_pred_masks
        
        _, self.train_num_templates, self.train_class_names = self.prepare_class_names_from_metadata(train_metadata, train_metadata)
        self.category_overlapping_mask, self.test_num_templates, self.test_class_names = self.prepare_class_names_from_metadata(test_metadata, train_metadata)
        self.freeze_sem_seg()

    def freeze_sem_seg(self):
        for param in self.sem_seg_head.parameters():
            param.requires_grad = False
            
    def prepare_class_names_from_metadata(self, metadata, train_metadata):
        def split_labels(x):
            res = []
            for x_ in x:
                x_ = x_.replace(', ', ',')
                x_ = x_.split(',') # there can be multiple synonyms for single class
                res.append(x_)
            return res
        # get text classifier
        try:
            class_names = split_labels(metadata.stuff_classes) # it includes both thing and stuff
            train_class_names = split_labels(train_metadata.stuff_classes)
        except:
            # this could be for insseg, where only thing_classes are available
            class_names = split_labels(metadata.thing_classes)
            train_class_names = split_labels(train_metadata.thing_classes)
        train_class_names = {l for label in train_class_names for l in label}
        category_overlapping_list = []
        for test_class_names in class_names:
            is_overlapping = not set(train_class_names).isdisjoint(set(test_class_names))
            category_overlapping_list.append(is_overlapping)
        category_overlapping_mask = torch.tensor(
            category_overlapping_list, dtype=torch.long)
        
        def fill_all_templates_ensemble(x_=''):
            res = []
            for x in x_:
                for template in VILD_PROMPT:
                    res.append(template.format(x))
            return res, len(res) // len(VILD_PROMPT)
       
        num_templates = []
        templated_class_names = []
        for x in class_names:
            templated_classes, templated_classes_num = fill_all_templates_ensemble(x)
            templated_class_names += templated_classes
            num_templates.append(templated_classes_num) # how many templates for current classes
        class_names = templated_class_names
        #print("text for classification:", class_names)
        return category_overlapping_mask, num_templates, class_names

    def set_metadata(self, metadata):
        self.test_metadata = metadata
        self.category_overlapping_mask, self.test_num_templates, self.test_class_names = self.prepare_class_names_from_metadata(metadata, self.train_metadata)
        self.test_text_classifier = None
        return

    def get_text_classifier(self):
        if self.training:
            if self.train_text_classifier is None:
                text_classifier = []
                # this is needed to avoid oom, which may happen when num of class is large
                bs = 128
                for idx in range(0, len(self.train_class_names), bs):
                    text_classifier.append(self.backbone.get_text_classifier(self.train_class_names[idx:idx+bs], self.device).detach())
                text_classifier = torch.cat(text_classifier, dim=0)

                # average across templates and normalization.
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                text_classifier = text_classifier.reshape(text_classifier.shape[0]//len(VILD_PROMPT), len(VILD_PROMPT), text_classifier.shape[-1]).mean(1)
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                
                self.train_text_classifier = text_classifier
            return self.train_text_classifier, self.train_num_templates
        else:
            if self.test_text_classifier is None:
                text_classifier = []
                # this is needed to avoid oom, which may happen when num of class is large
                bs = 128
                for idx in range(0, len(self.test_class_names), bs):
                    text_classifier.append(self.backbone.get_text_classifier(self.test_class_names[idx:idx+bs], self.device).detach())
                text_classifier = torch.cat(text_classifier, dim=0)

                # average across templates and normalization.
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                text_classifier = text_classifier.reshape(text_classifier.shape[0]//len(VILD_PROMPT), len(VILD_PROMPT), text_classifier.shape[-1]).mean(1)
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                self.test_text_classifier = text_classifier
            return self.test_text_classifier, self.test_num_templates

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())
        mask_adapter = build_mask_adapter(cfg, cfg.MODEL.MASK_ADAPTER.NAME)

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        cosine_weight = cfg.MODEL.MASK_ADAPTER.COS_WEIGHT
        
        weight_dict = {"loss_ce": class_weight, "loss_cosine": cosine_weight}

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "mask_adapter": mask_adapter,
            "weight_dict": weight_dict,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "train_metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "test_metadata": MetadataCatalog.get(cfg.DATASETS.TEST[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "geometric_ensemble_alpha": cfg.MODEL.FC_CLIP.GEOMETRIC_ENSEMBLE_ALPHA,
            "geometric_ensemble_beta": cfg.MODEL.FC_CLIP.GEOMETRIC_ENSEMBLE_BETA,
            "ensemble_on_valid_mask": cfg.MODEL.FC_CLIP.ENSEMBLE_ON_VALID_MASK,
            "num_output_maps": cfg.MODEL.MASK_ADAPTER.NUM_OUTPUT_MAPS,
            "iou_threshold": cfg.MODEL.MASK_ADAPTER.IOU_THRESHOLD,
            "mask_threshold": cfg.MODEL.MASK_ADAPTER.MASK_THRESHOLD,
            "num_gt_masks": cfg.MODEL.MASK_ADAPTER.NUM_GT_MASKS,
            "num_pred_masks": cfg.MODEL.MASK_ADAPTER.NUM_PRED_MASKS,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        
        features = self.backbone(images.tensor)
        
        clip_feature = features['clip_vis_dense']
        
        text_classifier, num_templates = self.get_text_classifier()
        
        text_classifier = torch.cat([text_classifier, F.normalize(self.void_embedding.weight, dim=-1)], dim=0)
        
        
        features['text_classifier'] = text_classifier
        features['num_templates'] = num_templates
        
        with torch.no_grad():
            outputs = self.sem_seg_head(features)
        
            clip_vis_dense = self.visual_prediction_forward_convnext_2d(clip_feature)
        
        if self.training:
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets, masks, labels = self.prepare_targets_for_maskadapter(gt_instances, images)
            else:
                targets = None
                
            mask_pred_results = outputs["pred_masks"]
            mask_cls_results = outputs["pred_logits"]

            src_masks, target_masks, mask_labels = self.match_via_iou(mask_pred_results, mask_cls_results, targets, iou_threshold=self.iou_threshold,max_matches=self.num_pred_masks)
            binary_src_masks = src_masks.sigmoid() > self.mask_threshold
            binary_src_masks = binary_src_masks.float()
            
            binary_src_masks = F.interpolate(binary_src_masks, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            target_masks = F.interpolate(target_masks, size=masks.shape[-2:], mode='bilinear', align_corners=False)

            mask_pred = torch.cat((masks, binary_src_masks, target_masks),dim=1)
            all_labels = torch.cat((labels, mask_labels, mask_labels),dim=1)
                        
            outputs = self.mask_adapter(clip_vis_dense, mask_pred)
            
            maps_for_pooling = F.interpolate(outputs, size=clip_feature.shape[-2:],
                                                mode='bilinear', align_corners=False)
            
            if "convnext" in self.backbone.model_name.lower():
                B,C = clip_feature.size(0),clip_feature.size(1)
                N = maps_for_pooling.size(1)
                num_instances = N // self.num_output_maps
                maps_for_pooling = F.softmax(F.logsigmoid(maps_for_pooling).view(B, N,-1), dim=-1)
                pooled_clip_feature = torch.bmm(maps_for_pooling, clip_feature.view(B, C, -1).permute(0, 2, 1))
                pooled_clip_feature = self.backbone.visual_prediction_forward(pooled_clip_feature)
                pooled_clip_feature = (pooled_clip_feature.reshape(B, num_instances, self.num_output_maps, -1).mean(dim=-2).contiguous())
            else:
                raise NotImplementedError

            loss_cosine_similarity = self.cosine_similarity_loss(pooled_clip_feature[:, 16:24, :], pooled_clip_feature[:, 24:, :].detach())

            mask_cls_results = get_classification_logits(pooled_clip_feature, text_classifier, self.backbone.clip_model.logit_scale, num_templates)
                        
            # bipartite matching-based loss
            losses = self.cross_entropy_loss(mask_cls_results, all_labels)
            losses.update(loss_cosine_similarity)
            
            for k in list(losses.keys()):
                if k in self.weight_dict:
                    losses[k] *= self.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            mask_pred_results = outputs["pred_masks"]
            mask_cls_results = outputs["pred_logits"]

            binary_masks = mask_pred_results.sigmoid() > self.mask_threshold
            
            outputs = self.mask_adapter(clip_vis_dense, binary_masks)
            
            maps_for_pooling = F.interpolate(outputs, size=clip_vis_dense.shape[-2:],
                                                mode='bilinear', align_corners=False)
            if "convnext" in self.backbone.model_name.lower():
                B,C = clip_feature.size(0),clip_feature.size(1)
                N = maps_for_pooling.size(1)
                num_instances = N // self.num_output_maps
                maps_for_pooling = F.softmax(F.logsigmoid(maps_for_pooling).view(B, N,-1), dim=-1)
                pooled_clip_feature = torch.bmm(maps_for_pooling, clip_feature.view(B, C, -1).permute(0, 2, 1))
                pooled_clip_feature = self.backbone.visual_prediction_forward(pooled_clip_feature)
                pooled_clip_feature = (pooled_clip_feature.reshape(B,num_instances, self.num_output_maps, -1).mean(dim=-2).contiguous())
            else:
                raise NotImplementedError

            out_vocab_cls_results = get_classification_logits(pooled_clip_feature, text_classifier, self.backbone.clip_model.logit_scale, num_templates)

            in_vocab_cls_results = mask_cls_results[..., :-1] # remove void
            out_vocab_cls_results = out_vocab_cls_results[..., :-1] # remove void
            # Reference: https://github.com/NVlabs/ODISE/blob/main/odise/modeling/meta_arch/odise.py#L1506
            out_vocab_cls_probs = out_vocab_cls_results.softmax(-1)
            in_vocab_cls_results = in_vocab_cls_results.softmax(-1)
            category_overlapping_mask = self.category_overlapping_mask.to(self.device)


            alpha = self.geometric_ensemble_alpha
            beta = self.geometric_ensemble_beta

            cls_logits_seen = (
                (in_vocab_cls_results ** (1 - alpha) * out_vocab_cls_probs**alpha).log()
                * category_overlapping_mask
            )
            cls_logits_unseen = (
                (in_vocab_cls_results ** (1 - beta) * out_vocab_cls_probs**beta).log()
                * (1 - category_overlapping_mask)
            )
            cls_results = cls_logits_seen + cls_logits_unseen
            # This is used to filtering void predictions.
            is_void_prob = F.softmax(mask_cls_results, dim=-1)[..., -1:]
            mask_cls_probs = torch.cat([
                cls_results.softmax(-1) * (1.0 - is_void_prob),
                is_void_prob], dim=-1)
            mask_cls_results = torch.log(mask_cls_probs + 1e-8)

            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            del outputs
            del features
            torch.cuda.empty_cache()
            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                # semantic segmentation inference
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                    processed_results[-1]["sem_seg"] = r

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["panoptic_seg"] = panoptic_r
                
                # instance segmentation inference
                if self.instance_on:
                    instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["instances"] = instance_r

            return processed_results

    def prepare_targets_for_maskadapter(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        masks_list = []
        labels_list = []

        num_masks = self.num_gt_masks  
        min_mask_area = 0  

        for targets_per_image in targets:
            gt_masks = targets_per_image.gt_masks
            if isinstance(gt_masks, BitMasks):
                gt_masks = gt_masks.tensor
            valid_mask_indices = [i for i, mask in enumerate(gt_masks) if mask.sum() > min_mask_area]  # 筛选掉面积小于阈值的掩码

            if len(valid_mask_indices) > 0:
                valid_gt_masks = gt_masks[valid_mask_indices]
                valid_gt_classes = targets_per_image.gt_classes[valid_mask_indices]

                padded_masks = torch.zeros((valid_gt_masks.shape[0], h_pad, w_pad), dtype=valid_gt_masks.dtype, device=valid_gt_masks.device)
                padded_masks[:, :valid_gt_masks.shape[1], :valid_gt_masks.shape[2]] = valid_gt_masks
                new_targets.append(
                    {
                        "labels": valid_gt_classes,
                        "masks": padded_masks,
                    }
                )

                total_masks = torch.zeros((num_masks, h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
                selected_labels = torch.full((num_masks,), -1, dtype=valid_gt_classes.dtype, device=gt_masks.device)

                if valid_gt_masks.shape[0] > num_masks:
                    selected_indices = torch.randperm(valid_gt_masks.shape[0])[:num_masks]
                    for idx, mask_idx in enumerate(selected_indices):
                        total_masks[idx, :valid_gt_masks[mask_idx].shape[0], :valid_gt_masks[mask_idx].shape[1]] = valid_gt_masks[mask_idx]
                        selected_labels[idx] = valid_gt_classes[mask_idx]
                else:
                    for idx in range(valid_gt_masks.shape[0]):
                        total_masks[idx, :valid_gt_masks[idx].shape[0], :valid_gt_masks[idx].shape[1]] = valid_gt_masks[idx]
                        selected_labels[idx] = valid_gt_classes[idx]
                    
                    for idx in range(valid_gt_masks.shape[0], num_masks):
                        total_masks[idx] = torch.zeros((h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
                        selected_labels[idx] = -1
            else:
                total_masks = torch.zeros((num_masks, h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
                selected_labels = torch.full((num_masks,), -1, dtype=torch.long, device=gt_masks.device)
                
                padded_masks = torch.zeros((0, h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
                valid_gt_classes = torch.zeros((0), device=gt_masks.device)
                new_targets.append(
                    {
                        "labels": valid_gt_classes,
                        "masks": padded_masks,
                    }
                )

            masks_list.append(total_masks)
            labels_list.append(selected_labels)

        masks = torch.stack(masks_list, dim=0)
        labels = torch.stack(labels_list, dim=0)
        labels = labels.long()

        return new_targets, masks, labels
    
    @torch.no_grad()                            
    def match_via_iou(self, mask_pred_results, mask_cls_results, targets, iou_threshold=0.7, max_matches=8):
        batch_size = mask_pred_results.shape[0]
        matched_src_masks = []
        matched_target_masks = []
        matched_labels = []

        for b in range(batch_size):
            tgt_label = targets[b]["labels"]  
            tgt_mask = targets[b]["masks"].to(mask_pred_results.device) 
            num_tgt_masks = tgt_mask.shape[0]

            pred_mask = mask_pred_results[b]  
            pred_cls = mask_cls_results[b] 
            num_pred_masks = pred_mask.shape[0]

            tgt_mask = F.interpolate(tgt_mask[:, None].float(), size=pred_mask.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)

            pred_mask_flat = pred_mask.flatten(1)
            tgt_mask_flat = tgt_mask.flatten(1)

            with torch.no_grad():
                ious = compute_mask_iou(pred_mask_flat, tgt_mask_flat)

            matched_pred_idx = []
            matched_tgt_idx = []
            
            for j in range(num_tgt_masks):
                valid_pred_idx = (ious[:, j] > iou_threshold).nonzero(as_tuple=True)[0]
                if len(valid_pred_idx) > 0:
                    random_idx = torch.randint(0, len(valid_pred_idx), (1,)).item()
                    best_pred_idx = valid_pred_idx[random_idx]
                    matched_pred_idx.append(best_pred_idx.item())
                    matched_tgt_idx.append(j)


            if len(matched_pred_idx) > max_matches:
                selected_indices = torch.randperm(len(matched_pred_idx))[:max_matches]
                matched_pred_idx = [matched_pred_idx[i] for i in selected_indices]
                matched_tgt_idx = [matched_tgt_idx[i] for i in selected_indices]

            if len(matched_pred_idx) < max_matches:
                num_to_add = max_matches - len(matched_pred_idx)
                
                matched_src_masks.append(
                    torch.cat([pred_mask[matched_pred_idx], 
                            torch.zeros((num_to_add, *pred_mask.shape[1:]), device=pred_mask.device)], dim=0)
                )
                matched_target_masks.append(
                    torch.cat([tgt_mask[matched_tgt_idx], 
                            torch.zeros((num_to_add, *tgt_mask.shape[1:]), device=tgt_mask.device)], dim=0)
                )
                matched_labels.append(
                    torch.cat([tgt_label[matched_tgt_idx], 
                            torch.full((num_to_add,), -1, dtype=tgt_label.dtype, device=tgt_label.device)], dim=0)
                )
            else:
                matched_src_masks.append(pred_mask[matched_pred_idx])
                matched_target_masks.append(tgt_mask[matched_tgt_idx])
                matched_labels.append(tgt_label[matched_tgt_idx])

        matched_src_masks = torch.stack(matched_src_masks, dim=0) 
        matched_target_masks = torch.stack(matched_target_masks, dim=0)
        matched_labels = torch.stack(matched_labels, dim=0)

        return matched_src_masks, matched_target_masks, matched_labels

    def visual_prediction_forward_convnext_2d(self, x):
        
        clip_vis_dense = self.backbone.clip_model.visual.trunk.head.norm(x)
        clip_vis_dense = self.backbone.clip_model.visual.trunk.head.drop(clip_vis_dense.permute(0, 2, 3, 1))
        clip_vis_dense = self.backbone.clip_model.visual.head(clip_vis_dense).permute(0, 3, 1, 2)
        
        return clip_vis_dense
    
    def cross_entropy_loss(self, mask_cls_results, labels):
        
        if torch.all(labels == -1):
            loss_ce = mask_cls_results.sum() * 0.0 
        else:
            loss_ce = F.cross_entropy(mask_cls_results.transpose(1, 2), labels.to(torch.int64), ignore_index=-1)  #remove celoss weight because of multiple datasets training

        losses = {"loss_ce": loss_ce}
        return losses
    
    def cosine_similarity_loss(self, pred_features, gt_features):
    
        cosine_similarity_loss = {}
        
        cosine_sim = F.cosine_similarity(pred_features, gt_features, dim=-1)
        cosine_similarity_loss[f"loss_cosine"] = 1 - cosine_sim.mean()
        return cosine_similarity_loss

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):

        
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()
        num_classes = len(self.test_metadata.stuff_classes)
        keep = labels.ne(num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.test_metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        # if this is panoptic segmentation
        if self.panoptic_on:
            num_classes = len(self.test_metadata.stuff_classes)
        else:
            num_classes = len(self.test_metadata.thing_classes)
        labels = torch.arange(num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // num_classes
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.test_metadata.thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result
    
    def visual_prediction_forward_convnext(self, x):
        batch, channel, h, w = x.shape
        x = x.reshape(batch*h*w, channel).unsqueeze(-1).unsqueeze(-1) # fake 2D input
        x = self.backbone.clip_model.visual.trunk.head(x)
        x = self.backbone.clip_model.visual.head(x)
        return x.reshape(batch, h, w, x.shape[-1]).permute(0,3,1,2) 


    
def compute_mask_iou(pred_masks, tgt_masks):
    
    pred_masks = pred_masks.sigmoid()
    
    binarized_pred_masks = (pred_masks >= 0.4).float()
    binarized_tgt_masks = (tgt_masks > 0.5).float()

    intersection = torch.einsum('nc,mc->nm', binarized_pred_masks, binarized_tgt_masks)
    
    pred_area = binarized_pred_masks.sum(dim=-1)  
    tgt_area = binarized_tgt_masks.sum(dim=-1)    
    
    union = pred_area[:, None] + tgt_area[None, :] - intersection
    
    iou_matrix = intersection / (union + 1e-6)
    
    return iou_matrix


