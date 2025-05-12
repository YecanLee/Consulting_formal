## Getting Started with Mask-Adapter

This document provides a brief intro of the usage of Mask-Adapter.

Please see [Getting Started with Detectron2](https://github.com/facebookresearch/detectron2/blob/master/GETTING_STARTED.md) for full usage.


### Inference Demo with Pre-trained Models

We provide `demo.py` that is able to demo builtin configs. Run it with:
```
cd demo/
python demo.py \
  --input input1.jpg input2.jpg \
  [--other-options]
  --opts MODEL.WEIGHTS /path/to/checkpoint_file
```
The configs are made for training, therefore we need to specify `MODEL.WEIGHTS` to a model from model zoo for evaluation.
This command will run the inference and show visualizations in an OpenCV window.

For details of the command line arguments, see `demo.py -h` or look at its source code
to understand its behavior. Some common arguments are:
* To run __on your webcam__, replace `--input files` with `--webcam`.
* To run __on a video__, replace `--input files` with `--video-input video.mp4`.
* To run __on cpu__, add `MODEL.DEVICE cpu` after `--opts`.
* To save outputs to a directory (for images) or a file (for webcam or video), use `--output`.


### Ground-truth Warmup Training 
We provide the script `train_net_maskadapter.py` to train the mask-adapter using ground-truth masks.To train a model with `train_net_maskadapter.py`, first set up the corresponding datasets as described in [datasets/README.md](https://chatgpt.com/c/datasets/README.md) , and then run the following command:

```
python train_net_maskadapter.py --num-gpus 4 \
  --config-file configs/ground-truth-warmup/mask-adapter/mask_adapter_convnext_large_cocopan_eval_ade20k.yaml
```

For the MAFTP model, run:


```
python train_net_maskadapter.py --num-gpus 4 \
  --config-file configs/ground-truth-warmup/mask-adapter/mask_adapter_maft_convnext_large_cocostuff_eval_ade20k.yaml \
  MODEL.WEIGHTS /path/to/maftp_l.pth
```

The configurations are set for 4-GPU training. Since we use the ADAMW optimizer, it is unclear how to scale the learning rate with batch size. If training with a single GPU, you will need to manually adjust the learning rate and batch size:


```
python train_net_maskadapter.py \
  --config-file configs/ground-truth-warmup/mask-adapter/mask_adapter_convnext_large_cocopan_eval_ade20k.yaml \
  --num-gpus 1 SOLVER.IMS_PER_BATCH SET_TO_SOME_REASONABLE_VALUE SOLVER.BASE_LR SET_TO_SOME_REASONABLE_VALUE
```

### Combining Mask-Adapter Weights with Mask2Former 

Since the ground-truth warmup phase for training the mask-adapter does not involve training Mask2Former, the weights obtained in the first phase will not include Mask2Former weights. To combine the weights, run the following command:


```
python tools/weight_fuse.py \
  --model_first_phase_path /path/to/first_phase.pth \
  --model_sem_seg_path /path/to/maftp_l.pth \
  --output_path /path/to/maftp_l_withadapter.pth
```

### Mixed-Masks Training 
For the mixed-masks training phase, we provide two scripts: `train_net_fcclip.py` and `train_net_maftp.py`, which train the mask-adapter for FC-CLIP and MAFTP models, respectively. These two models use different backbones (CLIP) and training source data.
For FC-CLIP, run:


```
python train_net_fcclip.py --num-gpus 4 \
  --config-file configs/mixed-mask-training/fc-clip/fcclip/fcclip_convnext_large_eval_ade20k.yaml MODEL.WEIGHTS /path/to/checkpoint_file
```

For MAFTP, run:


```
python train_net_maftp.py --num-gpus 4 \
  --config-file configs/mixed-mask-training/maftp/semantic/train_semantic_large_eval_a150.yaml MODEL.WEIGHTS /path/to/checkpoint_file
```

To evaluate a model’s performance, for FC-CLIP, use:


```
python train_net_fcclip.py \
  --config-file configs/mixed-mask-training/fc-clip/fcclip/fcclip_convnext_large_eval_ade20k.yaml \
  --eval-only MODEL.WEIGHTS /path/to/checkpoint_file
```

For MAFTP, use:


```
python train_net_maftp.py \
  --config-file configs/mixed-mask-training/maftp/semantic/train_semantic_large_eval_a150.yaml \
  --eval-only MODEL.WEIGHTS /path/to/checkpoint_file
```