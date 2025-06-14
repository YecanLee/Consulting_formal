This repo includes our customized implementation of several Open Vocabulary Segmentation papers from some top Computer Vision Conferences.

We modified the model configurations and dataset loaders according to our own customized ceiling paiting datasets labelled by `Roboflow`.

## Starting Point
Please have some brief idea about what "Open Vocabulary Segmentation Task", "Super Resolution Task" and "Open Object Detection Task" are by reading some introduction blogs as well as some examples on Github. It will be helpful if you also understand some basic concepts of image storage format, i.e. coco-format.

## Introduction of our project pipeline
We include multiple features in our repo, this includes:
- Automatic image upscaling with Open-source Super Resolution Model, we only include the following models now:
  - `stablediffusionupscalepipeline` from `diffusers` package.
  - `DRCT` (DRCT: Saving Image Super-resolution away from Information Bottleneck) published in CVPR 2024 and is SOTA on several super resolution benchmarks.

- Open Vocabulary Segmentation Task, this includes:
  - `SAN` paper, published in CVPR 2024.
  - `MaskQCLIP` paper, published in CVPR 2024.
  - `MasQCLIP` paper, published in CVPR 2024.

- Open Object Detection Task, this includes:
  - `FC-Clip` paper, published in CVPR 2024.

- Open Vocabulary Object Detection Task, this includes:
  - `FC-Clip` paper, published in CVPR 2024.

- Multi-model pre-training with our customized ceiling painting dataset, this includes:
    - `GroupViT` paper, published in CVPR 2022.
    - `SegCLIP` paper, published in ICML 2023.

- Some side features and funny ideas, you can find them in the `Fancy_Ideas` folder. This includes the following parts right now:
    - `DragGAN` with GUI installation and usage case, you can manually drag a ceiling painting image to the target position. And create a short "video" which shows the whole process.
    - `google_veo_video`, which includes a python ipynb script to create a video by using one of the `ceiling_painting` dataset's image as part of the input prompt.
    - `diffuser_upscaler`, which uses an open-sourced upscaler to upscale a ceiling painting images which are super blured in the original dataset.


## Environment Installation
### SAN paper env installation instructions (An example for `detectron2` based model)
To install the environment we used in our projects's `SAN` paper implementation, please follow the instructions below, beware that the env installaton was only tested on a Linux machine, you may encounter specific issues when you try to install the `detectron2` package.

__Alert__! We installed `detectron2` package from source by cloning the Meta official repo first, you may encounter issues if you installed the package by using `conda` or follow the original official repo's implementation method. __We pinned the pytorch as well as other dependencies to other versions, which is different from the README.md in the original repo__.

We recommend you to use conda for reproducibility and version control reason.

```bash
# Please run the following command from the root directory of this repo.
conda create -n san python==3.11
conda activate san

# install torch and torchvision with corresponding python version.
# ignore the following line will cause installation failure for `detectron2`.
pip install -r san_requirements.txt

# install detectron2 from source 
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2

# install other dependencies for san model
cd SAN/san
pip install -r requirements.txt

# The following command is OPTIONAL
# Test if the detectron2 has been installed successfully
# python -c "import detectron2 ; print(detectron2.__version__)"
```

### SegCLIP paper env installation instructions (An example for `mm-lab` based model)
Since the `mm-lab` developed packages are really hard to install, we provide a detailed installation instruction for the `SegCLIP` paper. Please check the `Debug Guidance` section for more details. 


### 🪄Special packages installation instructions

#### `OneFormer`

`oneformer` used `natten` package which requries newest `torch` version (torch>=2.7), to compile with most of the packages used in our project, we pinned the `torch` installation command to `pip install torch==2.4`. To install the `natten` package which compiled with `torch==2.4`, please run the following command:

```bash
pip3 install natten==0.17.1+torch240cu121 -f https://shi-labs.com/natten/wheels
```

#### `ViT-P`

`ViT-P` used `xformers` package, to pin the `xformers` package to a version which is compatible with `torch==2.4`, please run the following command:

```bash
pip install xformers==0.0.28
```

## Folder Structure
### SAN paper folder structure
We concentrate on the different parts we modified compared to the original repo in the `SAN` folder. 

To add your config file for training and testing, please add your customized config `yaml` file to the `SAN/configs` folder. Please follow the naming convention, for example, you should name your base config file as `Base-<your_dataset_name>.yaml`. To add a specific config file for your training experiments, you can name it as `san_clip_vit_<your_dataset_name>.yaml`.

You could put your downloaded pretrained CLIP model weights in the `SAN/pretrained_weights` folder.

#### Customized Dataset
You need to register your dataset first in order to compile with the `detectron2` framework.

You could put your downloaded dataset under the path `SAN/san/data/` folder. You can also put your dataset under other path as long as `detectron2` can find it.

To register your own custom dataset, you need to add one registration file in the `SAN/san/data/datasets` folder. Please follow the naming convention, for example, you should name your registration file as `register_<your_dataset_name>.py`.

If your labeled dataset is in `COCO` format, you could use the `register_coco_instances` function to register your dataset. This is the easiest way to register your dataset. We used this method since `Roboflow` provides the dataset to be downloaded in `COCO` format.

Please do not forget to call the defined registration function in the `SAN/san/data/datasets/register_<your_dataset_name>.py` file. Otherwise, the model will not be able to find your dataset.  

## 🚀Train Command Encyclopedia
### ⭐CAT-Seg
To finetune the pretrained `CAT-Seg` model with our customized ceiling painting dataset, please run the following command:

```bash
cd CAT-Seg
mkdir pretrained_weights   

# download the pretrained weights with ViT-L backbone
wget -P pretrained_weights https://huggingface.co/spaces/hamacojr/CAT-Seg-weights/resolve/main/model_large.pth   
# Or
# wget -P pretrained_weights https://huggingface.co/spaces/hamacojr/CAT-Seg-weights/resolve/main/model_base.pth

# finetune the pretrained model with ViT-L backbone
python train_net.py --config configs/ceiling_config.yaml  \
--num-gpus 1 \
--dist-url "auto"  \
--resume MODEL.WEIGHTS pretrained_weights/model_large.pth \
OUTPUT_DIR output    

# finetune the pretrained model with ViT-B backbone
# python train_net.py 
# --config configs/ceiling_config.yaml  \
# --num-gpus 1 \
# --dist-url "auto"  \
# --resume MODEL.WEIGHTS pretrained_weights/model_base.pth \
# OUTPUT_DIR output
```
### ⭐ebseg
To finetune the pretrained `ebseg` model with our customized ceiling painting dataset, please run the following command:
```bash
python train_net.py  \
--config-file configs/ebseg/ceiling_painting.yaml  \
--num-gpus 1  \
OUTPUT_DIR output
```   

### ⭐fc-clip
To finetune the pretrained `fc-clip` model with our customized ceiling painting dataset, please run the following command:
```bash
python train_net.py  \
--config-file configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_large_train_ceiling.yaml \
--resume \
--num-gpus 1 \
SOLVER.IMS_PER_BATCH 6 \ # tested on 1 GPU with 24GB vram
SOLVER.BASE_LR 0.00006 \
MODEL.WEIGHTS pretrained_weights/fcclip_cocopan.pth
```   

### ⭐GroupViT
Since the `GroupViT` repo is trained with multiple `text-to-image` pairs datasets, we will only include the inference command in the `GroupViT` folder right now at this moment, the training command will come soon, stay tuned! 
```bash
# GroupViT inference command
./tools/dist_launch.sh main_seg.py \
configs/group_vit_gcc_ceiling.yml 1 \
--resume \
pretrained_weights/group_vit_gcc_yfcc_30e-879422e0.pth \
--opts evaluate.seg.cfg=segmentation/configs/_base_/datasets/ceiling_painting.py
```   

### ⭐maskclippp
To finetune the pretrained `MaskCLIP++` model with our customized ceiling painting dataset, please run the following command:
```bash
python train_net.py \
--config-file configs/mask_clippp/train_ceiling_mask_clippp.yaml \
--num-gpus 1 \
```

### ⭐maskclip
To finetune the pretrained `MaskCLIP` model with our customized ceiling painting dataset, please run the following command:
```bash
python train_net.py \
--num-gpus 1 \
--config-file configs/coco/ceiling_maskformer2_R50_bs2_10ep.yaml
```

### ⭐MAFT-Plus
To finetune the pretrained `MAFT-Plus` model with our customized ceiling painting dataset, please run the following command:
```bash
python train_net.py \
--config-file configs/semantic/train_ceiling_semantic_large.yaml  \
--num-gpus 1  \
MODEL.WEIGHTS pretrained_weights/maftp_l.pth
```

### ⭐MasQCLIP
To finetune the pretrained `MasQCLIP` model with our customized ceiling painting dataset, please run the following command:
```bash
python train_net.py \
--num-gpus 1 \
--config-file configs/base-novel/coco-semantic/student_ceiling_R50_30k_base48.yaml \
OUTPUT_DIR output \
MODEL.WEIGHTS pretrained_weights/cross_dataset.pth
```

### ⭐MaskAdapter
The MaskAdapter model has a mixed-masks training phase and another finetuning phase, we provide two scripts: `train_net_fcclip.py` and `train_net_maftp.py`, which train the mask-adapter for FC-CLIP and MAFTP models, respectively. These two models use different backbones (CLIP) and training source data. 

To train the MaskAdapter model for FC-CLIP, please run the following command:
```bash
python train_net_fcclip.py \
--num-gpus 1 \
--config-file configs/mixed-mask-training/fc-clip/fcclip/fcclip_convnext_large_train_ceiling.yaml \
MODEL.WEIGHTS pretrained_weights/fcclip_cocopan.pth
```

To train the MaskAdapter model for MAFTP, please run the following command:
```bash
python train_net_maftp.py \
--num-gpus 1   \
--config-file configs/mixed-mask-training/maftp/semantic/train_semantic_large_train_ceiling.yaml  \
MODEL.WEIGHTS pretrained_weights/maftp_l.pth
```

To finetune the MaskAdapter itself, please run the following command:
```bash
python train_net.py \
--config-file configs/mixed-mask-training/fc-clip/fcclip/fcclip_convnext_large_train_ceiling.yaml \
MODEL.WEIGHTS pretrained_weights/fcclip_cocopan.pth
```

### ⭐SegCLIP
To evaluate the SegCLIP model based on pretrained weights, please run the following command:
```bash
CUDA_VISIBLE_DEVICES=0 \
python -m torch.distributed.launch \
--nproc_per_node=1 \
main_seg_zeroshot.py \
--dataset ceiling_painting \
--init_model checkpoints/segclip.bin 
```

### ⭐SCAN 
To finetune the SCAN model, please run the following command:
```bash
python train_net.py  \
--num-gpu 1 \
--config-file configs/scan_vitL_ceiling.yaml
```

### ⭐sed
To finetune the sed model, please run the following command:
```bash
python train_net.py \
--config configs/convnextL_768_ceiling.yaml \
--num-gpus 1 \
--dist-url "auto" \
OUTPUT_DIR output \
MODEL.WEIGHTS pretrained_weights/sed_model_large.pth
```

### ⭐ov-seg 
To finetune the ov-seg model, please run the following command:
```bash
python train_net.py \
--num-gpu 1 \
--config-file configs/ovseg_swinB_vitL_bs2_ceiling.yaml \
MODEL.WEIGHTS pretrained_weights/ovseg_swinbase_vitL14_ft_mpt.pth
```

### ⭐zsseg_baseline
To finetune the zsseg_baseline model, please run the following command:
```bash
python train_net.py \
--config-file configs/coco-stuff-164k-156/ceiling_proposal_classification_learn_prompt_bs2_1k.yaml \
--num-gpus 1
```

### ⭐Zegformer
To finetune the zegformer model, please run the following command:
```bash
python train_net.py   \
--config-file configs/coco-stuff/zegformer_R101_bs1_60k_vit16_ceiling.yaml   \
--num-gpus 1 \
SOLVER.IMS_PER_BATCH 2 \
SOLVER.BASE_LR 0.0001
```


## 🔥Demo Command Encyclopedia

### Built upon `mm-lab`
- SegCLIP   
To run the `main_seg_vis.py` script in `SegCLIP` folder to do inference upon __one__ image, please run the following command:

```bash
# ALERT: multiple images are not supported yet.
python main_seg_vis.py --input YOUR_IMAGE_PATH \
--device cuda:0 \
--vis input CHOOSE_VIS_MODE \
--dataset CHOOSE_DATASET \
--init_model YOUR_TRAINED_WEIGHTS_PATH
```

### Built upon `detectron2`
- OV-Seg   
To run the `demo.py` script in `ov-seg` folder to do inference upon one or more images, please run the following command:

```bash
python demo.py --config-file configs/ovseg_swinB_vitL_ceiling_demo.yaml \
--class-names 'NAME' 'YOUR' 'CLASS' 'NAMES' ... \
--input YOUR_IMAGE_PATH1 YOUR_IMAGE_PATH2 ... \
--output ./pred \
--opts MODEL.WEIGHTS YOUR_TRAINED_WEIGHTS_PATH 
```

There is a numpy version issue in the original repo, please pin to our version to use the demo.

## Fancy Ideas Part Usage Guidance
To use the `DragGAN` GUI on our own ceiling painting dataset instead of the default images from `DragGAN` repo, you need to fist perform GAN inversion by using one of the available tools. We used `PTI` to perform the GAN inversion. Please refer to the `Fancy_Ideas/PTI/` for more details.

## 🚧In construction and Alert
Some of the models here require more than one train scripts, this section is mainly used to record which repo requires more than one train scripts.

The `MasQCLIP` requires two Progressive Distillation traning and one Mask-Q Tuning, they all used the `train_net.py` script, but with different config files.

~~The `MasQCLIP` was trained with only two classes, `backdground` and `non-background`. This means that the model may have a different class_emb shape compared to our customized ceiling painting dataset. If we only have "mural" as the only class, this is fine. Otherwise, we need to modify the `class_emb` layer in the `mask_distill.py` file `MaskFormer` class to match our dataset.~~ 
The `MasQCLIP` was built upon the instance segmentation task, the original source codes were using input["instances"] to get the ground truth masks. To use the model for our custom ceiling painting segmentation dataset, we offered a modified segmentation model in our codes. Please refer to the `MasQCLIP/masqclip/mask_distill.py` file for more details.

The mmcv-full from the GroupViT and SegCLIP are only compiled with numpy version smaller than 2.0.0, otherwise the installation of the mmcv-full will fail.

We pinned our mmcv as well as our mmsegmentation package to older versions since those papers were developed based on 1.X version instead of 2.X version.

## 📝TODO(A quick to-do list)
- [ ] Check if the evaluators are working for different models. This may require us to change the way we register the `ceiling_painting` dataset. Or we may need to modify the metadata `evaluator_type` of the dataset.
- [ ] Add pre-commit hooks to check the code style. We may only check the files we changed during the whole project instead of every file.
- [ ] Add dockerfile for each model. We will only offer one training dockerfile and one deployment dockerfile.

- [x] Pin the debug guidance in fc-clip as an example, this should be pinned very precisely to which line and which file should be modified if `detectron2` package is used to trian and validate the model proposed in the paper.

- [ ] Add some open sourced `video generation` models into the `Fancy_Ideas` folder. I propose we could use this [new paper](https://github.com/thu-ml/RIFLEx) since this new `ICML` paper can generate a little bit longer video compared to the previous models.

## 🐛Debug Guidance
### mmcv and mmsegmentation installation debug guidance
The repos which were built upon the `mm-lab` series of packages require extra attention when you try to install the environment since the `mmcv` or `mmsegmentation` package are compiled with C++ backend, which may cause some issues when you try to install the environment.

We pinned our environment installation as a combination of `segclip` and `groupvit` repo's environment installation instructions. But we only used `apex` installation part of the `groupvit` repo's installation instructions. However, extra attention should be paid to this part.

You may run the following command to initialize the environment.

```bash
conda create -n segclip python=3.8 -y
conda activate segclip
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install mmcv-full==1.3.14 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html
pip install mmsegmentation==0.18.0
pip install webdataset==0.1.103
pip install timm==0.4.12
pip install opencv-python==4.4.0.46 termcolor==1.1.0 diffdist einops omegaconf
pip install nltk ftfy regex tqdm
pip install prefetch_generator
pip install Pillow==8.2.0
```
Please be aware about the conda channel you are using for installing the `torch` and other important packages like `mmcv` and `mmsegmentation`. They are from different channels, please only use the command we provided above, official website guidance of the installation of those packages may not work.

You may then need to install the `apex` package from the `groupvit` repo. The official github repo's command does not work, you may encounter the following issue when you run the installation command from the `GroupViT` repo:

```bash
apex RuntimeError: Cuda extensions are being compiled with a version of Cuda that does not match the version used to compile Pytorch binaries. Pytorch binaries were compiled with Cuda 11.1.
```

To fix this issue, please run the following command to install the `apex` package.

```bash
git clone https://github.com/ptrblck/apex.git
cd apex
git checkout apex_no_distributed
pip install -v --no-cache-dir ./
``` 

Since this is an `outdated` version of the `apex` package, you will encounter some issues when you try to run the `GroupViT` repo inference command, especially this `apex` package was built upon an older version of `pytorch` package. You will encounter this issue when you run the inference command of the `GroupViT` repo:

```bash
AttributeError: module 'torch.nn' has no attribute 'backends'
```

This error means that we need to modify the source code of the `apex` package, please beware that since we are using a different branch of this `outdated` cloned `apex` repo, you should modify the source code of the `apex` installed in your conda environment.

Suppose your conda environment path is `/home/user/anaconda3/envs/segclip`, first we need to use `vim` to open the `apex` source code file, please run the following command:

```bash
sudo apt install vim
vim /home/user/anaconda3/envs/segclip/lib/python3.8/site-packages/apex/amp/utils.py
```

Then in this file, starting from line 132, there are three consecutive functions which used `torch.nn.backends.backend.FunctionBackend`, this is a fairly old legacy version of `pytorch`, please change those three functions with the following one:

```bash 
def has_func(mod, fn):
    if isinstance(mod, dict):
      return fn in mod
    else:
      return hasattr(mod, fn)

def get_func(mod, fn):
    if isinstance(mod, dict):
      return mod[fn]
    else:
    return getattr(mod, fn)

def set_func(mod, fn, new_fn):
    if isinstance(mod, dict):
        mod[fn] = new_fn
    else:
        setattr(mod, fn, new_fn)
```

Then you need to save the file and exit from `vim`.

After this, you can run the `GroupViT` repo's inference command again.

```bash
./tools/dist_launch.sh main_seg.py configs/group_vit_gcc_ceiling.yml 1 --resume pretrained_weights/group_vit_gcc_yfcc_30e-879422e0.pth --opts evaluate.seg.cfg=segmentation/configs/_base_/datasets/ceiling_painting.py
```

And this should work.

To add new datasets into the repo built upon the `detectron2` framework, you should always try to pin to the way we used in the `fc-clip` repo folder first, this is the vanilla method from the `detectron2` package, in the case of encountering errors, try to register the dataset by using the `register_ceiling_dataset` defined in the `consulting_pro/fc-clip/fcclip/data/datasets/register_ceiling.py` file. Some of the models may require you to use a different dataset register format during the training and inference phase.

### dataset format debug guidance
If you encounter the following error during the __training__ or __inference__ phase:   
```bash
AttributeError: Cannot find field 'gt_masks' in the given Instances
```
Please check your config file and add this line in your `MODEL` section:
```bash
MODEL:
  ...
  MASK_ON: True
```

### model source code debug guidance
If you encounter the following error during the __training__ or __inference__ phase:
```bash   
AttributeError: 'PolygonMasks' object has no attribute 'shape'
```
Please check the source code of the model and modify the source code to fix this issue. The model file should be two lines above the final line of the error message.    
Please modify the `prepare_targets` function in the model source code to fix this issue.
You may need to change from this:   
```bash
    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        labels = []
        for targets_per_image in targets:
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros(
                (gt_masks.shape[0], h_pad, w_pad),
                dtype=gt_masks.dtype,
                device=gt_masks.device,
            )
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            labels.append(targets_per_image.gt_classes)
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets, labels
```

to this:   
```bash
    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        labels = []
        for targets_per_image in targets:
            gt_masks = targets_per_image.gt_masks
            gt_masks = BitMasks.from_polygon_masks(gt_masks, h_pad, w_pad)
            gt_masks_tensor = gt_masks.tensor
            padded_masks = torch.zeros(
                (gt_masks_tensor.shape[0], h_pad, w_pad),
                dtype=gt_masks_tensor.dtype,
                device=gt_masks_tensor.device,
            )
            padded_masks[:, : gt_masks_tensor.shape[1], : gt_masks_tensor.shape[2]] = gt_masks_tensor
            labels.append(targets_per_image.gt_classes)
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets, labels
```

### tensor shape and device debug guidance
If you encounter the following error during the __training__ or __inference__ phase:
```bash
RuntimeError: weight tensor should be defined either for all or no classes
```
Please add the following line in your config file:
```bash
MODEL:
  EBSEG:
    ...
    NUM_CLASSES: 4 # your dataset's class number
```



### Special format requirements related to the `detectron2` package
The `detectron2` package does not really offer a good way to directly register the semantic segmentation dataset, the `register_coco_instances` function is mainly used to register the instance segmentation dataset if the dataset itself follows the `COCO` format.

We need to adjust the dataset first to register the semantic segmentation dataset if the model strictly requires to work only with semantic segmentation dataset. The semantic segmentation dataset we got from the `Roboflow` follows the `COCO` format, however, this means that we only have unannotated images with one json meta file inside the folder. Some models like `zsseg_baseline` does not compile with this way of annotation.

To solve this issue, we need to register the dataset with images and annotations in two subfolders, this is similar to the way for registering dataset in the `mmsegmentation` package.You can check the `register_ceiling_semantic_datasets` function in the `zsseg_baseline/mask_former/data/datasets/register_ceiling_semantic.py` file for more details.You can modify based on this file later for other use cases.
