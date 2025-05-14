This repo includes our customized implementation of several Open Vocabulary Segmentation papers from some top Computer Vision Conferences.

We modified the model configurations and dataset loaders according to our own customized ceiling paiting datasets labelled by `Roboflow`.

## Starting Point
Please have some brief idea about what "Open Vocabulary Segmentation Task" is by reading some introduction blogs as well as some examples on Github.

## Environment Installation
### SAN paper env installation instructions
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

### In construction and Alert
Some of the models here require more than one train scripts, this section is mainly used to record which repo requires more than one train scripts.

The `MasQCLIP` requires two Progressive Distillation traning and one Mask-Q Tuning, they all used the `train_net.py` script, but with different config files.

The `MaskQCLIP` was trained with only two classes, `backdground` and `non-background`. This means that the model may have a different class_emb shape compared to our customized ceiling painting dataset. If we only have "mural" as the only class, this is fine. Otherwise, we need to modify the `class_emb` layer in the `mask_distill.py` file `MaskFormer` class to match our dataset.

### TODO(A quick to-do list)
- [ ] Check if the evaluators are working for different models. This may require us to change the way we register the `ceiling_painting` dataset. Or we may need to modify the metadata `evaluator_type` of the dataset.
- [ ] Add pre-commit hooks to check the code style. We may only check the files we changed during the whole project instead of every file.
- [ ] Add dockerfile for each model. We will only offer one training dockerfile and one deployment dockerfile.
