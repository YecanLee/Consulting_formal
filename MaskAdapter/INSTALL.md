## Installation
### Requirements
1. Clone this repository  
    ```
    git clone https://github.com/hustvl/MaskAdapter.git
    ```
2. Install the appropriate version of PyTorch for your CUDA version. Ensure that the PyTorch version is ≥ 1.9 and compatible with the version required by Detectron2. For CUDA 11.8, you can install the following:
    ```
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
    ```
3. Following [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html) to install Detectron2.
    ```
    git clone https://github.com/facebookresearch/detectron2.git
    python -m pip install -e detectron2
    ```
4. Install other requirements.
    ```
    pip install -r requirements.txt
    cd fcclip/modeling/pixel_decoder/ops
    sh make.sh
    ```

### Example conda environment configuration


```bash
conda create --name mask_adapter python=3.8
conda activate mask_adapter
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118

git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2

pip install git+https://github.com/cocodataset/panopticapi.git
git clone https://github.com/hustvl/MaskAdapter.git
cd MaskAdapter
pip install -r requirements.txt
cd fcclip/modeling/pixel_decoder/ops
sh make.sh
cd ../../../..
```
