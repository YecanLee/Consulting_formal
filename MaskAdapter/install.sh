# Please run the `install.sh` file from root folder of the cloned `consulting_formal` repo.
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118

# install detectron2 from the source
python -m pip install -e detectron2

pip install git+https://github.com/cocodataset/panopticapi.git
cd MaskAdapter
pip install -r requirements.txt
cd fcclip/modeling/pixel_decoder/ops
sh make.sh
cd ../../../..
