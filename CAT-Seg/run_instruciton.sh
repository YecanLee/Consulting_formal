# To run experiments with ViT_L, run this file with `bash run_instructions.sh`

python train_net.py --config configs/ceiling_config.yaml --num-gpus 1 --dist-url "auto" --resume OUTPUT_DIR output/ 
