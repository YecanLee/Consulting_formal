import torch
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Fuse weights from two models")
    parser.add_argument("--model_first_phase_path", type=str, required=True, help="Path to the first phase model")
    parser.add_argument("--model_sem_seg_path", type=str, required=True, help="Path to the semantic segmentation model")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the fused model")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    model_first_phase_dict = torch.load(args.model_first_phase_path)
    model_dict_sem_seg = torch.load(args.model_sem_seg_path)

    for key in model_dict_sem_seg.keys():
        if key.startswith("sem_seg_head"):
            model_first_phase_dict["model"][key] = model_dict_sem_seg[key]

    torch.save(model_first_phase_dict, args.output_path)

if __name__ == "__main__":
    main()