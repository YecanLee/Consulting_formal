import json

coco_meta_file = 'datasets/coco.json'

with open(coco_meta_file, "r") as f_in:
    content = json.load(f_in)
    print(content)
