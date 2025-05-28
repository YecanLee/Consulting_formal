import os
import csv
import json
import argparse
from pathlib import Path
import webdataset as wds

def create_webdataset(tsv_path: Path, image_dir: Path, output_dir: Path, shard_pattern: str, maxcount: int = 100):
    """
    Converts a TSV + image folder into sharded WebDataset (.tar) format with JSON metadata.
    
    :param tsv_path: TSV file with 'image_filename' and 'description'
    :param image_dir: Directory containing images
    :param output_dir: Directory to save .tar shard files
    :param shard_pattern: Shard name pattern like 'baroque-%05d.tar'
    :param maxcount: Max samples per shard
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    full_output_pattern = str(output_dir / shard_pattern)

    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        with wds.ShardWriter(full_output_pattern, maxcount=maxcount) as sink:
            for row in reader:
                image_filename = row["image_filename"]
                description = row["description"]

                image_path = image_dir / image_filename
                if not image_path.exists():
                    print(f"⚠️ Skipping missing image: {image_path}")
                    continue

                sample_key = image_path.stem
                sample = {
                    "__key__": sample_key,
                    "jpg": image_path.read_bytes(),
                    "json": json.dumps({"description": description}, ensure_ascii=False),
                }

                sink.write(sample)

    print(f"✅ WebDataset written to: {full_output_pattern}")

def main():
    parser = argparse.ArgumentParser(description="Convert TSV + images into a WebDataset (sharded .tar format with JSON).")
    parser.add_argument("--tsv", required=True, help="Path to TSV file")
    parser.add_argument("--image_dir", required=True, help="Directory containing image files")
    parser.add_argument("--output_dir", required=True, help="Output directory for shards")
    parser.add_argument("--shard_pattern", default="baroque-%05d.tar", help="Shard filename pattern (default: baroque-%%05d.tar)")
    parser.add_argument("--maxcount", type=int, default=100, help="Max samples per shard")

    args = parser.parse_args()

    create_webdataset(
        Path(args.tsv),
        Path(args.image_dir),
        Path(args.output_dir),
        args.shard_pattern,
        args.maxcount
    )

if __name__ == "__main__":
    main()
