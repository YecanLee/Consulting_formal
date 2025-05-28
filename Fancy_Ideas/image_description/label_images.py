import os
import argparse
import csv
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
from tqdm import tqdm
from google import genai



def resize_if_needed(image, max_pixels=89478485):
    width, height = image.size
    current_pixels = width * height

    if current_pixels > max_pixels:
        scale_factor = (max_pixels / current_pixels) ** 0.5
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        image = image.resize((new_width, new_height), Image.LANCZOS)
    
    return image


def describe_images(image_dir: Path, output_path: Path):
    # Load API key
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file")

    # Configure Gemini client
    client = genai.Client(api_key=api_key)

    # Gather images
    image_paths = list(image_dir.glob("*.[pjw][pn]*g"))  # matches png, jpg, jpeg, webp

    if not image_paths:
        print(f"No images found in directory: {image_dir}")
        return

    system_instruction = (
    "You are an expert in Baroque ceiling painting. "
    "Describe the artistic style, symbolism, composition, and historical context of the painting in the image. "
    "Focus on the people present in the painting; if there isn't a painting present and just a ceiling or architectural structure, mention that. "
    "Keep the description short and precise. A maximum of 3 sentences."
)


    # Prepare TSV output
    with open(output_path, "w", newline="", encoding="utf-8") as tsvfile:
        writer = csv.writer(tsvfile, delimiter="\t")
        writer.writerow(["image_filename", "description"])

        for image_path in tqdm(image_paths, desc="Describing images"):
            try:
                image = Image.open(image_path)
                image = resize_if_needed(image)

                response = client.models.generate_content(
                    model="models/gemini-2.5-flash-preview-05-20",
                    contents=[image, system_instruction])

                description = response.text.strip()
                writer.writerow([image_path.name, description])

            except Exception as e:
                print(f"❌ Error with {image_path.name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Describe images using Gemini and save output to a TSV file.")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to the folder containing images.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output TSV file.")
    args = parser.parse_args()

    describe_images(Path(args.image_dir), Path(args.output_path))

if __name__ == "__main__":
    main()
