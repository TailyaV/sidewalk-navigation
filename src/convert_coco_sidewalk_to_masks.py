import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

"""This script converts a Roboflow COCO sidewalk dataset into an image-mask dataset for semantic segmentation training."""

def polygon_to_mask(segmentation, height, width):
    # Create an empty binary mask for one image.
    mask = np.zeros((height, width), dtype=np.uint8)

    # Convert each COCO polygon into pixel coordinates and fill it on the mask.
    for polygon in segmentation:
        points = np.array(polygon, dtype=np.float32).reshape(-1, 2)
        points = np.round(points).astype(np.int32)
        cv2.fillPoly(mask, [points], 1)

    return mask


def convert_split(coco_split_dir, output_split_dir):
    coco_split_dir = Path(coco_split_dir)
    output_split_dir = Path(output_split_dir)

    # Each Roboflow COCO split is expected to contain this annotation file.
    annotation_path = coco_split_dir / "_annotations.coco.json"
    if not annotation_path.exists():
        raise FileNotFoundError(f"Missing annotation file: {annotation_path}")

    # The converted dataset is saved as separate image and mask folders.
    images_out = output_split_dir / "images"
    masks_out = output_split_dir / "masks"
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)

    with open(annotation_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # Store image metadata by image ID for quick access while processing annotations.
    images = {img["id"]: img for img in coco["images"]}

    # Group all annotations by their related image ID.
    annotations_by_image = {}
    for ann in coco["annotations"]:
        image_id = ann["image_id"]
        annotations_by_image.setdefault(image_id, []).append(ann)

    for image_id, image_info in images.items():
        file_name = image_info["file_name"]
        width = image_info["width"]
        height = image_info["height"]

        src_image_path = coco_split_dir / file_name
        if not src_image_path.exists():
            print(f"Warning: missing image {src_image_path}")
            continue

        # Copy the original image to the converted dataset folder.
        dst_image_path = images_out / file_name
        shutil.copy2(src_image_path, dst_image_path)

        # Build one binary mask that combines all sidewalk polygons in this image.
        mask = np.zeros((height, width), dtype=np.uint8)

        for ann in annotations_by_image.get(image_id, []):
            segmentation = ann.get("segmentation", [])

            if isinstance(segmentation, list):
                ann_mask = polygon_to_mask(segmentation, height, width)
                mask = np.maximum(mask, ann_mask)

        # Save the mask with the same base name as the original image.
        mask_name = Path(file_name).with_suffix(".png").name
        mask_path = masks_out / mask_name

        Image.fromarray(mask).save(mask_path)

    print(f"Converted {len(images)} images from {coco_split_dir} to {output_split_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to Roboflow COCO dataset folder")
    parser.add_argument("--output", required=True, help="Path to output mask dataset folder")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    # Convert the training split.
    convert_split(input_dir / "train", output_dir / "train")

    # Support both common validation folder names: valid and val.
    if (input_dir / "valid").exists():
        convert_split(input_dir / "valid", output_dir / "val")
    elif (input_dir / "val").exists():
        convert_split(input_dir / "val", output_dir / "val")
    else:
        print("Warning: no valid/val folder found")


if __name__ == "__main__":
    main()