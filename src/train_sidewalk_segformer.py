"""Fine-tune SegFormer for binary sidewalk segmentation.

Expected dataset structure:
  data/datasets/sidewalk/train/images/*.jpg
  data/datasets/sidewalk/train/masks/*.png
  data/datasets/sidewalk/val/images/*.jpg
  data/datasets/sidewalk/val/masks/*.png

Each mask must be a single-channel image with values:
  0 = background
  1 = sidewalk
The mask filename must match the image stem, e.g. image_001.jpg -> image_001.png.
"""

"""This script fine-tunes a SegFormer model to perform binary semantic segmentation of sidewalks."""

import argparse
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoImageProcessor,
    SegformerForSemanticSegmentation,
    TrainingArguments,
    Trainer,
)


# Label mapping used by the binary sidewalk segmentation model.
ID2LABEL = {0: "background", 1: "sidewalk"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


class SidewalkSegmentationDataset(Dataset):
    def __init__(self, root: Path, split: str, processor: AutoImageProcessor):
        # Define the image and mask folders for the requested dataset split.
        self.images_dir = root / split / "images"
        self.masks_dir = root / split / "masks"
        self.processor = processor

        # Collect all supported image files in a stable order.
        self.image_paths = sorted([p for p in self.images_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        if not self.image_paths:
            raise RuntimeError(f"No images found in {self.images_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        image_path = self.image_paths[idx]
        mask_path = self.masks_dir / f"{image_path.stem}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {image_path.name}: {mask_path}")

        # Load the image and its matching segmentation mask.
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Convert every non-zero mask pixel to the sidewalk class.
        mask_np = np.array(mask, dtype=np.int64)
        mask_np = np.where(mask_np > 0, 1, 0).astype(np.int64)

        # Apply the SegFormer processor to prepare tensors for training.
        encoded = self.processor(images=image, segmentation_maps=mask_np, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in encoded.items()}


@dataclass
class SegMetrics:
    def __call__(self, eval_pred):
        logits, labels = eval_pred
        logits = torch.tensor(logits)
        labels = torch.tensor(labels)

        # Resize prediction logits to the mask size before comparing predictions with labels.
        logits = torch.nn.functional.interpolate(
            logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        preds = logits.argmax(dim=1).numpy()
        labels_np = labels.numpy()

        # Compute IoU for the sidewalk class and general pixel accuracy.
        sidewalk_pred = preds == 1
        sidewalk_true = labels_np == 1
        intersection = np.logical_and(sidewalk_pred, sidewalk_true).sum()
        union = np.logical_or(sidewalk_pred, sidewalk_true).sum()
        iou = float(intersection / union) if union > 0 else 1.0
        pixel_acc = float((preds == labels_np).mean())

        return {"sidewalk_iou": iou, "pixel_accuracy": pixel_acc}


def main() -> None:
    # Read training paths and hyperparameters from the command line.
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/datasets/sidewalk", type=Path)
    parser.add_argument("--base", default="nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
    parser.add_argument("--out", default="models/sidewalk_segformer/best", type=Path)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch", default=2, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)
    args = parser.parse_args()

    # Load the base image processor and adapt the SegFormer model to two classes.
    processor = AutoImageProcessor.from_pretrained(args.base, do_reduce_labels=False)
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.base,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    # Create training and validation datasets.
    train_ds = SidewalkSegmentationDataset(args.data, "train", processor)
    val_ds = SidewalkSegmentationDataset(args.data, "val", processor)

    # Define the Hugging Face training configuration.
    training_args = TrainingArguments(
        output_dir=str(args.out),
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="sidewalk_iou",
        greater_is_better=True,
        remove_unused_columns=False,
        logging_steps=20,
        save_total_limit=2,
        fp16=False,
        report_to="none",
    )

    # Train the model and evaluate it using the custom segmentation metrics.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=SegMetrics(),
    )

    trainer.train()

    # Save both the trained model and the processor needed for inference.
    trainer.save_model(str(args.out))
    processor.save_pretrained(str(args.out))

    print(f"Saved sidewalk model to {args.out}")


if __name__ == "__main__":
    main()