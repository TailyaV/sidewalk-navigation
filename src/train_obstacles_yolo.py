"""Train YOLO for sidewalk obstacle detection."""

import argparse
from pathlib import Path
from ultralytics import YOLO

"""This script trains a YOLO model to detect sidewalk obstacles using a labeled obstacle dataset."""

def main() -> None:
    # Read the training configuration from command-line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/datasets/obstacles/data.yaml", help="YOLO dataset yaml")
    parser.add_argument("--model", default="yolo11n.pt", help="Starting YOLO weights")
    parser.add_argument("--epochs", default=80, type=int)
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--batch", default=8, type=int)
    parser.add_argument("--device", default="cpu", help="Use cpu on this laptop; use 0 only with NVIDIA CUDA")
    args = parser.parse_args()

    # Load the starting YOLO model weights before fine-tuning on the obstacle dataset.
    model = YOLO(args.model)

    # Train the model and save the training run under the obstacles_yolo model folder.
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="models/obstacles_yolo/runs",
        name="train",
        patience=20,
        pretrained=True,
    )

    print(results)
    print("Best model is usually at: models/obstacles_yolo/runs/train/weights/best.pt")


if __name__ == "__main__":
    main()