"""Export a trained YOLO model to OpenVINO for faster Intel inference."""

import argparse
from ultralytics import YOLO

"""This script loads trained YOLO weights and exports them to OpenVINO format for faster inference on Intel hardware."""

def main() -> None:
    # Read the YOLO weights path and the image size from the command line.
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to trained YOLO .pt weights")
    parser.add_argument("--imgsz", default=640, type=int)
    args = parser.parse_args()

    # Load the trained YOLO model from the given weights file.
    model = YOLO(args.weights)

    # Export the model to OpenVINO format.
    # half=False keeps full precision, which is usually safer for compatibility.
    out = model.export(format="openvino", imgsz=args.imgsz, half=False)

    print(f"Exported OpenVINO model: {out}")


if __name__ == "__main__":
    main()