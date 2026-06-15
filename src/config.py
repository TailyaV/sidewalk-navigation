"""This file defines the main project paths, model locations, runtime settings, and class labels used by the sidewalk navigation system."""
from pathlib import Path

# Resolve the main project folders relative to this configuration file.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Dataset locations used for sidewalk segmentation and obstacle detection.
SIDEWALK_DATASET_DIR = DATA_DIR / "datasets" / "sidewalk"
OBSTACLE_DATASET_DIR = DATA_DIR / "datasets" / "obstacles"

# Base sidewalk segmentation model and the local directory where the trained model is saved.
# B0 is lightweight and suitable as a starting point for real-time inference.
SIDEWALK_BASE_MODEL = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
SIDEWALK_MODEL_DIR = MODELS_DIR / "sidewalk_segformer" / "best"

# Base YOLO model and the local path of the trained obstacle detection model.
# The nano model is used here because it is faster for real-time processing.
YOLO_BASE_MODEL = "yolo11n.pt"
YOLO_MODEL_PATH = MODELS_DIR / "obstacles_yolo" / "best.pt"

# Runtime inference settings.
# Lower resolution and skipping segmentation on some frames improve real-time performance.
RUNTIME_WIDTH = 640
RUNTIME_HEIGHT = 360
SEGMENT_EVERY_N_FRAMES = 3

# Visualization and confidence thresholds used during sidewalk and obstacle processing.
SIDEWALK_ALPHA = 0.35
SIDEWALK_CONFIDENCE_THRESHOLD = 0.55
OBSTACLE_CONFIDENCE_THRESHOLD = 0.35
OBSTACLE_SIDEWALK_OVERLAP_THRESHOLD = 0.30

# Only detections from these classes are treated as obstacles when they overlap the sidewalk mask.
OBSTACLE_CLASSES = {
    "trash_bin", "car", "truck", "bus", "motorcycle", "bicycle", "scooter",
    "traffic_cone", "pole", "bench", "box", "rock", "stone", "person"
}

# Label mappings for the sidewalk segmentation model.
ID2LABEL = {0: "background", 1: "sidewalk"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}