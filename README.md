# Sidewalk Navigation System for Blind Assistance

This project implements a real-time sidewalk navigation system designed to assist blind or visually impaired users.  
The system receives a live camera stream or a video file, segments the sidewalk, detects obstacles, and draws a stable navigation arrow that indicates a safe walking direction.

## Main Goal

The goal of the system is to analyze a street or sidewalk video in real time and provide visual guidance by:

- Segmenting the sidewalk area.
- Detecting obstacles such as people, bicycles, cars, poles, benches, boxes, and similar objects.
- Checking whether detected obstacles are actually located on the sidewalk.
- Choosing a safe walking direction.
- Drawing a stable arrow that guides the user to continue straight or bypass obstacles safely.

## Project Assets

The source code is available in this GitHub repository.
Large files such as trained models, raw videos, datasets, and example outputs are stored separately.
Download the project assets from:

[Download project assets from Google Drive](https://drive.google.com/file/d/1muWf4_AWb5v5-zwgXqI555DEEm6Dscn1/view?usp=drive_link)

After downloading, extract the ZIP file into the project root directory.

## Pipeline

The general processing pipeline is:

```text
Input video / live camera
        ↓
Frame extraction or real-time frame reading
        ↓
Sidewalk segmentation using SegFormer
        ↓
Obstacle detection using YOLO
        ↓
Check obstacle overlap with sidewalk mask
        ↓
Build safe walking corridor candidates
        ↓
Score possible directions
        ↓
Smooth the selected direction between frames
        ↓
Draw sidewalk mask, obstacles, and navigation arrow
        ↓
Display and/or save output video
```

For dataset preparation and training, the pipeline is:

```text
Training videos
        ↓
Extract frames
        ↓
Annotate frames in Roboflow
        ↓
Convert COCO polygon annotations to segmentation masks in Roboflow
        ↓
Train SegFormer sidewalk segmentation model
        ↓
Train YOLO obstacle detection model
        ↓
Run real-time sidewalk navigation app

```

## File Explanations

### `config.py`

Defines the main project paths, dataset locations, model paths, runtime settings, confidence thresholds, and class labels used by the sidewalk navigation system.

Main contents:

- Project folders: `data`, `models`, `outputs`.
- Dataset paths for sidewalk segmentation and obstacle detection.
- Base model names for SegFormer and YOLO.
- Runtime resolution and segmentation frequency.
- Confidence thresholds for sidewalk and obstacle processing.
- Obstacle class names.
- Label mappings for binary segmentation.

### `extract_frames.py`

Extracts selected frames from a video and saves them as image files for annotation.

This file is useful when creating a dataset from raw training videos.

Example use:

```bash
python src/extract_frames.py --video data/videos/train_video.mp4 --out data/frames --every 15 --max 300
```

Arguments:

- `--video`: path to the input video.
- `--out`: output folder for extracted frames.
- `--every`: save one frame every N frames.
- `--max`: optional maximum number of frames to save.

### `convert_coco_sidewalk_to_masks.py`

Converts a Roboflow COCO sidewalk dataset into an image-mask dataset for semantic segmentation training.

The script reads polygon annotations from `_annotations.coco.json` and creates binary mask images:

```text
0 = background
1 = sidewalk
```

### `train_sidewalk_segformer.py`

Fine-tunes a SegFormer model for binary semantic segmentation of sidewalks.

The model learns to classify each pixel as either:

```text
0 = background
1 = sidewalk

```

### `train_obstacles_yolo.py`

Trains a YOLO model for sidewalk obstacle detection.

The model detects objects that may block the sidewalk, such as people, bicycles, vehicles, poles, benches, boxes, and similar obstacles.

```

```
### `export_yolo_openvino.py`

Exports a trained YOLO model to OpenVINO format for faster inference on Intel hardware.

This step is optional. It is useful mainly when running inference on Intel CPUs or Intel hardware that benefits from OpenVINO optimization.

### `realtime_app.py`

Runs the full real-time sidewalk navigation system.

It performs:

1. Sidewalk segmentation using the trained SegFormer model.
2. Optional obstacle detection using YOLO.
3. Obstacle-sidewalk overlap filtering.
4. Safe walking direction selection.
5. Arrow smoothing between frames.
6. Displaying and/or saving the output video.

Example run on a video file:

```bash
python src/realtime_app.py \
  --source data/videos/test_sidewalk_3.mp4 \
  --sidewalk-model models/sidewalk_segformer/best \
  --yolo-weights models/obstacles_yolo/best.pt \
  --output outputs/videos/result.mp4 \
  --width 640 \
  --height 360 \
  --seg-every 3
```

Example run from a live camera:

```bash
 python src/realtime_app.py \
  --source data/raw_videos/test_sidewalk_2.mp4 \
  --sidewalk-model models/sidewalk_segformer/best \
  --yolo-weights models/obstacles_yolo/best.pt \
  --output outputs/videos/result_live.mp4 \
  --width 640 \
  --height 360 \
  --seg-every 3 \
  --live-sim    
```

`--live-sim` imitates a real live camera. If processing is slow, old frames are skipped so the displayed frame stays close to real time.

## Output

The system displays or saves a processed video that includes:

- Blue overlay: detected sidewalk area.
- Red transparent regions and red boxes: obstacles located on the sidewalk.
- Gray boxes: detected objects that are not considered sidewalk obstacles.
- Green arrow: continue straight.
- Yellow arrow: turn slightly or bypass an obstacle.
- Text label: the selected navigation instruction.

Example output path:

```text
outputs/videos/result.mp4
```

## Authors
- Sagi Rahat
- Taliya Levin 
