"""Extract frames from training videos for annotation."""

import argparse
from pathlib import Path
import cv2
from tqdm import tqdm

"""This script extracts selected frames from a video and saves them as image files for dataset annotation."""

def extract_frames(video_path: Path, output_dir: Path, every_n_frames: int, max_frames: int | None) -> None:
    # Create the output directory if it does not already exist.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open the input video file with OpenCV.
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # Read basic video information and initialize frame counters.
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    saved = 0
    frame_idx = 0
    stem = video_path.stem

    # Iterate over the video and save one frame every N frames.
    with tqdm(total=total, desc=f"Extracting {video_path.name}") as pbar:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % every_n_frames == 0:
                out_path = output_dir / f"{stem}_frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(out_path), frame)
                saved += 1

                # Stop early if a maximum number of frames was requested.
                if max_frames is not None and saved >= max_frames:
                    break

            frame_idx += 1
            pbar.update(1)

    cap.release()
    print(f"Saved {saved} frames to {output_dir}")


def main() -> None:
    # Read the input video path and extraction settings from the command line.
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path, help="Path to input video")
    parser.add_argument("--out", default=Path("data/frames"), type=Path, help="Output frames directory")
    parser.add_argument("--every", default=15, type=int, help="Save one frame every N frames")
    parser.add_argument("--max", default=None, type=int, help="Optional maximum number of saved frames")
    args = parser.parse_args()

    extract_frames(args.video, args.out, args.every, args.max)


if __name__ == "__main__":
    main()