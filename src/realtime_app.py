import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

"""This script runs the real-time sidewalk navigation system: it segments sidewalks, detects obstacles, and draws a stable walking direction arrow."""
#running command -  python src/realtime_app.py --source data/raw_videos/test_sidewalk_3.mp4 --sidewalk-model models/sidewalk_segformer/best --yolo-weights models/obstacles_yolo/best_openvino_model --output outputs/videos/result_live_test_3.mp4 --width 640 --height 360 --seg-every 3 --live-sim

def load_sidewalk_model(model_dir):
    #Load the trained SegFormer sidewalk segmentation model.
    model_dir = Path(model_dir)

    if not model_dir.exists():
        raise FileNotFoundError(f"Sidewalk model folder not found: {model_dir}")

    # Load both the image processor and the trained segmentation model from the same directory.
    processor = SegformerImageProcessor.from_pretrained(str(model_dir))
    model = SegformerForSemanticSegmentation.from_pretrained(str(model_dir))

    # Use GPU if available, otherwise fall back to CPU.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    print(f"Loaded sidewalk model from: {model_dir}")
    print(f"Using device: {device}")

    return processor, model, device


def load_yolo_model(yolo_weights):
    """
    Load YOLO obstacle model if weights exist.
    If weights are missing, the app continues with sidewalk segmentation only.
    """
    if yolo_weights is None:
        print("YOLO weights were not provided. Running sidewalk segmentation only.")
        return None

    yolo_path = Path(yolo_weights)

    if not yolo_path.exists():
        print(f"YOLO weights not found: {yolo_path}")
        print("Running sidewalk segmentation only.")
        return None

    try:
        from ultralytics import YOLO

        model = YOLO(str(yolo_path))
        print(f"Loaded YOLO model from: {yolo_path}")
        return model

    except Exception as e:
        print("Could not load YOLO model.")
        print("Running sidewalk segmentation only.")
        print("Error:", e)
        return None


def predict_sidewalk_mask(frame_bgr, processor, model, device, target_size):
    """
    Predict a binary sidewalk mask for one frame.
    Output:
        0 = background
        1 = sidewalk
    """
    # Convert the OpenCV BGR frame to RGB because the SegFormer processor expects RGB images.
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    inputs = processor(images=pil_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        # Resize model logits back to the original runtime frame size.
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        predicted = upsampled_logits.argmax(dim=1)[0].detach().cpu().numpy()

    return predicted.astype(np.uint8)


def overlay_sidewalk_blue(frame_bgr, sidewalk_mask, alpha=0.35):
    """
    Paint sidewalk pixels in blue with transparency.
    """
    overlay = frame_bgr.copy()

    blue_color = np.array([255, 0, 0], dtype=np.uint8)  # BGR blue
    overlay[sidewalk_mask == 1] = blue_color

    blended = cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0)
    return blended


def bottom_box_overlap_with_sidewalk(box, sidewalk_mask):
    """
    Strictly check whether an object is really standing on the sidewalk.

    The object is considered on the sidewalk only if:
    1. The lower center area of the box is on sidewalk.
    2. The actual bottom contact strip has enough sidewalk pixels.
    3. The object is not just next to the sidewalk.
    """
    x1, y1, x2, y2 = box

    h, w = sidewalk_mask.shape[:2]

    # Clamp the box coordinates to the image boundaries.
    x1 = max(0, min(int(x1), w - 1))
    x2 = max(0, min(int(x2), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    y2 = max(0, min(int(y2), h - 1))

    if x2 <= x1 or y2 <= y1:
        return 0.0

    box_height = y2 - y1
    box_width = x2 - x1

    # Use only the bottom 10% of the object box.
    bottom_y1 = int(y2 - 0.10 * box_height)
    bottom_y2 = y2

    # Use only the inner 50% width of the object.
    inner_x1 = int(x1 + 0.25 * box_width)
    inner_x2 = int(x2 - 0.25 * box_width)

    if inner_x2 <= inner_x1:
        inner_x1, inner_x2 = x1, x2

    bottom_strip = sidewalk_mask[bottom_y1:bottom_y2, inner_x1:inner_x2]

    if bottom_strip.size == 0:
        return 0.0

    bottom_overlap = float(np.mean(bottom_strip == 1))

    # Strong center-bottom check.
    center_x = int((x1 + x2) / 2)
    center_y = int(y2)

    patch_radius_x = max(4, int(0.08 * box_width))
    patch_radius_y = max(4, int(0.04 * box_height))

    x_a = max(0, center_x - patch_radius_x)
    x_b = min(w, center_x + patch_radius_x + 1)
    y_a = max(0, center_y - patch_radius_y)
    y_b = min(h, center_y + patch_radius_y + 1)

    center_patch = sidewalk_mask[y_a:y_b, x_a:x_b]

    if center_patch.size == 0:
        return 0.0

    center_overlap = float(np.mean(center_patch == 1))

    # If the center-bottom area is not clearly sidewalk, reject completely.
    if center_overlap < 0.60:
        return 0.0

    return bottom_overlap


def tree_or_pole_near_sidewalk(box, sidewalk_mask):
    """
    Special check for trees/poles.

    Trees often stand inside a dirt square that is embedded in the sidewalk.
    The dirt square is not labeled as sidewalk, but the tree is still an obstacle
    in the sidewalk walking area.

    This function checks whether the lower part of the tree/pole is close to
    the sidewalk area, using a dilated sidewalk mask.
    """
    x1, y1, x2, y2 = box

    h, w = sidewalk_mask.shape[:2]

    x1 = max(0, min(int(x1), w - 1))
    x2 = max(0, min(int(x2), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    y2 = max(0, min(int(y2), h - 1))

    if x2 <= x1 or y2 <= y1:
        return False

    box_height = y2 - y1
    box_width = x2 - x1

    # Expand the sidewalk mask a little.
    kernel_size = max(9, int(0.04 * w))
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    dilated_sidewalk = cv2.dilate(
        (sidewalk_mask == 1).astype(np.uint8),
        kernel,
        iterations=1,
    )

    # Check the lower part of the tree/pole, not the entire box.
    lower_y1 = int(y1 + 0.60 * box_height)
    lower_y2 = y2

    # Use a central vertical strip of the box.
    inner_x1 = int(x1 + 0.25 * box_width)
    inner_x2 = int(x2 - 0.25 * box_width)

    if inner_x2 <= inner_x1:
        inner_x1, inner_x2 = x1, x2

    lower_part = dilated_sidewalk[lower_y1:lower_y2, inner_x1:inner_x2]

    if lower_part.size == 0:
        return False

    near_sidewalk_ratio = float(np.mean(lower_part == 1))

    return near_sidewalk_ratio >= 0.25


def draw_yolo_obstacles(
    frame_bgr,
    yolo_model,
    sidewalk_mask,
    confidence_threshold=0.40,
    sidewalk_overlap_threshold=0.70,
):
    """
    Run YOLO and draw detected obstacles.

    Obstacles that are on the sidewalk:
    - are painted in transparent red
    - get a red bounding box
    - are returned for navigation arrow logic

    Other detected objects:
    - get a gray bounding box
    """
    obstacles_on_sidewalk = []

    if yolo_model is None:
        return frame_bgr, obstacles_on_sidewalk

    results = yolo_model.predict(frame_bgr, conf=confidence_threshold, verbose=False)

    if not results:
        return frame_bgr, obstacles_on_sidewalk

    result = results[0]

    if result.boxes is None:
        return frame_bgr, obstacles_on_sidewalk

    names = result.names

    for box_data in result.boxes:
        xyxy = box_data.xyxy[0].detach().cpu().numpy()
        conf = float(box_data.conf[0].detach().cpu().numpy())
        cls_id = int(box_data.cls[0].detach().cpu().numpy())

        x1, y1, x2, y2 = xyxy
        class_name = names.get(cls_id, str(cls_id))

        # Check whether the lower part of the object actually intersects the sidewalk.
        overlap = bottom_box_overlap_with_sidewalk((x1, y1, x2, y2), sidewalk_mask)
        is_on_sidewalk = overlap >= sidewalk_overlap_threshold

        # Special rule:
        # Trees and poles may stand on dirt/tree pits inside the sidewalk.
        if class_name in ["Tree", "Column", "tree", "pole"]:
            is_on_sidewalk = is_on_sidewalk or tree_or_pole_near_sidewalk(
                box=(x1, y1, x2, y2),
                sidewalk_mask=sidewalk_mask,
            )

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        if is_on_sidewalk:
            color = (0, 0, 255)  # red in BGR

            # Save this obstacle for the path-planning logic.
            obstacles_on_sidewalk.append(
                {
                    "box": (x1, y1, x2, y2),
                    "class_name": class_name,
                    "confidence": conf,
                    "overlap": overlap,
                }
            )

            # Paint only this obstacle area in transparent red.
            obstacle_overlay = frame_bgr.copy()
            obstacle_overlay[y1:y2, x1:x2] = color
            frame_bgr = cv2.addWeighted(obstacle_overlay, 0.30, frame_bgr, 0.70, 0)

            label = f"OBSTACLE: {class_name} {conf:.2f}"

        else:
            color = (160, 160, 160)  # gray
            label = f"{class_name} {conf:.2f}"

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

        label_y = max(20, y1 - 8)
        cv2.putText(
            frame_bgr,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    return frame_bgr, obstacles_on_sidewalk


class ArrowController:
    """
    Keeps the navigation arrow stable between frames.

    The controller stores the previous walking direction. The path planner uses
    this direction to prefer the closest safe bypass instead of jumping toward a
    far sidewalk-like area such as a parking area.
    """

    def __init__(self, alpha=0.18, switch_frames=4):
        self.smoothed_angle = 0.0
        self.current_label = "GO STRAIGHT"
        self.current_target_angle = 0.0
        self.candidate_label = "GO STRAIGHT"
        self.candidate_angle = 0.0
        self.candidate_count = 0
        self.alpha = alpha
        self.switch_frames = switch_frames

    def update(self, target_angle, target_label):
        # Require the new direction to remain stable for several frames before switching labels.
        if target_label == self.current_label:
            self.current_target_angle = target_angle
            self.candidate_label = target_label
            self.candidate_angle = target_angle
            self.candidate_count = 0
        else:
            if target_label == self.candidate_label:
                self.candidate_count += 1
                self.candidate_angle = target_angle
            else:
                self.candidate_label = target_label
                self.candidate_angle = target_angle
                self.candidate_count = 1

            if self.candidate_count >= self.switch_frames:
                self.current_label = target_label
                self.current_target_angle = self.candidate_angle
                self.candidate_count = 0

        # Smooth the arrow angle to reduce visual jumping between frames.
        self.smoothed_angle = (
            (1.0 - self.alpha) * self.smoothed_angle
            + self.alpha * self.current_target_angle
        )

        return self.smoothed_angle, self.current_label


def make_obstacle_mask(obstacles_on_sidewalk, height, width, expand_px=10):
    """
    Create a navigation obstacle mask.

    For navigation, a full YOLO box can be too aggressive because it includes
    image area above the actual walking contact region. The mask therefore uses
    the lower part of each box, expands it as a safety margin, and keeps it as
    forbidden walking space.
    """
    obstacle_mask = np.zeros((height, width), dtype=np.uint8)

    for obstacle in obstacles_on_sidewalk:
        x1, y1, x2, y2 = obstacle["box"]

        box_h = max(1, int(y2) - int(y1))

        # Use mostly the lower part of the object for path collision.
        # This avoids a large car/person box blocking safe corridors only because
        # the upper part of the rectangle overlaps the corridor.
        nav_y1 = int(y1 + 0.20 * box_h)

        x1 = max(0, int(x1) - expand_px)
        y1 = max(0, nav_y1 - expand_px)
        x2 = min(width - 1, int(x2) + expand_px)
        y2 = min(height - 1, int(y2) + expand_px)

        if x2 > x1 and y2 > y1:
            obstacle_mask[y1:y2, x1:x2] = 1

    return obstacle_mask


def estimate_forward_obstacle_side(obstacles_on_sidewalk, start_point, height, width):
    """
    Estimate whether the most relevant obstacle is mostly left or right of the
    current walking line.

    The previous version used only the box center with a wide dead zone. Small
    obstacles near the center, such as bicycles, could therefore be treated as
    "no clear side" and the planner was allowed to steer toward them.

    Return:
        -1 = obstacle is mostly on the left
         0 = no clear side
         1 = obstacle is mostly on the right
    """
    if not obstacles_on_sidewalk:
        return 0

    sx, sy = start_point
    best = None

    for obstacle in obstacles_on_sidewalk:
        x1, y1, x2, y2 = obstacle["box"]
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)

        # Ignore objects behind the user and objects very high in the image.
        if cy > sy or y2 < height * 0.25:
            continue

        # Prefer obstacles that are close to the current walking corridor.
        box_width = max(1.0, float(x2 - x1))
        center_distance = abs(cx - sx) / max(1.0, width)
        vertical_distance = abs(sy - cy) / max(1.0, height)

        # Obstacles far away to the side should not decide the bypass direction.
        expanded_x1 = x1 - 0.35 * box_width
        expanded_x2 = x2 + 0.35 * box_width
        near_current_line = expanded_x1 <= sx <= expanded_x2 or center_distance < 0.22
        if not near_current_line:
            continue

        relevance = vertical_distance + 0.45 * center_distance
        if best is None or relevance < best[0]:
            best = (relevance, x1, x2, cx)

    if best is None:
        return 0

    _, x1, x2, cx = best

    # If the box crosses the current line, choose the side with more box area.
    if x1 <= sx <= x2:
        left_part = sx - x1
        right_part = x2 - sx
        if right_part > left_part * 1.15:
            return 1
        if left_part > right_part * 1.15:
            return -1
        return 0

    # Use a small dead zone so near-center bicycles/poles still get a side.
    side_dead_zone = 0.02 * width
    if cx < sx - side_dead_zone:
        return -1
    if cx > sx + side_dead_zone:
        return 1
    return 0


def make_main_sidewalk_mask(sidewalk_mask, start_point):
    """
    Keep the sidewalk component connected to the user's current walking area.
    This reduces attraction to side areas that may be labeled as sidewalk.
    """
    h, w = sidewalk_mask.shape[:2]
    sx, sy = start_point

    binary = (sidewalk_mask == 1).astype(np.uint8)

    # Close small holes so the current sidewalk area is treated as one connected region.
    kernel_size = max(5, int(w * 0.015))
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if num_labels <= 1:
        return binary

    sx = max(0, min(int(sx), w - 1))
    sy = max(0, min(int(sy), h - 1))
    start_label = labels[sy, sx]

    if start_label != 0:
        return (labels == start_label).astype(np.uint8)

    # If the exact start pixel is not sidewalk, search near the bottom-center area.
    search_radius_x = max(25, int(w * 0.12))
    search_radius_y = max(20, int(h * 0.12))
    x1 = max(0, sx - search_radius_x)
    x2 = min(w, sx + search_radius_x + 1)
    y1 = max(0, sy - search_radius_y)
    y2 = min(h, sy + search_radius_y + 1)

    nearby_labels = labels[y1:y2, x1:x2]
    candidates, counts = np.unique(nearby_labels[nearby_labels != 0], return_counts=True)

    if len(candidates) > 0:
        chosen_label = int(candidates[np.argmax(counts)])
        return (labels == chosen_label).astype(np.uint8)

    # Fallback: use the largest sidewalk component.
    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(component_areas) + 1)
    return (labels == largest_label).astype(np.uint8)


def make_path_corridor_mask(height, width, start_point, end_point, corridor_width):
    """
    Create a thick line mask representing the walking corridor.
    If obstacles intersect this corridor, the user may collide with them.
    """
    corridor = np.zeros((height, width), dtype=np.uint8)

    cv2.line(
        corridor,
        start_point,
        end_point,
        color=1,
        thickness=corridor_width,
    )

    return corridor


def classify_arrow_label(angle_deg):
    """
    Convert a continuous angle into a readable navigation label.
    """
    if abs(angle_deg) < 8:
        return "GO STRAIGHT"
    if angle_deg < -25:
        return "GO LEFT"
    if angle_deg < 0:
        return "GO SLIGHT LEFT"
    if angle_deg > 25:
        return "GO RIGHT"
    return "GO SLIGHT RIGHT"


def candidate_endpoint_from_angle(start_point, forward_y, angle_deg, width):
    """
    Convert an arrow angle to an endpoint in the image.
    0 degrees means straight up, negative means left, positive means right.
    """
    start_x, start_y = start_point
    forward_distance = max(1, start_y - forward_y)
    dx = int(np.tan(np.deg2rad(angle_deg)) * forward_distance)
    end_x = max(0, min(width - 1, start_x + dx))
    return end_x, forward_y


def score_candidate_path(
    sidewalk_mask,
    main_sidewalk_mask,
    obstacle_mask,
    start_point,
    end_point,
    corridor_width,
    candidate_angle,
    previous_angle,
    obstacle_side=0,
):
    """
    Score a possible walking path.

    Safety is the first priority, but the planner does not reject a path for a
    tiny overlap with an inflated YOLO box. Instead, small overlaps get a strong
    penalty and large overlaps are rejected. This is important because YOLO boxes
    are rectangular and can cover pixels that are not actually part of the
    walking collision area.
    """
    h, w = sidewalk_mask.shape[:2]

    corridor = make_path_corridor_mask(
        height=h,
        width=w,
        start_point=start_point,
        end_point=end_point,
        corridor_width=corridor_width,
    )

    corridor_pixels = corridor == 1

    if np.sum(corridor_pixels) == 0:
        return -999.0, 0.0, 1.0, 0.0

    # Measure how much of the proposed corridor is sidewalk, main sidewalk, and obstacle.
    sidewalk_ratio = float(np.mean(sidewalk_mask[corridor_pixels] == 1))
    main_sidewalk_ratio = float(np.mean(main_sidewalk_mask[corridor_pixels] == 1))
    obstacle_ratio = float(np.mean(obstacle_mask[corridor_pixels] == 1))

    # A large overlap with the expanded obstacle mask is unsafe.
    if obstacle_ratio > 0.12:
        return -999.0, sidewalk_ratio, obstacle_ratio, main_sidewalk_ratio

    # A corridor with too little sidewalk is not a valid walking option.
    if sidewalk_ratio < 0.38:
        return -500.0, sidewalk_ratio, obstacle_ratio, main_sidewalk_ratio

    deviation_from_previous = abs(candidate_angle - previous_angle) / 45.0
    deviation_from_straight = abs(candidate_angle) / 45.0

    # Walking straight is the preferred human behavior. A turn should happen
    # only when it is needed to avoid an obstacle or to stay on the main
    # sidewalk. After a bypass, this term naturally pulls the arrow back toward
    # straight walking instead of continuing sideways.
    straight_recovery_bonus = max(0.0, 1.0 - deviation_from_straight)

    # If the nearest relevant obstacle is on one side, do not steer toward it.
    # This is intentionally stronger than a soft penalty, because in assistive
    # navigation a small safe-looking region near the obstacle is still risky.
    toward_obstacle_penalty = 0.0
    away_from_obstacle_bonus = 0.0
    if obstacle_side < 0:
        if candidate_angle < -4:
            return -850.0, sidewalk_ratio, obstacle_ratio, main_sidewalk_ratio
        elif candidate_angle > 4:
            away_from_obstacle_bonus = 0.80
    elif obstacle_side > 0:
        if candidate_angle > 4:
            return -850.0, sidewalk_ratio, obstacle_ratio, main_sidewalk_ratio
        elif candidate_angle < -4:
            away_from_obstacle_bonus = 0.80

    # When there is an obstacle ahead, staying close to the previous direction
    # is less important than moving away from the obstacle. In all cases, keep
    # a strong preference for returning to straight walking when it is safe.
    previous_weight = 1.1 if obstacle_side != 0 else 2.6
    straight_weight = 3.8 if obstacle_side == 0 else 2.7

    score = (
        3.0 * sidewalk_ratio
        + 4.2 * main_sidewalk_ratio
        - 16.0 * obstacle_ratio
        - previous_weight * deviation_from_previous
        - straight_weight * deviation_from_straight
        - toward_obstacle_penalty
        + away_from_obstacle_bonus
        + 1.2 * straight_recovery_bonus
    )

    return score, sidewalk_ratio, obstacle_ratio, main_sidewalk_ratio


def choose_navigation_target(sidewalk_mask, obstacles_on_sidewalk, previous_angle=0.0):
    """
    Choose where the arrow head should point.

    The planner checks several possible walking corridors. It prefers walking
    straight whenever that is safe. A side direction is used only as a temporary
    bypass around an obstacle, and after the bypass the planner is pulled back
    toward straight walking.
    """
    h, w = sidewalk_mask.shape[:2]

    # The walking path starts near the bottom-center of the frame and points forward.
    start_point = (w // 2, int(h * 0.84))
    forward_y = int(h * 0.45)

    # A slightly narrower corridor is less likely to mark every option as
    # blocked when YOLO boxes are coarse. The obstacle mask still has a safety
    # expansion, so the path remains conservative.
    corridor_width = max(30, int(w * 0.105))

    obstacle_mask = make_obstacle_mask(
        obstacles_on_sidewalk=obstacles_on_sidewalk,
        height=h,
        width=w,
        expand_px=max(14, int(w * 0.035)),
    )

    main_sidewalk_mask = make_main_sidewalk_mask(
        sidewalk_mask=sidewalk_mask,
        start_point=start_point,
    )

    obstacle_side = estimate_forward_obstacle_side(
        obstacles_on_sidewalk=obstacles_on_sidewalk,
        start_point=start_point,
        height=h,
        width=w,
    )

    candidate_angles = [-45, -35, -25, -15, -8, 0, 8, 15, 25, 35, 45]
    candidates = []

    # Evaluate several possible walking directions and keep their safety scores.
    for angle in candidate_angles:
        end_point = candidate_endpoint_from_angle(
            start_point=start_point,
            forward_y=forward_y,
            angle_deg=angle,
            width=w,
        )

        score, sidewalk_ratio, obstacle_ratio, main_sidewalk_ratio = score_candidate_path(
            sidewalk_mask=sidewalk_mask,
            main_sidewalk_mask=main_sidewalk_mask,
            obstacle_mask=obstacle_mask,
            start_point=start_point,
            end_point=end_point,
            corridor_width=corridor_width,
            candidate_angle=angle,
            previous_angle=previous_angle,
            obstacle_side=obstacle_side,
        )

        candidates.append(
            {
                "angle": angle,
                "score": score,
                "sidewalk_ratio": sidewalk_ratio,
                "obstacle_ratio": obstacle_ratio,
                "main_sidewalk_ratio": main_sidewalk_ratio,
            }
        )

    safe_candidates = [c for c in candidates if c["score"] > -100]

    # Strong human-like rule: if walking straight is safe enough, prefer it.
    # This prevents the arrow from keeping a side direction after a bypass and
    # makes the system return to normal straight walking as soon as possible.
    straight_candidate = next((c for c in candidates if c["angle"] == 0), None)
    if straight_candidate is not None:
        straight_is_safe = (
            straight_candidate["score"] > -100
            and straight_candidate["obstacle_ratio"] < 0.025
            and straight_candidate["sidewalk_ratio"] > 0.44
            and straight_candidate["main_sidewalk_ratio"] > 0.36
        )
        if straight_is_safe:
            return 0.0, "GO STRAIGHT"

    if safe_candidates:
        # Prefer the best score, but break close ties toward a straighter path.
        best_score = max(c["score"] for c in safe_candidates)
        close_to_best = [c for c in safe_candidates if c["score"] >= best_score - 0.35]
        best = min(close_to_best, key=lambda c: (abs(c["angle"]), c["obstacle_ratio"]))
        return float(best["angle"]), classify_arrow_label(best["angle"])

    # Emergency fallback: do not prefer "more sidewalk" toward an obstacle.
    # First minimize obstacle intersection, then prefer the main sidewalk, then
    # prefer staying close to the previous direction.
    best = min(
        candidates,
        key=lambda c: (
            c["obstacle_ratio"],
            -c["main_sidewalk_ratio"],
            abs(c["angle"] - previous_angle),
            abs(c["angle"]),
        ),
    )

    # If the best emergency option still points toward the nearest obstacle,
    # choose the closest option in the opposite direction with comparable
    # obstacle overlap.
    if obstacle_side != 0 and best["angle"] * obstacle_side > 0:
        opposite = [
            c for c in candidates
            if c["angle"] * obstacle_side < 0
            and c["obstacle_ratio"] <= best["obstacle_ratio"] + 0.04
            and c["sidewalk_ratio"] >= 0.30
        ]
        if opposite:
            best = min(opposite, key=lambda c: (abs(c["angle"]), c["obstacle_ratio"]))

    return float(best["angle"]), classify_arrow_label(best["angle"])


def draw_arrow_head_by_angle(frame_bgr, center, angle_deg, color):
    """
    Draw only a rotated arrow head, without a long arrow line.

    angle_deg:
    - 0 means straight up
    - negative means left
    - positive means right
    """
    cx, cy = center
    size = 42

    # Base arrow head pointing up.
    points = np.array(
        [
            [0, -size],
            [-int(size * 0.55), int(size * 0.45)],
            [0, int(size * 0.20)],
            [int(size * 0.55), int(size * 0.45)],
        ],
        dtype=np.float32,
    )

    angle_rad = np.deg2rad(angle_deg)

    # Rotate the arrow head according to the selected navigation angle.
    rotation = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ],
        dtype=np.float32,
    )

    rotated = points @ rotation.T
    rotated[:, 0] += cx
    rotated[:, 1] += cy
    rotated = rotated.astype(np.int32)

    cv2.fillConvexPoly(frame_bgr, rotated, color)
    cv2.polylines(frame_bgr, [rotated], isClosed=True, color=(0, 0, 0), thickness=2)

    return frame_bgr


def draw_navigation_arrow(frame_bgr, sidewalk_mask, obstacles_on_sidewalk, arrow_controller):
    """
    Draw a stable navigation arrow head.

    It acts like guidance for walking:
    - straight if the forward corridor is clear.
    - left/right only if the forward corridor intersects an obstacle.
    - smoothed between frames to prevent flickering.
    """
    h, w = frame_bgr.shape[:2]

    target_angle, target_label = choose_navigation_target(
        sidewalk_mask=sidewalk_mask,
        obstacles_on_sidewalk=obstacles_on_sidewalk,
        previous_angle=arrow_controller.smoothed_angle,
    )

    smooth_angle, stable_label = arrow_controller.update(
        target_angle=target_angle,
        target_label=target_label,
    )

    arrow_center = (w // 2, int(h * 0.80))

    if stable_label == "GO STRAIGHT":
        color = (0, 255, 0)  # green
    else:
        color = (0, 255, 255)  # yellow

    frame_bgr = draw_arrow_head_by_angle(
        frame_bgr=frame_bgr,
        center=arrow_center,
        angle_deg=smooth_angle,
        color=color,
    )

    cv2.putText(
        frame_bgr,
        stable_label,
        (30, h - 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        3,
        cv2.LINE_AA,
    )

    return frame_bgr


def open_video_source(source, width, height):
    """
    Open either a live camera or a video file.
    """
    source_is_camera = str(source).lower() == "camera"

    if source_is_camera:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    else:
        cap = cv2.VideoCapture(str(source))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    return cap, source_is_camera


def create_video_writer(output, fps, width, height):
    """
    Create output video writer if output path was provided.
    """
    if output is None:
        return None

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_path}")

    print(f"Saving output video to: {output_path}")
    return writer


def skip_to_live_frame_if_needed(cap, source_fps, live_start_time, total_frames):
    """
    For video-file live simulation:
    skip old frames and jump to the frame that should be visible now.
    This imitates a real live camera, where the camera does not wait for slow processing.
    """
    elapsed_from_start = time.time() - live_start_time
    target_frame_idx = int(elapsed_from_start * source_fps)

    current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    if total_frames > 0:
        target_frame_idx = min(target_frame_idx, total_frames - 1)

    if target_frame_idx > current_frame_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)


def run(
    source,
    sidewalk_model_dir,
    yolo_weights,
    output,
    width,
    height,
    seg_every,
    no_show,
    realtime_playback,
    live_sim,
):
    """
    Run sidewalk segmentation and optional obstacle detection on camera/video.

    Modes:
    - Normal mode:
      processes frames in order.

    - Realtime playback:
      processes frames in order, but waits according to the original video FPS.

    - Live simulation:
      imitates a real camera. If processing is slow, old frames are skipped.
    """
    # Load all models before starting the video loop.
    processor, sidewalk_model, device = load_sidewalk_model(sidewalk_model_dir)
    yolo_model = load_yolo_model(yolo_weights)

    cap, source_is_camera = open_video_source(source, width, height)

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if not source_fps or source_fps <= 1:
        source_fps = 25.0

    frame_period = 1.0 / source_fps

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_is_camera:
        total_frames = 0

    writer = create_video_writer(output, source_fps, width, height)

    last_sidewalk_mask = None
    processed_frames = 0
    live_start_time = time.time()
    arrow_controller = ArrowController(alpha=0.22, switch_frames=3)

    # Lower alpha = smoother/slower arrow.
    # Higher switch_frames = direction changes only after more stable evidence.
    arrow_controller = ArrowController(alpha=0.18, switch_frames=4)

    print("Press Q to stop.")
    print(f"Source FPS: {source_fps:.2f}")
    print(f"Segmentation every {seg_every} frame(s)")

    if live_sim and not source_is_camera:
        print("Mode: LIVE SIMULATION - old frames will be skipped if processing is slow.")
    elif realtime_playback and not source_is_camera:
        print("Mode: REALTIME PLAYBACK - all frames are processed in order.")
    else:
        print("Mode: NORMAL - processing as fast as possible.")

    while True:
        loop_start = time.time()

        if live_sim and not source_is_camera:
            skip_to_live_frame_if_needed(
                cap=cap,
                source_fps=source_fps,
                live_start_time=live_start_time,
                total_frames=total_frames,
            )

        ok, frame = cap.read()
        if not ok:
            break

        current_video_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

        frame = cv2.resize(frame, (width, height))

        # Segmentation is not necessarily run on every frame to improve runtime speed.
        should_run_segmentation = (
            last_sidewalk_mask is None
            or processed_frames % seg_every == 0
        )

        if should_run_segmentation:
            last_sidewalk_mask = predict_sidewalk_mask(
                frame_bgr=frame,
                processor=processor,
                model=sidewalk_model,
                device=device,
                target_size=(height, width),
            )

        # Build the displayed frame: sidewalk overlay, obstacle overlay, and navigation arrow.
        display_frame = overlay_sidewalk_blue(frame, last_sidewalk_mask)

        display_frame, obstacles_on_sidewalk = draw_yolo_obstacles(
            frame_bgr=display_frame,
            yolo_model=yolo_model,
            sidewalk_mask=last_sidewalk_mask,
        )

        display_frame = draw_navigation_arrow(
            frame_bgr=display_frame,
            sidewalk_mask=last_sidewalk_mask,
            obstacles_on_sidewalk=obstacles_on_sidewalk,
            arrow_controller=arrow_controller,
        )

        mode_text = "LIVE-SIM" if live_sim else "NORMAL"
        info_text = (
            f"{mode_text} | video frame: {current_video_frame} | "
            f"processed: {processed_frames} | seg every: {seg_every}"
        )

        cv2.putText(
            display_frame,
            info_text,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if writer is not None:
            writer.write(display_frame)

        if not no_show:
            cv2.imshow("Sidewalk Real-Time App", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        # Realtime playback keeps the original video pace without skipping frames.
        if realtime_playback and not source_is_camera and not live_sim:
            elapsed = time.time() - loop_start
            sleep_time = frame_period - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)

        # Live simulation also keeps real time, but may skip frames before reading them.
        if live_sim and not source_is_camera:
            elapsed = time.time() - loop_start
            sleep_time = frame_period - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)

        processed_frames += 1

    cap.release()

    if writer is not None:
        writer.release()

    cv2.destroyAllWindows()
    print("Done.")


def parse_args():
    # Read all runtime settings from the command line.
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        required=True,
        help="Use 'camera' for live webcam, or provide a path to a video file.",
    )

    parser.add_argument(
        "--sidewalk-model",
        default="models/sidewalk_segformer/best",
        help="Path to trained SegFormer sidewalk model folder.",
    )

    parser.add_argument(
        "--yolo-weights",
        default=None,
        help="Optional path to YOLO obstacle model weights. Example: models/obstacles_yolo/best.pt",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save output video. Example: outputs/videos/result.mp4",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Frame width used for inference.",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=360,
        help="Frame height used for inference.",
    )

    parser.add_argument(
        "--seg-every",
        type=int,
        default=3,
        help="Run sidewalk segmentation every N processed frames. Higher value is faster.",
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not show OpenCV window. Useful when only saving output video.",
    )

    parser.add_argument(
        "--realtime-playback",
        action="store_true",
        help="For video files, play according to original FPS but process all frames in order.",
    )

    parser.add_argument(
        "--live-sim",
        action="store_true",
        help="For video files, imitate a live camera by skipping old frames if processing is slow.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Live simulation has priority because it better represents a real camera stream.
    if args.live_sim and args.realtime_playback:
        print("Both --live-sim and --realtime-playback were provided.")
        print("Using --live-sim. It has priority over --realtime-playback.")

    run(
        source=args.source,
        sidewalk_model_dir=args.sidewalk_model,
        yolo_weights=args.yolo_weights,
        output=args.output,
        width=args.width,
        height=args.height,
        seg_every=args.seg_every,
        no_show=args.no_show,
        realtime_playback=args.realtime_playback,
        live_sim=args.live_sim,
    )


if __name__ == "__main__":
    main()