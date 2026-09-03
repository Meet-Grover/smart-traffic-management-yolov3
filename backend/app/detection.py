"""
Vehicle + emergency-vehicle detection using YOLOv8 (Ultralytics).

Honest limitation: COCO (the dataset YOLOv8-n ships pretrained on) has no
"ambulance"/"emergency vehicle" class, and ambulances are frequently
classified as plain "car" rather than "truck"/"bus". Rather than fake
accuracy numbers, emergency vehicles are flagged with a documented color
heuristic across every detected vehicle box: a red/blue light-bar
signature (common on US-style vehicles), OR a high-visibility
yellow/green "Battenburg" pattern (common on UK/EU ambulances). This is
still clearly weaker than a purpose-trained model, and will miss
liveries it wasn't tuned against, or false-positive on unrelated
yellow/green vehicles. To do this properly, fine-tune YOLOv8 on a
labeled emergency-vehicle dataset (e.g. Kaggle's "Emergency Vehicle
Detection" set) and load those weights instead — see README for the
swap-in point.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from ultralytics import YOLO

from app.config import settings

_model: YOLO | None = None


def get_model() -> YOLO:
    global _model
    if _model is None:
        # yolov8s.pt trades a bit of speed/download size for meaningfully
        # better accuracy on small, overlapping objects (e.g. a dense,
        # bird's-eye traffic-jam photo) than the smaller yolov8n.pt.
        # ultralytics downloads it on first use and caches it locally.
        _model = YOLO("yolov8s.pt")
    return _model


@dataclass
class DetectionResult:
    vehicle_count: int
    boxes: list = field(default_factory=list)  # [(x1,y1,x2,y2,label), ...]
    emergency_detected: bool = False


def _looks_like_emergency(frame: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    """Heuristic: look for colors associated with emergency-vehicle
    livery — red/blue light bars near the roof (US/generic styling), OR
    a high-visibility yellow/green "Battenburg" checkerboard pattern
    (common on UK/EU ambulances). Still a heuristic, not a trained
    classifier — see the module docstring.

    An earlier version checked for "any meaningful patch of yellow
    anywhere in the box", which false-positived on ordinary yellow
    vehicles (taxis, construction equipment) that happened to appear in
    a scene. A real Battenburg pattern isn't just "some yellow" — it's
    yellow-green AND white/silver squares covering MOST of the vehicle
    in roughly equal measure. Requiring the box to be dominated by both
    colors together, not just one, cuts out most false positives from
    solid-colored yellow vehicles."""
    x1, y1, x2, y2 = box
    if y2 <= y1 or x2 <= x1:
        return False
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return False
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    total = patch.shape[0] * patch.shape[1]

    # Light bar: a small, strongly saturated red/blue patch near the
    # roof (top quarter of the box) — a solid-colored cargo truck won't
    # have this concentrated near the top.
    roof_strip = hsv[: max(1, hsv.shape[0] // 4), :]
    red_mask = cv2.inRange(roof_strip, (0, 120, 120), (10, 255, 255)) | cv2.inRange(
        roof_strip, (170, 120, 120), (180, 255, 255)
    )
    blue_mask = cv2.inRange(roof_strip, (100, 120, 120), (130, 255, 255))
    light_bar_ratio = cv2.countNonZero(red_mask | blue_mask) / max(1, roof_strip.size // 3)

    # Battenburg checkerboard: yellow-green AND white/silver must BOTH
    # be substantial AND together dominate the box — not just "yellow
    # present somewhere in frame".
    hi_vis_mask = cv2.inRange(hsv, (25, 90, 150), (75, 255, 255))
    white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 40, 255))
    hi_vis_ratio = cv2.countNonZero(hi_vis_mask) / total
    white_ratio = cv2.countNonZero(white_mask) / total
    is_checkerboard = (
        hi_vis_ratio > 0.15 and white_ratio > 0.15 and (hi_vis_ratio + white_ratio) > 0.5
    )

    return light_bar_ratio > 0.02 or is_checkerboard


def detect_frame(frame: np.ndarray) -> DetectionResult:
    """Run detection on a single BGR frame (as read by cv2)."""
    model = get_model()
    results = model.predict(
        frame, conf=settings.detection_confidence, verbose=False
    )[0]

    boxes = []
    emergency = False
    names = results.names

    for b in results.boxes:
        cls_id = int(b.cls[0])
        label = names.get(cls_id, str(cls_id))
        if label not in settings.vehicle_classes:
            continue
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        boxes.append((x1, y1, x2, y2, label))

        # Ambulances/emergency vans are frequently classified as "car" by
        # COCO-pretrained YOLO (COCO has no ambulance class), not just
        # "truck"/"bus" — so we check every vehicle box, not a subset.
        if _looks_like_emergency(frame, (x1, y1, x2, y2)):
            emergency = True

    return DetectionResult(vehicle_count=len(boxes), boxes=boxes, emergency_detected=emergency)


def detect_image_path(path: str) -> DetectionResult:
    frame = cv2.imread(path)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return detect_frame(frame)
