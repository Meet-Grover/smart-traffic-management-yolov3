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
    livery across the whole box — red/blue light bars (US/generic
    styling) OR a high-visibility yellow/green "Battenburg" pattern
    (common on UK/EU ambulances). This is still a heuristic, not a
    trained classifier — see the module docstring."""
    x1, y1, x2, y2 = box
    if y2 <= y1 or x2 <= x1:
        return False
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return False
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

    red_mask = cv2.inRange(hsv, (0, 120, 120), (10, 255, 255)) | cv2.inRange(
        hsv, (170, 120, 120), (180, 255, 255)
    )
    blue_mask = cv2.inRange(hsv, (100, 120, 120), (130, 255, 255))
    # Bright safety yellow-green, e.g. UK/EU "Battenburg" ambulance livery.
    hi_vis_mask = cv2.inRange(hsv, (25, 90, 150), (75, 255, 255))

    total = patch.shape[0] * patch.shape[1]
    signal_ratio = cv2.countNonZero(red_mask | blue_mask) / total
    hi_vis_ratio = cv2.countNonZero(hi_vis_mask) / total

    return signal_ratio > 0.02 or hi_vis_ratio > 0.12


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
