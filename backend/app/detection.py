"""
Vehicle + emergency-vehicle detection using YOLOv8 (Ultralytics).

Honest limitation: COCO (the dataset YOLOv8-n ships pretrained on) has no
"ambulance"/"emergency vehicle" class. Rather than fake accuracy numbers,
emergency vehicles are flagged with a documented heuristic (large box
classified as "truck" + a red/blue light-bar color signature near the
roof of the bounding box). This is clearly weaker than a purpose-trained
model. To do this properly, fine-tune YOLOv8 on a labeled emergency-vehicle
dataset (e.g. Kaggle's "Emergency Vehicle Detection" set) and load those
weights instead — see README for the swap-in point.
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
        # yolov8n.pt is the small/fast variant; ultralytics downloads it
        # on first use and caches it locally.
        _model = YOLO("yolov8n.pt")
    return _model


@dataclass
class DetectionResult:
    vehicle_count: int
    boxes: list = field(default_factory=list)  # [(x1,y1,x2,y2,label), ...]
    emergency_detected: bool = False


def _looks_like_emergency(frame: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    """Heuristic: look for a saturated red/blue patch near the top of a
    large "truck"-classified box, roughly where a light bar would sit."""
    x1, y1, x2, y2 = box
    h = y2 - y1
    if h <= 0:
        return False
    strip = frame[y1 : y1 + max(1, h // 4), x1:x2]
    if strip.size == 0:
        return False
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv, (0, 120, 120), (10, 255, 255)) | cv2.inRange(
        hsv, (170, 120, 120), (180, 255, 255)
    )
    blue_mask = cv2.inRange(hsv, (100, 120, 120), (130, 255, 255))
    signal_ratio = (cv2.countNonZero(red_mask) + cv2.countNonZero(blue_mask)) / strip.size
    return signal_ratio > 0.02


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

        if label in ("truck", "bus") and _looks_like_emergency(frame, (x1, y1, x2, y2)):
            emergency = True

    return DetectionResult(vehicle_count=len(boxes), boxes=boxes, emergency_detected=emergency)


def detect_image_path(path: str) -> DetectionResult:
    frame = cv2.imread(path)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return detect_frame(frame)
