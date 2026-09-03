"""Central configuration, loaded from environment variables (.env)."""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    detection_confidence: float = float(os.getenv("DETECTION_CONFIDENCE", 0.25))
    emergency_confidence: float = float(os.getenv("EMERGENCY_CONFIDENCE", 0.5))
    min_green_seconds: int = int(os.getenv("MIN_GREEN_SECONDS", 10))
    max_green_seconds: int = int(os.getenv("MAX_GREEN_SECONDS", 60))

    # Vehicle classes we care about, from the COCO dataset YOLOv8 is
    # pretrained on. "truck" is used as a stand-in for larger emergency
    # vehicles when a dedicated emergency-vehicle model isn't loaded.
    vehicle_classes = {"car", "motorcycle", "bus", "truck", "bicycle"}


settings = Settings()
