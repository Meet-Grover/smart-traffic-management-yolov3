# Smart Traffic Management System (YOLO-based)

## Overview
This project demonstrates real-time vehicle detection using a YOLO-based object detection model. The system processes video input to identify vehicles and visualize traffic conditions, making it suitable for smart traffic monitoring applications.

## Project Timeline

- Initial development: July 2025
- Refactoring, cleanup, and GitHub publication: January 2026


## Objective
The objective of this project was to understand how real-time object detection works and how YOLO can be applied to traffic analysis scenarios.

## Technologies Used
- Python
- OpenCV
- YOLO (pre-trained model)
- NumPy
- Flask

## System Workflow
1. Video frames are captured using OpenCV.
2. Each frame is processed through the YOLO detection pipeline.
3. Vehicles are detected using bounding boxes and class labels.
4. The output is visualized frame by frame.

## Key Learnings
- Understanding YOLO’s single-stage detection approach.
- Observing real-time performance constraints.
- Studying detection behavior under varying lighting and traffic density.

## Notes
- This project uses a pretrained YOLOv3 model for vehicle detection. 
- The focus of the work is on system integration, inference pipeline design, and traffic decision logic rather than model training.
- YOLOv3 weights are excluded from version control due to size limits.
- Download them separately and place inside the `yolo-coco/` directory.


## Limitations
- Accuracy decreases in low-light or crowded scenes.
- Real-time performance depends on system hardware.
- No traffic signal decision logic implemented.

## Future Enhancements
- Vehicle counting and congestion estimation.
- Signal timing optimization.
- Web-based dashboard for monitoring.

## Execution

This project focuses on system integration and inference pipeline design.
The YOLOv3 model is loaded using OpenCV DNN for vehicle detection.

Frame extraction from video is treated as a preprocessing step and is not
executed by default in this demo.

To validate the inference pipeline:

```bash
python -m src.yolo
```
