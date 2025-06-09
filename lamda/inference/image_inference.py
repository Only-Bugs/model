"""
image_inference.py

Performs bird detection on a static image using the YOLOv8 object detection model 
from Ultralytics. Optionally draws bounding boxes on the image and saves the result.

The model is loaded once at cold start for efficiency in environments like 
Lambda, Docker, or Fargate.

Functions:
- detect_birds_in_image(): Runs YOLOv8 inference and returns detected birds with metadata.

Environment:
- Requires a `model.pt` file in the working directory.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os

# Load YOLO model once (cold start optimization)
MODEL_PATH = "./model.pt"
yolo_model = YOLO(MODEL_PATH)

def detect_birds_in_image(image_path, output_path=None, confidence_threshold=0.5):
    """
    Detects birds in an input image using a YOLOv8 model.

    Parameters:
        image_path (str): Path to the input image file.
        output_path (str, optional): If provided, saves the annotated image with bounding boxes.
        confidence_threshold (float): Minimum confidence score to include a detection (default: 0.5).

    Returns:
        List[str]: A list of detected bird labels with confidence scores 
                   in the format: ["label1 87.50%", "label2 75.34%", ...].

    Raises:
        ValueError: If the image cannot be loaded from the provided path.
    """
    print(f"[IMAGE] Processing: {image_path}")

    # Read image from disk
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image could not be loaded")

    # Run YOLOv8 inference
    results = yolo_model(image)
    boxes = results[0].boxes

    detections = []
    labels = []

    for box in boxes:
        conf = float(box.conf[0])
        if conf < confidence_threshold:
            continue

        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = yolo_model.names[cls_id]
        labels.append(f"{label} {conf * 100:.2f}%")

        detections.append({
            "label": label,
            "confidence": round(conf, 2),
            "bbox": [x1, y1, x2, y2]
        })

        # Draw annotated bounding box if output path is specified
        if output_path:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save annotated image if requested
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, image)

    return labels
