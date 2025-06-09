"""
video_inference.py

Runs object detection on video files using the Ultralytics YOLO model 
and the `supervision` annotation toolkit. Supports:
- Annotating bounding boxes and class labels per frame
- Tracking object motion with ByteTrack
- Exporting annotated video to disk (optional)

The model and annotation tools are optimized for runtime reuse,
and results are returned as aggregated label+confidence strings.

Dependencies:
- cv2 (OpenCV)
- ultralytics
- supervision
- ByteTrack (via supervision)
"""

import logging
from ultralytics import YOLO
import supervision as sv
import cv2 as cv
import os

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load YOLO model once globally to avoid repeated cold starts
MODEL_PATH = "./model.pt"
yolo_model = YOLO(MODEL_PATH)

def run_video_detection(video_path, result_filename=None, output_path=None, confidence=0.5):
    """
    Run bird detection on a video using YOLOv8, with real-time tracking and annotations.

    Parameters:
        video_path (str): Path to the input video file.
        result_filename (str, optional): Filename for the annotated output video (e.g., 'result.avi').
        output_path (str, optional): Directory to save the annotated video if result_filename is set.
        confidence (float): Minimum confidence threshold to include detections (default: 0.5).

    Returns:
        List[str]: A list of detection labels with confidence scores aggregated from all frames.
            Format: ["label1 91.2%", "label2 84.6%", ...]
    
    Raises:
        IOError: If the input video cannot be opened.
        Exception: For any unexpected processing failures.
    """
    collected_labels = []

    try:
        # Extract video metadata
        video_info = sv.VideoInfo.from_video_path(video_path=video_path)
        w, h, fps = int(video_info.width), int(video_info.height), int(video_info.fps)

        # Configure annotation tools based on video resolution
        thickness = sv.calculate_optimal_line_thickness(video_info.resolution_wh)
        text_scale = sv.calculate_optimal_text_scale(video_info.resolution_wh)

        box_annotator = sv.BoxAnnotator(thickness=thickness, color_lookup=sv.ColorLookup.TRACK)
        label_annotator = sv.LabelAnnotator(
            text_scale=text_scale,
            text_thickness=thickness,
            text_position=sv.Position.TOP_LEFT,
            color_lookup=sv.ColorLookup.TRACK
        )

        # Tracker for maintaining object IDs across frames
        tracker = sv.ByteTrack(frame_rate=fps)
        class_dict = yolo_model.names

        # Prepare output video writer if saving results
        if result_filename and output_path:
            os.makedirs(output_path, exist_ok=True)
            save_path = os.path.join(output_path, result_filename)
            out = cv.VideoWriter(save_path, cv.VideoWriter_fourcc(*"XVID"), fps, (w, h))
        else:
            out = None

        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Unable to open video file: {video_path}")

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            result = yolo_model(frame)[0]
            detections = sv.Detections.from_ultralytics(result)

            # Apply tracking and confidence filtering
            detections = tracker.update_with_detections(detections)
            detections = detections[detections.confidence > confidence]

            # Extract detection labels
            labels = [
                f"{class_dict[cls]} {conf*100:.1f}%"
                for cls, conf in zip(detections.class_id, detections.confidence)
            ]

            if labels:
                logger.debug(f"[FRAME {frame_count}] Detections: {labels}")
            collected_labels.extend(labels)

            # Annotate and optionally save frame
            box_annotator.annotate(frame, detections=detections)
            label_annotator.annotate(frame, detections=detections, labels=labels)

            if out:
                out.write(frame)

    except Exception as e:
        logger.error(f"Video processing failed: {e}")

    finally:
        # Release resources
        cap.release()
        if out:
            out.release()
        logger.info("Video processing complete. Resources released.")

    return collected_labels
