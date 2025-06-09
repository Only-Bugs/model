"""
app.py

FastAPI backend for triggering bird detection inference on S3-stored media files.

Exposes a `/infer` POST endpoint that:
- Downloads media from a provided S3 bucket and key
- Detects the media type (audio, image, video)
- Routes to the correct model (BirdNET or YOLO)
- Parses and stores results in DynamoDB

Dependencies:
- boto3
- FastAPI
- Ultralytics YOLO
- Custom inference utils from lamda/
"""

from fastapi import FastAPI, Request
from pydantic import BaseModel
import boto3, os, urllib.parse, json
from datetime import datetime
from decimal import Decimal
from collections import Counter

from lamda.inference.audio_inference import run_audio_detection
from lamda.inference.image_inference import detect_birds_in_image
from lamda.inference.video_inference import run_video_detection
from lamda.utils.db_writer import upload_to_dynamodb

app = FastAPI()
s3 = boto3.client("s3")

TEMP_FILE_PATH = "/tmp/input_media"
BUCKET_NAME = os.getenv("BUCKET_NAME", "birdtag-data-bucket")


class S3Event(BaseModel):
    """
    Schema for incoming POST request to /infer

    Attributes:
        bucket (str): S3 bucket name.
        key (str): S3 key of the uploaded media file.
    """
    bucket: str
    key: str


def download_from_s3(bucket, key, local_path=TEMP_FILE_PATH):
    """
    Download a file from S3 to the Lambda /tmp directory.

    Parameters:
        bucket (str): S3 bucket name.
        key (str): Object key.
        local_path (str): Where to save the file locally.

    Returns:
        str: Local file path.
    """
    s3.download_file(bucket, key, local_path)
    print(f"[INFO] Downloaded file to {local_path}")
    return local_path


def save_results(results):
    """
    Save inference results to DynamoDB. Handles both audio and image/video formats.

    Parameters:
        results (dict): Contains source_path, media_type, and results from inference.
    """
    try:
        timestamp = datetime.utcnow().isoformat()
        source_path = results["source_path"]
        media_type = results["media_type"]
        entries = results["results"]

        if media_type == "audio":
            top_result = max(entries, key=lambda x: x["confidence"])
            item = {
                "source_path": {"S": source_path},
                "timestamp": {"S": timestamp},
                "file_type": {"S": "audio"},
                "file_url": {"S": f"https://{BUCKET_NAME}.s3.amazonaws.com/{source_path}"},
                "label": {"S": top_result["label"]},
                "confidence": {"N": f"{top_result['confidence']:.3f}"}
            }
            upload_to_dynamodb([item])

        else:
            for result in entries:
                label, confidence = result.split()
                label = label.strip()
                confidence = float(confidence.strip('%')) / 100
                item = {
                    "source_path": {"S": source_path},
                    "timestamp": {"S": timestamp},
                    "file_type": {"S": media_type},
                    "file_url": {"S": f"https://{BUCKET_NAME}.s3.amazonaws.com/{source_path}"},
                    "label": {"S": label},
                    "confidence": {"N": f"{confidence:.3f}"}
                }
                upload_to_dynamodb([item])
    except Exception as e:
        print(f"[ERROR] Failed to save results: {e}")
        raise


@app.get("/")
def read_root():
    """
    Health check endpoint.

    Returns:
        dict: Service status message.
    """
    return {"message": "BirdTag API is running 🎯"}


@app.post("/infer")
async def infer(event: S3Event):
    """
    Trigger inference by uploading a JSON payload with S3 bucket/key.

    Example Payload:
    {
        "bucket": "birdtag-data-bucket",
        "key": "uploads/audiofile.wav"
    }

    Parameters:
        event (S3Event): Validated payload with bucket and key.

    Returns:
        dict: Status and inference results.
    """
    bucket = event.bucket
    key = urllib.parse.unquote_plus(event.key)
    print(f"[INFO] Received S3 file: {key} in bucket: {bucket}")

    file_ext = key.lower().split(".")[-1]
    media_type = (
        "audio" if file_ext in ["mp3", "wav", "flac"]
        else "image" if file_ext in ["jpg", "jpeg", "png"]
        else "video" if file_ext in ["mp4", "mov", "avi"]
        else "unknown"
    )

    if media_type == "unknown":
        return {"error": "Unsupported media type"}

    local_file = download_from_s3(bucket, key)
    print(f"[INFO] Detected media type: {media_type}")

    if media_type == "audio":
        results = run_audio_detection(local_file)
    elif media_type == "image":
        results = detect_birds_in_image(local_file)
    elif media_type == "video":
        results = run_video_detection(local_file)
    else:
        raise ValueError("Unsupported media type")

    final_result = {
        "media_type": media_type,
        "source_path": key,
        "results": results
    }

    print("[RESULT] Inference Results:", json.dumps(final_result, indent=2))
    save_results(final_result)

    return {
        "status": "success",
        "media_type": media_type,
        "results": results
    }
