"""
lambda_function.py

AWS Lambda entrypoint for the BirdTag project. Automatically triggers when a new file 
is uploaded to the configured S3 bucket. Handles audio, image, and video inference 
using model-specific detectors, and stores the results in DynamoDB.

Core Tasks:
- Detect media type from file extension
- Route to appropriate model runner (BirdNET for audio, YOLO for image/video)
- Parse and format detection results
- Write processed results to DynamoDB
- Archive media file to 'temp/' S3 folder

Supports: .mp3, .wav, .flac, .jpg, .jpeg, .png, .mp4, .mov, .avi

AWS Resources:
- S3 (trigger + storage)
- DynamoDB (for storing detection results)
"""

import time
cold_start_time = time.time()

import json
import urllib.parse
import boto3
import os
from datetime import datetime
from decimal import Decimal
from collections import Counter

print("[DEBUG] ✅ Core Python modules loaded.")

from lamda.inference.audio_inference import run_audio_detection
from lamda.inference.image_inference import detect_birds_in_image
from lamda.inference.video_inference import run_video_detection
from lamda.utils.db_writer import upload_to_dynamodb
from lamda.utils.copy_to_temp import copy_media_to_s3_folder

print(f"[DEBUG] ✅ Custom modules imported in {time.time() - cold_start_time:.2f} seconds")

s3 = boto3.client("s3")
TEMP_FILE_PATH = "/tmp/input_media"

print("[INFO] Lambda function cold-start initialized.")


def download_from_s3(bucket, key, local_path=TEMP_FILE_PATH):
    """
    Download a file from S3 to local /tmp directory.

    Parameters:
        bucket (str): Source S3 bucket name.
        key (str): Key of the object to download.
        local_path (str): Local path to save the file.

    Returns:
        str: Path to the downloaded local file.
    """
    s3.download_file(bucket, key, local_path)
    print(f"[INFO] Downloaded file to {local_path}")
    return local_path


def generate_dynamodb_entry(source_path, timestamp, media_type, parsed_results):
    """
    Construct a DynamoDB-compatible entry for audio-based predictions.

    Parameters:
        source_path (str): S3 key of the original media file.
        timestamp (str): UTC ISO timestamp of inference.
        media_type (str): Type of media ("audio", etc.)
        parsed_results (list): List of dicts with 'label' and 'confidence'.

    Returns:
        dict: DynamoDB item in AttributeValue format.
    """
    top_result = parsed_results[0] if parsed_results else {}

    bucket_name = os.environ.get("BUCKET_NAME", "birdtag-data-bucket")
    file_url = f"https://{bucket_name}.s3.amazonaws.com/{source_path}"

    return {
        "source_path": {"S": source_path},
        "timestamp": {"S": timestamp},
        "media_type": {"S": media_type},
        "file_type": {"S": media_type},
        "file_url": {"S": file_url},
        "label": {"S": top_result.get("label", "unknown")},
        "confidence": {"N": str(Decimal(str(top_result.get("confidence", -1.0))))}
    }


def generate_video_image_entry(source_path, timestamp, media_type, parsed_results):
    """
    Generate a DynamoDB entry for video or image detection results.

    Parameters:
        source_path (str): S3 key of the media file.
        timestamp (str): ISO UTC timestamp.
        media_type (str): "image" or "video".
        parsed_results (list): List of string labels (e.g., "Bird 93.1%").

    Returns:
        dict: DynamoDB-formatted item with tag counts.
    """
    bucket_name = os.environ.get("BUCKET_NAME", "birdtag-data-bucket")
    file_url = f"https://{bucket_name}.s3.amazonaws.com/{source_path}"

    tag_counts = Counter()
    for result in parsed_results:
        label = result.get("label")
        if label:
            tag_counts[label] += 1

    tag_map = {label: {"N": str(count)} for label, count in tag_counts.items()}

    return {
        "source_path": {"S": source_path},
        "timestamp": {"S": timestamp},
        "file_type": {"S": media_type},
        "file_url": {"S": file_url},
        "tags": {"M": tag_map}
    }


def save_results(results):
    """
    Unified saver for inference results to DynamoDB.

    Parameters:
        results (dict): Dict with keys `source_path`, `media_type`, and `results` (inference output).
    """
    try:
        timestamp = datetime.utcnow().isoformat()
        source_path = results["source_path"]
        media_type = results["media_type"]
        entries = results["results"]

        # Format results into tag counts
        tag_counts = Counter()
        if media_type == "audio":
            for entry in entries:
                label = entry.get("label")
                if label:
                    tag_counts[label] += 1
        else:
            for entry in entries:
                label, _ = entry.split()
                tag_counts[label.strip()] += 1

        tag_map = {label: {"N": str(count)} for label, count in tag_counts.items()}

        bucket_name = os.environ.get("BUCKET_NAME", "birdtag-data-bucket")
        file_url = f"https://{bucket_name}.s3.amazonaws.com/{source_path}"

        item = {
            "source_path": {"S": source_path},
            "timestamp": {"S": timestamp},
            "file_type": {"S": media_type},
            "file_url": {"S": file_url},
            "tags": {"M": tag_map}
        }

        upload_to_dynamodb([item])
        print("[INFO] ✅ Saved tags to DynamoDB")
        print("[DEBUG] DynamoDB entry created:", json.dumps(item, indent=2))

    except Exception as e:
        print(f"[ERROR] Failed to save results: {e}")


def lambda_handler(event, context):
    """
    AWS Lambda entrypoint. Triggered via S3 upload event.

    Parameters:
        event (dict): AWS event payload from S3 trigger.
        context (LambdaContext): Runtime context (unused).

    Returns:
        dict: HTTP-like status response.
    """
    handler_start = time.time()
    print("[INFO] Event Received:", json.dumps(event, indent=2))

    try:
        for record in event["Records"]:
            bucket = record["s3"]["bucket"]["name"]
            key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])
            print(f"[INFO] File uploaded → Bucket: {bucket}, Key: {key}")

            # Determine file type
            file_ext = key.lower().split(".")[-1]
            media_type = (
                "audio" if file_ext in ["mp3", "wav", "flac"]
                else "image" if file_ext in ["jpg", "jpeg", "png"]
                else "video" if file_ext in ["mp4", "mov", "avi"]
                else "unknown"
            )

            print(f"[INFO] Detected media type: {media_type}")
            local_file = download_from_s3(bucket, key)

            infer_start = time.time()

            # Route to inference
            if media_type == "audio":
                results = run_audio_detection(local_file)
            elif media_type == "image":
                results = detect_birds_in_image(local_file, output_path=None)
            elif media_type == "video":
                results = run_video_detection(local_file, output_path=None)
            else:
                raise ValueError("Unsupported media type")

            infer_end = time.time()
            print(f"[DEBUG] Inference for {media_type} took {infer_end - infer_start:.2f} seconds")

            final_result = {
                "media_type": media_type,
                "source_path": key,
                "results": results
            }

            print("[RESULT] Inference Results:")
            print(json.dumps(final_result, indent=2))

            save_results(final_result)

        print(f"[DEBUG] Handler completed in {time.time() - handler_start:.2f} seconds")
        copy_media_to_s3_folder(bucket, key, dest_folder="temp/")
        return {
            "statusCode": 200,
            "body": json.dumps("Inference complete.")
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[DEBUG] Error: {e}")
        return {"statusCode": 500, "body": "Error processing media."}
