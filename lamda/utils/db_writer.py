"""
db_writer.py

Handles uploading inference results to DynamoDB using the low-level Boto3 client.

Supports:
- Converting float values to Decimal (DynamoDB requirement)
- Parsing YOLO-style string results (e.g., "Sparrow 92.1%")
- Building DynamoDB-compatible entries with tag counts and metadata
- Uploading entries with detailed error handling and logging

Environment Variables:
- AWS_REGION: AWS region for the DynamoDB client (default: ap-southeast-2)
- DETECTION_TABLE_NAME: Name of the DynamoDB table to insert items into
"""

import os
import boto3
from botocore.exceptions import ClientError
from collections import Counter
from decimal import Decimal
from datetime import datetime

# Environment configuration
DYNAMODB_REGION = os.environ.get("AWS_REGION", "ap-southeast-2")
TABLE_NAME = os.environ.get("DETECTION_TABLE_NAME", "BirdDetections")
BASE_S3_URL = "https://birdtag-data-bucket.s3.amazonaws.com"

# Initialize DynamoDB low-level client
dynamodb_client = boto3.client("dynamodb", region_name=DYNAMODB_REGION)


def convert_floats_to_decimal(obj):
    """
    Recursively convert all float values in a nested structure to Decimal.

    Required for compatibility with DynamoDB, which does not accept Python floats.

    Parameters:
        obj (Any): A dict, list, or float-containing structure.

    Returns:
        Any: The same structure with all floats converted to Decimal.
    """
    if isinstance(obj, list):
        return [convert_floats_to_decimal(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: convert_floats_to_decimal(v) for k, v in obj.items()}
    elif isinstance(obj, float):
        return Decimal(str(obj))
    return obj


def parse_results_string(results):
    """
    Converts YOLO-style result strings (e.g., "Sparrow 92.54%") into structured dicts.

    Parameters:
        results (List[str]): Raw model output strings.

    Returns:
        List[Dict[str, float]]: List of { "label": str, "confidence": float } dicts.
    """
    parsed = []
    for result in results:
        try:
            label, conf = result.rsplit(" ", 1)
            confidence = float(conf.replace("%", "")) / 100
            parsed.append({"label": label.strip(), "confidence": confidence})
        except Exception as e:
            print(f"[WARN] Skipping result '{result}' due to parsing error: {e}")
    return parsed


def generate_dynamodb_entry(source_path, timestamp, media_type, results, base_url=BASE_S3_URL):
    """
    Builds a DynamoDB-compatible item from inference results.

    Parameters:
        source_path (str): Original S3 key.
        timestamp (str): ISO 8601 UTC timestamp of inference.
        media_type (str): One of "audio", "image", or "video".
        results (List[Dict[str, Any]]): Parsed detection results.

    Returns:
        dict: DynamoDB item formatted using low-level API types.
    """
    label_counts = Counter([res["label"] for res in results])
    tags_map = {
        label: {"N": str(count)} for label, count in label_counts.items()
    }

    file_url = f"{base_url}/{source_path}"
    filename = os.path.basename(source_path)
    name, ext = os.path.splitext(filename)
    thumbnail_url = f"{base_url}/thumbnails/{name}_thumb{ext}"

    return {
        "source_path":    {"S": source_path},
        "timestamp":      {"S": timestamp},
        "file_type":      {"S": media_type},
        "file_url":       {"S": file_url},
        "thumbnail_url":  {"S": thumbnail_url},
        "tags":           {"M": tags_map}
    }


def upload_to_dynamodb(entries):
    """
    Upload a list of items to DynamoDB using the low-level Boto3 client.

    Parameters:
        entries (List[dict]): List of DynamoDB-formatted items to upload.

    Returns:
        None
    """
    if not entries:
        print("[WARN] No entries to upload.")
        return

    success_count = 0

    for item in entries:
        try:
            dynamodb_client.put_item(
                TableName=TABLE_NAME,
                Item=item
            )
            success_count += 1
        except ClientError as e:
            print(f"[ERROR] DynamoDB ClientError → {e.response['Error']['Message']}")
            print(f"[DEBUG] Failed item → {item}")
        except Exception as e:
            print(f"[ERROR] Unexpected error → {str(e)}")
            print(f"[DEBUG] Failed item → {item}")

    print(f"[INFO] Successfully uploaded {success_count}/{len(entries)} entries to DynamoDB.")
