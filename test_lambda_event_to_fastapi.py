"""
test_lambda_event_to_fastapi.py

This script simulates an AWS Lambda S3 trigger locally by:
1. Reading a mock S3 event from test_event.json (e.g., output from S3 trigger).
2. Extracting the bucket and key.
3. Sending a POST request to the FastAPI /infer endpoint.

Usage:
    python test_lambda_event_to_fastapi.py

Pre-requisites:
- FastAPI server must be running locally on port 8000.
- test_event.json must contain a valid S3 event structure.
"""

import json
import requests

# Load mock Lambda event (similar to what S3 triggers send)
with open("test_event.json", "r") as f:
    event = json.load(f)

# Extract S3 bucket name and object key
bucket = event["Records"][0]["s3"]["bucket"]["name"]
key = event["Records"][0]["s3"]["object"]["key"]

# Send request to the running FastAPI server
response = requests.post(
    "http://localhost:8000/infer",
    json={"bucket": bucket, "key": key}
)

# Print response for debugging
print(f"Status: {response.status_code}")
print("Response JSON:")
print(response.json())
