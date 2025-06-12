# BirdTag FastAPI Inference Service

This repository provides a local simulation pipeline for the **BirdTag** backend inference system. It mimics an AWS Lambda function that triggers on S3 file uploads, using a FastAPI application to detect media types (image, audio, or video), perform inference using the appropriate model, and write results to DynamoDB.

## Features

* Automatic media type detection based on file extension.
* Model-specific routing:

  * **YOLOv8** for image and video files.
  * **BirdNET V2.4** for audio detection.
* Tag-count-based result formatting stored in DynamoDB.
* Optionally copies processed media into a secondary folder (`temp/` or `processed/`) in S3.

## Setup Instructions

### 1. Install Requirements

Ensure Python 3.10+ is installed. Then:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run the FastAPI Server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

**Prerequisites:**
Ensure your environment is authenticated to AWS using either environment variables or `~/.aws/credentials`. Your IAM identity must have:

* `s3:GetObject` for reading uploaded files.
* `dynamodb:PutItem` for writing results.
* `s3:PutObject` for writing to `temp/` or `processed/` folders.

### 3. Send a Test Event to `/infer`

Create a file named `test_event.json` with the following structure:

```json
{
  "bucket": "birdtag-data-bucket",
  "key": "uploads/sample.jpg"
}
```

Then simulate the trigger:

```bash
curl -X POST http://localhost:8000/infer \
     -H "Content-Type: application/json" \
     -d @test_event.json
```

## Sample Output

### Image Inference

```
[INFO] File uploaded → Bucket: birdtag-data-bucket, Key: uploads/kingfisher_3.jpg
[INFO] Detected media type: image
[IMAGE] Processing: /tmp/input_media
...
[RESULT] Inference Results:
{
  "media_type": "image",
  "source_path": "uploads/kingfisher_3.jpg",
  "results": ["Kingfisher 93.70%"]
}
[DEBUG] DynamoDB entry created:
{
  "source_path": "uploads/kingfisher_3.jpg",
  "file_type": "image",
  "tags": {"Kingfisher": 1}
}
```

### Audio Inference

```
[INFO] Detected media type: audio
[AUDIO] Processing: /tmp/input_media
[MODEL] Loading BirdNET TFLite model...
[RUNNER] Top predictions: [...]
[RESULT] Inference Results:
{
  "media_type": "audio",
  "source_path": "uploads/sample.wav",
  "results": [
    {"label": "Species_X", "confidence": 0.53},
    {"label": "Species_Y", "confidence": 0.11}
  ]
}
[DEBUG] DynamoDB entry created with tag counts:
{
  "tags": {
    "Species_X": 1,
    "Species_Y": 1
  }
}
```

### Video Inference

```
[INFO] Detected media type: video
[VIDEO] Processing: /tmp/input_media
...
[RESULT] Inference Results:
{
  "media_type": "video",
  "source_path": "uploads/sample.mp4",
  "results": ["Kingfisher 92.1%", "Sparrow 90.6%", ...]
}
[DEBUG] DynamoDB entry created with tag counts:
{
  "tags": {
    "Kingfisher": 107,
    "Sparrow": 105
  }
}
```

## Troubleshooting

### Access Denied Errors (S3 Uploads)

If you see:

```
[ERROR] Failed to copy media to temp/: AccessDenied: ... s3:PutObject on resource ...
```

Ensure your IAM identity has permissions:

```json
{
  "Effect": "Allow",
  "Action": "s3:PutObject",
  "Resource": "arn:aws:s3:::birdtag-data-bucket/temp/*"
}
```

### Verifying Output

* Logs are printed to your CLI.
* DynamoDB entries can be viewed via the AWS Console under the configured table.
* S3 `temp/` or `processed/` folders should reflect copies of the input media.


