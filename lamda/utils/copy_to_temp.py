import os
import boto3

s3 = boto3.client("s3")

def copy_media_to_s3_folder(bucket, source_key, dest_folder="temp/"):
    try:
        # Get original file content
        response = s3.get_object(Bucket=bucket, Key=source_key)
        file_bytes = response['Body'].read()

        # Extract file extension and name
        filename = os.path.basename(source_key)
        ext = os.path.splitext(filename)[1].lower()
        dest_key = f"{dest_folder}{filename}"

        # Infer content type
        content_type = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".flac": "audio/flac",
            ".mp4": "video/mp4",
            ".mov": "video/quicktime",
            ".avi": "video/x-msvideo"
        }.get(ext, "binary/octet-stream")

        # Put the object in the new location
        s3.put_object(
            Bucket=bucket,
            Key=dest_key,
            Body=file_bytes,
            ContentType=content_type
        )

        print(f"[INFO] ✅ Copied '{source_key}' → '{dest_key}' in bucket '{bucket}'")
        return dest_key

    except Exception as e:
        print(f"[ERROR] Failed to copy media to {dest_folder}: {e}")
