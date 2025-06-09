"""
settings.py

Loads and manages configuration for the audio detection model using environment variables.
This module uses Pydantic's BaseSettings for structured settings management, with defaults 
falling back to values suitable for local development or testing.

Environment Variables:
- MODEL_DIR: Directory path where model files are stored.
- MODEL_FILENAME: Name of the TFLite model file.
- LABELS_FILENAME: Name of the label mapping file.
"""

from dotenv import load_dotenv
import os
from pydantic import BaseSettings

# Load environment variables from a .env file (if present)
load_dotenv()

class Settings(BaseSettings):
    """
    Defines model-related configuration settings for audio detection.

    Attributes:
        model_dir (str): Directory path to the model files.
        model_filename (str): Filename of the TensorFlow Lite model.
        labels_filename (str): Filename of the labels file for class mapping.
    """
    model_dir: str = os.getenv("MODEL_DIR", "audio_detection/model_files")
    model_filename: str = os.getenv("MODEL_FILENAME", "birdnet.tflite")
    labels_filename: str = os.getenv("LABELS_FILENAME", "labels.txt")

# Global instance of settings to be used across the application
settings = Settings()
