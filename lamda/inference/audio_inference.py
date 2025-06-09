"""
audio_inference.py

Handles the end-to-end workflow for running BirdNET model inference on audio files.

This module performs:
- File existence validation
- Calls into BirdNetRunner for raw inference
- Output schema validation and key normalization
- Structured error handling for graceful upstream failure reporting

Expects audio files to be valid and readable by `librosa` (e.g., .wav, .mp3).
"""

import os
import numpy as np
from audio_detection.model_runner import BirdNetRunner

def run_audio_detection(audio_path):
    """
    Performs bird sound detection on a given audio file using the BirdNET model.

    Parameters:
        audio_path (str): Path to the input audio file.

    Returns:
        List[Dict[str, float]]: List of predictions with "label" and "confidence" fields.
            Format: [{ "label": str, "confidence": float }, ...]

    Raises:
        FileNotFoundError: If the input audio file is not found.
        Exception: If model inference fails or produces malformed results.
    """
    print(f"[AUDIO] Processing: {audio_path}")

    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

    try:
        # Run BirdNET model inference and get results
        results = BirdNetRunner.run_audio_inference(audio_path)

        # Validate output structure
        validated = []
        for result in results:
            if isinstance(result, dict) and all(k in result for k in ("label", "confidence")):
                validated.append({
                    "label": result["label"],
                    "confidence": result["confidence"]
                })
            else:
                print(f"[WARN] Skipping malformed result: {result}")

        return validated

    except Exception as e:
        print(f"[ERROR] Audio detection failed: {e}")
        raise
