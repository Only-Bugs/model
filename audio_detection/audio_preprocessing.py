"""

Prepares audio input for BirdNET's TensorFlow Lite model by loading, resampling, 
padding/trimming, and reshaping it to the expected format.

This script disables Numba's cache to prevent permission-related warnings or slowdowns 
when running inside constrained environments like AWS Lambda.

Functions:
- preprocess_audio(): Loads and normalizes audio to the model’s input shape.
- process_audio_file: Alias for preprocess_audio to maintain consistent naming elsewhere.
"""

import os
os.environ["NUMBA_DISABLE_CACHE"] = "1"

import librosa
import numpy as np

def preprocess_audio(audio_path: str, target_sr: int = 48000, target_samples: int = 144000):
    """
    Load and preprocess an audio file for input into the BirdNET TFLite model.

    Parameters:
        audio_path (str): Path to the input audio file.
        target_sr (int): Target sample rate for resampling (default: 48000 Hz).
        target_samples (int): Total number of samples expected (default: 144000, i.e., 3 seconds at 48kHz).

    Returns:
        np.ndarray: A 2D numpy array of shape (1, target_samples), dtype float32,
                    ready for TFLite model inference.
    """
    y, sr = librosa.load(audio_path, sr=target_sr, mono=True)

    # Adjust audio length to match expected input shape
    if len(y) > target_samples:
        y = y[:target_samples]
    elif len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)), mode='constant')

    y = y.astype(np.float32)
    return y.reshape(1, target_samples)

# Alias for easier import elsewhere in the codebase
process_audio_file = preprocess_audio
