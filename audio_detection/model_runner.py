"""
model_runner.py

Performs audio inference using the BirdNET TFLite model.

This module defines the BirdNetRunner class which handles:
1. Loading the TFLite model and labels only once (singleton-style).
2. Preprocessing and reshaping audio to expected model format.
3. Running inference and extracting top predictions with softmax scores.

The model expects mono audio sampled at 32kHz for a duration of 4.5 seconds
(144000 samples). The output is softmax-normalized and sorted by confidence.

Environment:
- NUMBA_DISABLE_CACHE is set to prevent Numba cache issues in constrained runtimes.
"""

import os
import tensorflow as tf
import numpy as np
import librosa
from scipy.special import softmax

# Set global constants for audio format
SAMPLE_RATE = 32000
DURATION = 4.5
NUM_SAMPLES = int(SAMPLE_RATE * DURATION)

# Disable numba caching for environments like AWS Lambda
os.environ["NUMBA_DISABLE_CACHE"] = "1"

class BirdNetRunner:
    """
    A singleton-style runner class for performing inference with BirdNET TFLite models.

    Methods:
        - load_model(): Loads and initializes the TFLite interpreter.
        - load_labels(): Loads and caches label list from file.
        - run_audio_inference(): Performs full inference on an audio file.
    """
    _model = None
    _labels = None

    @classmethod
    def load_model(cls, model_path):
        """
        Load and return the TFLite model interpreter.

        Parameters:
            model_path (str): Path to the .tflite model file.

        Returns:
            tf.lite.Interpreter: The loaded and allocated model interpreter.
        """
        if cls._model is None:
            print(f"[MODEL] Loading model from {model_path}")
            cls._model = tf.lite.Interpreter(model_path=model_path)
            cls._model.allocate_tensors()
            print("[MODEL] Model loaded and allocated")
        return cls._model

    @classmethod
    def load_labels(cls, labels_path):
        """
        Load and return the list of class labels from a file.

        Parameters:
            labels_path (str): Path to the label text file.

        Returns:
            List[str]: List of labels, one per line in the file.
        """
        if cls._labels is None:
            print(f"[LABELS] Loading labels from {labels_path}")
            with open(labels_path, "r") as f:
                cls._labels = [line.strip() for line in f.readlines()]
            print("[LABELS] Labels loaded")
        return cls._labels

    @classmethod
    def run_audio_inference(cls, audio_path):
        """
        Perform bird species inference on an input audio file.

        Parameters:
            audio_path (str): Path to the .wav or .mp3 audio file.

        Returns:
            List[Dict[str, Any]]: A list of predictions in the format:
                [{ "label": str, "confidence": float }, ...]

        Raises:
            ValueError: If the preprocessed audio shape doesn't match model input.
        """
        print(f"[RUNNER] Running inference for {audio_path}")

        # Load and preprocess audio
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)
        if len(audio) < NUM_SAMPLES:
            pad_width = NUM_SAMPLES - len(audio)
            audio = np.pad(audio, (0, pad_width))
        else:
            audio = audio[:NUM_SAMPLES]

        audio_input = np.expand_dims(audio, axis=0).astype(np.float32)

        model = cls.load_model("audio_detection/model_files/BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite")
        labels = cls.load_labels("audio_detection/model_files/BirdNET_GLOBAL_6K_V2.4_Labels.txt")

        input_details = model.get_input_details()
        output_details = model.get_output_details()

        # Shape validation
        expected_shape = tuple(input_details[0]['shape'])
        if audio_input.shape != expected_shape:
            raise ValueError(f"[SHAPE ERROR] Expected {expected_shape}, got {audio_input.shape}")

        # Inference
        model.set_tensor(input_details[0]['index'], audio_input)
        model.invoke()
        output_data = model.get_tensor(output_details[0]['index'])[0]

        print("[DEBUG] Raw output (first 10):", output_data[:10])

        output_data = softmax(output_data)
        print("[DEBUG] Softmax output (first 10):", output_data[:10])

        # Get top predictions
        top_indices = np.argsort(output_data)[::-1][:10]
        results = []

        for i in top_indices:
            confidence = float(output_data[i])
            label = labels[i]
            print(f"[INFO] Candidate: {label} -> {confidence:.4f}")
            if confidence > 0.01:
                results.append({"label": label, "confidence": confidence})
            else:
                print(f"[WARN] Skipping low confidence: {label} @ {confidence:.4f}")

        print(f"[RUNNER] Top predictions: {results}")
        return results
