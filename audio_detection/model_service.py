"""
model_service.py

Provides modular utilities for running bird sound classification inference 
using the BirdNET TensorFlow Lite model. This service is designed for server-side 
use (e.g., in FastAPI) and expects a preprocessed spectrogram as input.

It uses lazy-loading and global caching for model and label loading 
to optimize runtime performance in constrained environments (e.g., Lambda or Docker).

Functions:
- load_model(): Loads the BirdNET TFLite model and returns interpreter metadata.
- load_labels(): Loads and caches the class label names.
- run_inference(): Runs inference on a spectrogram input and returns top predictions.
"""

import os
import numpy as np
import tensorflow as tf

# Paths to model and label resources
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model_files")
MODEL_PATH = os.path.join(MODEL_DIR, "BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite")
LABELS_PATH = os.path.join(MODEL_DIR, "BirdNET_GLOBAL_6K_V2.4_Labels.txt")

# Internal caches for model and labels
_interpreter = None
_input_details = None
_output_details = None
_labels = None


def load_model():
    """
    Loads and returns the BirdNET TFLite model interpreter along with 
    its input and output tensor details.

    Returns:
        Tuple[tf.lite.Interpreter, List, List]: The model interpreter, 
        input tensor info, and output tensor info.
    """
    global _interpreter, _input_details, _output_details
    if _interpreter is None:
        print("[MODEL] Loading and allocating BirdNET model...")
        _interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        _interpreter.allocate_tensors()
        _input_details = _interpreter.get_input_details()
        _output_details = _interpreter.get_output_details()
        print("[MODEL] Model loaded and ready.")
    return _interpreter, _input_details, _output_details


def load_labels():
    """
    Loads and returns the list of class labels associated with the BirdNET model.

    Returns:
        List[str]: List of label strings.
    """
    global _labels
    if _labels is None:
        print("[LABELS] Loading label file...")
        with open(LABELS_PATH, "r") as f:
            _labels = [line.strip() for line in f.readlines()]
        print(f"[LABELS] Loaded {len(_labels)} labels.")
    return _labels


def run_inference(spectrogram: np.ndarray):
    """
    Runs BirdNET inference on a precomputed spectrogram input.

    Parameters:
        spectrogram (np.ndarray): The preprocessed input spectrogram 
        (usually shape (96, 64) or as required by the model).

    Returns:
        List[Dict[str, float]]: Top 5 predictions with their associated scores.
            Format: [{ "label": str, "score": float }, ...]
    
    Raises:
        ValueError: If the spectrogram shape cannot be reshaped to match model input.
    """
    interpreter, input_details, output_details = load_model()
    labels = load_labels()

    input_shape = input_details[0]['shape']
    print(f"[DEBUG] Model expects shape: {input_shape}")
    print(f"[DEBUG] Spectrogram shape before reshape: {spectrogram.shape}")

    # Ensure the input shape matches model requirements
    if spectrogram.shape != tuple(input_shape):
        try:
            spectrogram = np.reshape(spectrogram, tuple(input_shape))
            print(f"[DEBUG] Reshaped spectrogram to: {spectrogram.shape}")
        except Exception as e:
            raise ValueError(f"Failed to reshape spectrogram: {e}")

    interpreter.set_tensor(input_details[0]['index'], spectrogram.astype(np.float32))
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    top_indices = np.argsort(output_data)[::-1][:5]

    return [{
        "label": labels[i],
        "score": float(output_data[i])
    } for i in top_indices]


# Optional CLI test for local dev
if __name__ == "__main__":
    dummy_input = np.random.rand(96, 64)
    results = run_inference(dummy_input)
    for r in results:
        print(f"{r['label']} ({r['score']:.4f})")
