from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort


PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "aasist" / "aasist.onnx"
FAKE_DIR = PROJECT_DIR / "data" / "fake"


def main() -> None:
    # Find the first MP3 in data/fake
    audio_files = sorted(FAKE_DIR.glob("*.mp3"))

    if not audio_files:
        raise FileNotFoundError(f"No MP3 files found in {FAKE_DIR}")

    audio_path = audio_files[0]

    print(f"Audio: {audio_path.name}")
    print(f"Model: {MODEL_PATH}")
    print()

    # Load the ONNX model.
    # CoreML is preferred on your Mac, with CPU as fallback.
    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=[
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ],
    )

    print("Providers used:")
    print(session.get_providers())
    print()

    # Load audio as mono 16 kHz float32.
    waveform, sample_rate = librosa.load(
        audio_path,
        sr=16000,
        mono=True,
    )

    waveform = waveform.astype(np.float32)

    print(f"Sample rate: {sample_rate}")
    print(f"Samples: {len(waveform)}")
    print()

    # Get model input/output information.
    model_input = session.get_inputs()[0]
    model_output = session.get_outputs()[0]

    print(f"Input name : {model_input.name}")
    print(f"Input shape: {model_input.shape}")
    print(f"Output name: {model_output.name}")
    print(f"Output shape: {model_output.shape}")
    print()

    # AASIST ONNX expects exactly 64,600 samples.
    target_length = 64600

    if len(waveform) < target_length:
        waveform = np.pad(
            waveform,
            (0, target_length - len(waveform)),
            mode="constant",
        )
    else:
        waveform = waveform[:target_length]

    # Add batch dimension:
    # (64600,) -> (1, 64600)
    model_input_data = waveform[np.newaxis, :]

    print(f"Model input shape: {model_input_data.shape}")
    print()

    # Run inference.
    result = session.run(
        [model_output.name],
        {model_input.name: model_input_data},
    )

    # Extract the two logits.
    raw_output = np.asarray(result[0][0], dtype=np.float32)

    print("Raw logits:", raw_output)

    # Convert logits to normalized probabilities.
    exp_scores = np.exp(raw_output - np.max(raw_output))
    probabilities = exp_scores / np.sum(exp_scores)

    class_0_probability = float(probabilities[0])
    class_1_probability = float(probabilities[1])

    predicted_class = int(np.argmax(probabilities))

    print()
    print("========================================")
    print("       AASIST VOICE ANALYSIS")
    print("========================================")
    print(f"Class 0 probability: {class_0_probability * 100:.2f}%")
    print(f"Class 1 probability: {class_1_probability * 100:.2f}%")
    print(f"Predicted class:     {predicted_class}")
    print("========================================")


if __name__ == "__main__":
    main()