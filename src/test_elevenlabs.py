from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_DIR / "models" / "aasist" / "aasist.onnx"

FAKE_DIR = PROJECT_DIR / "data" / "fake"


# ============================================================
# LOAD AASIST
# ============================================================

print("Loading AASIST...")

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=[
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ],
)

model_input = session.get_inputs()[0]
model_output = session.get_outputs()[0]

print("AASIST loaded!")
print("Input shape:", model_input.shape)
print("Output shape:", model_output.shape)


# ============================================================
# FIND ELEVENLABS FILES
# ============================================================

audio_files = sorted(
    [
        f for f in FAKE_DIR.glob("*.mp3")
        if f.name.startswith("ElevenLabs_")
    ]
)

print()
print("ElevenLabs files found:", len(audio_files))


# ============================================================
# AASIST PREDICTION FOR ONE WINDOW
# ============================================================

def predict_window(waveform):

    target_length = 64600

    # ----------------------------------------
    # Make exactly 64600 samples
    # ----------------------------------------

    if len(waveform) >= target_length:

        waveform = waveform[:target_length]

    else:

        num_repeats = (
            int(target_length / len(waveform))
            + 1
        )

        waveform = np.tile(
            waveform,
            num_repeats
        )[:target_length]

    # ----------------------------------------
    # Add batch dimension
    # ----------------------------------------

    model_input_data = waveform[np.newaxis, :]

    # ----------------------------------------
    # AASIST inference
    # ----------------------------------------

    result = session.run(
        [model_output.name],
        {
            model_input.name: model_input_data
        },
    )

    logits = np.asarray(
        result[0][0],
        dtype=np.float32
    )

    # ----------------------------------------
    # Softmax
    # ----------------------------------------

    exp_scores = np.exp(
        logits - np.max(logits)
    )

    probabilities = (
        exp_scores / np.sum(exp_scores)
    )

    # AASIST class mapping:
    #
    # 0 = SPOOF / FAKE
    # 1 = BONAFIDE / REAL

    fake_probability = float(
        probabilities[0]
    )

    real_probability = float(
        probabilities[1]
    )

    return fake_probability, real_probability


# ============================================================
# ANALYZE ONE COMPLETE RECORDING
# ============================================================

def analyze_file(audio_path):

    waveform, sample_rate = librosa.load(
        audio_path,
        sr=16000,
        mono=True,
    )

    waveform = waveform.astype(
        np.float32
    )

    # 4-second window
    window_length = 64600

    # 2-second step
    hop_length = 32000

    fake_probabilities = []

    # ----------------------------------------
    # Sliding windows
    # ----------------------------------------

    start = 0

    while start < len(waveform):

        window = waveform[
            start:start + window_length
        ]

        # Ignore extremely tiny final windows
        if len(window) < 16000:
            break

        fake_probability, real_probability = (
            predict_window(window)
        )

        fake_probabilities.append(
            fake_probability
        )

        start += hop_length

    # ----------------------------------------
    # Combine windows
    # ----------------------------------------

    average_fake_probability = float(
        np.mean(fake_probabilities)
    )

    # Final decision
    #
    # 50% threshold for now

    if average_fake_probability >= 0.50:

        prediction = "FAKE / SPOOF"

    else:

        prediction = "REAL / BONAFIDE"

    return (
        len(fake_probabilities),
        average_fake_probability,
        prediction
    )


# ============================================================
# TEST ALL ELEVENLABS FILES
# ============================================================

correct = 0
total = len(audio_files)

print()
print("========================================")
print("     SLIDING-WINDOW AASIST TEST")
print("========================================")

for index, audio_path in enumerate(
    audio_files,
    start=1
):

    (
        number_of_windows,
        fake_probability,
        prediction
    ) = analyze_file(audio_path)

    if prediction == "FAKE / SPOOF":

        correct += 1

    print()
    print(
        f"[{index}/{total}] "
        f"{audio_path.name}"
    )

    print(
        "Windows analysed:",
        number_of_windows
    )

    print(
        f"Average fake probability: "
        f"{fake_probability * 100:.2f}%"
    )

    print(
        "Final prediction:",
        prediction
    )


# ============================================================
# SUMMARY
# ============================================================

print()
print("========================================")
print("             SUMMARY")
print("========================================")

print(
    "Total ElevenLabs files:",
    total
)

print(
    "Detected as fake:",
    correct
)

print(
    "Detected as real:",
    total - correct
)

if total > 0:

    detection_rate = (
        correct / total
    ) * 100

    print(
        f"Fake detection rate: "
        f"{detection_rate:.2f}%"
    )

print("========================================")