from pathlib import Path
from io import BytesIO

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf

from datasets import load_dataset, Audio
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_DIR / "models" / "aasist" / "aasist.onnx"


# ============================================================
# LOAD DATASET
# ============================================================

print("Loading ASVspoof2021-DF...")

ds = load_dataset(
    "SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF",
    split="test"
)

ds = ds.cast_column(
    "audio",
    Audio(decode=False)
)

print("Dataset loaded!")
print("Total samples:", len(ds))


# ============================================================
# SELECT BALANCED DATA
# ============================================================

real_samples = []
fake_samples = []

for sample in ds:

    if sample["label"] == 0:

        if len(real_samples) < 500:
            real_samples.append(sample)

    elif sample["label"] == 1:

        if len(fake_samples) < 500:
            fake_samples.append(sample)

    if len(real_samples) == 500 and len(fake_samples) == 500:
        break


samples = real_samples + fake_samples

print()
print("Selected REAL samples:", len(real_samples))
print("Selected FAKE samples:", len(fake_samples))
print("Total evaluation samples:", len(samples))


# ============================================================
# LOAD AASIST
# ============================================================

print()
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
print("Input:", model_input.name)
print("Output:", model_output.name)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_sample(sample):

    audio_bytes = sample["audio"]["bytes"]

    waveform, sample_rate = sf.read(
        BytesIO(audio_bytes),
        dtype="float32"
    )

    # Stereo -> mono
    if waveform.ndim > 1:
        waveform = np.mean(
            waveform,
            axis=1
        )

    # Resample -> 16 kHz
    if sample_rate != 16000:

        waveform = librosa.resample(
            waveform,
            orig_sr=sample_rate,
            target_sr=16000
        )

    waveform = waveform.astype(
        np.float32
    )

    # AASIST requires 64600 samples
    target_length = 64600

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

    # Add batch dimension
    model_input_data = waveform[
        np.newaxis,
        :
    ]

    # Run AASIST
    result = session.run(
        [model_output.name],
        {
            model_input.name:
            model_input_data
        }
    )

    logits = np.asarray(
        result[0][0],
        dtype=np.float32
    )

    # Softmax
    exp_scores = np.exp(
        logits - np.max(logits)
    )

    probabilities = (
        exp_scores /
        np.sum(exp_scores)
    )

    # IMPORTANT:
    # AASIST:
    # class 0 = SPOOF
    # class 1 = BONAFIDE

    aasist_class = int(
        np.argmax(probabilities)
    )

    # Convert to ASVspoof labels:
    # ASVspoof:
    # 0 = REAL
    # 1 = FAKE

    if aasist_class == 0:
        predicted_label = 1
    else:
        predicted_label = 0

    return predicted_label


# ============================================================
# RUN EVALUATION
# ============================================================

true_labels = []
predicted_labels = []

print()
print("Starting evaluation...")
print()


for i, sample in enumerate(samples):

    true_label = int(
        sample["label"]
    )

    predicted_label = predict_sample(
        sample
    )

    true_labels.append(
        true_label
    )

    predicted_labels.append(
        predicted_label
    )

    if (i + 1) % 50 == 0:

        print(
            f"Processed {i + 1}/"
            f"{len(samples)}"
        )


# ============================================================
# CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    true_labels,
    predicted_labels
)

precision = precision_score(
    true_labels,
    predicted_labels,
    zero_division=0
)

recall = recall_score(
    true_labels,
    predicted_labels,
    zero_division=0
)

f1 = f1_score(
    true_labels,
    predicted_labels,
    zero_division=0
)

cm = confusion_matrix(
    true_labels,
    predicted_labels
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print()
print("========================================")
print("        AASIST EVALUATION RESULTS")
print("========================================")

print(
    f"Accuracy : {accuracy * 100:.2f}%"
)

print(
    f"Precision: {precision * 100:.2f}%"
)

print(
    f"Recall   : {recall * 100:.2f}%"
)

print(
    f"F1 Score : {f1 * 100:.2f}%"
)

print()
print("Confusion Matrix:")
print(cm)

print()
print("========================================")