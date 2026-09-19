from pathlib import Path
import random

import librosa
import numpy as np
import torch
import torch.nn as nn

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

FAKE_DIR = PROJECT_DIR / "data" / "fake_wav"
REAL_DIR = PROJECT_DIR / "data" / "real_wav"

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "cnn"
    / "voice_cnn_windows_best.pth"
)


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_RATE = 16000

WINDOW_SIZE = 16000 * 4

HOP_SIZE = 16000 * 2

N_MELS = 64

N_FFT = 1024

HOP_LENGTH = 256

RANDOM_SEED = 42


# ============================================================
# REPRODUCE THE SAME RECORDING SPLIT
# ============================================================

random.seed(RANDOM_SEED)

fake_files = sorted(FAKE_DIR.glob("*.wav"))
real_files = sorted(REAL_DIR.glob("*.wav"))

random.shuffle(fake_files)
random.shuffle(real_files)


def split_files(files):

    total = len(files)

    train_end = int(total * 0.70)

    val_end = int(total * 0.85)

    train = files[:train_end]
    validation = files[train_end:val_end]
    test = files[val_end:]

    return train, validation, test


_, _, fake_test = split_files(fake_files)
_, _, real_test = split_files(real_files)


test_recordings = (
    [(file, 1) for file in fake_test]
    +
    [(file, 0) for file in real_test]
)


random.shuffle(test_recordings)


print("========================================")
print("RECORDING-LEVEL TEST SET")
print("========================================")

print("Fake test recordings:", len(fake_test))
print("Real test recordings:", len(real_test))
print("Total recordings:", len(test_recordings))


# ============================================================
# CNN MODEL
# ============================================================

class VoiceCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(16),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(128, 64),

            nn.ReLU(),

            nn.Dropout(0.4),

            nn.Linear(64, 2)
        )


    def forward(self, x):

        x = self.features(x)

        return self.classifier(x)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = VoiceCNN().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print("Model loaded.")
print("Device:", device)


# ============================================================
# AUDIO → MEL
# ============================================================

def waveform_to_mel(waveform):

    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    mean = mel_db.mean()
    std = mel_db.std()

    mel_db = (
        mel_db - mean
    ) / (std + 1e-8)

    return mel_db.astype(np.float32)


# ============================================================
# CREATE WINDOWS
# ============================================================

def create_windows(waveform):

    windows = []

    if len(waveform) <= WINDOW_SIZE:

        padded = np.pad(
            waveform,
            (
                0,
                WINDOW_SIZE - len(waveform)
            ),
            mode="constant"
        )

        windows.append(padded)

        return windows


    start = 0

    while start + WINDOW_SIZE <= len(waveform):

        window = waveform[
            start:start + WINDOW_SIZE
        ]

        windows.append(window)

        start += HOP_SIZE


    if start < len(waveform):

        windows.append(
            waveform[-WINDOW_SIZE:]
        )


    return windows


# ============================================================
# PREDICT ONE RECORDING
# ============================================================

def predict_recording(audio_path):

    waveform, _ = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    waveform = waveform.astype(
        np.float32
    )

    windows = create_windows(
        waveform
    )

    fake_probabilities = []

    with torch.no_grad():

        for window in windows:

            mel = waveform_to_mel(
                window
            )

            mel = np.expand_dims(
                mel,
                axis=0
            )

            mel = np.expand_dims(
                mel,
                axis=0
            )

            tensor = torch.tensor(
                mel,
                dtype=torch.float32
            ).to(device)

            output = model(tensor)

            probabilities = torch.softmax(
                output,
                dim=1
            )

            fake_probability = float(
                probabilities[0, 1].item()
            )

            fake_probabilities.append(
                fake_probability
            )


    average_fake_probability = float(
        np.mean(fake_probabilities)
    )

    prediction = (
        1
        if average_fake_probability >= 0.50
        else 0
    )

    return (
        len(windows),
        average_fake_probability,
        prediction
    )


# ============================================================
# RECORDING-LEVEL EVALUATION
# ============================================================

true_labels = []
predicted_labels = []


print()
print("========================================")
print("RECORDING-LEVEL PREDICTIONS")
print("========================================")


for audio_path, true_label in test_recordings:

    (
        num_windows,
        fake_probability,
        prediction
    ) = predict_recording(
        audio_path
    )

    true_labels.append(
        true_label
    )

    predicted_labels.append(
        prediction
    )

    if prediction == 1:

        prediction_text = "FAKE"

    else:

        prediction_text = "REAL"


    actual_text = (
        "FAKE"
        if true_label == 1
        else "REAL"
    )


    result = (
        "CORRECT"
        if prediction == true_label
        else "WRONG"
    )


    print()
    print("File:", audio_path.name)
    print("Actual:", actual_text)
    print("Windows:", num_windows)
    print(
        f"Fake probability: "
        f"{fake_probability * 100:.2f}%"
    )
    print("Prediction:", prediction_text)
    print("Result:", result)


# ============================================================
# METRICS
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
# FINAL RESULTS
# ============================================================

print()
print("========================================")
print("CNN RECORDING-LEVEL RESULTS")
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