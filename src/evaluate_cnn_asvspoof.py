from io import BytesIO
import random

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from datasets import Audio, load_dataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# CONFIG
# ============================================================

SEED = 42

SAMPLE_RATE = 16000

WINDOW_SECONDS = 4
WINDOW_SIZE = SAMPLE_RATE * WINDOW_SECONDS

N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256

MODEL_PATH = "models/cnn/voice_cnn_asvspoof2019_best.pth"

REAL_SAMPLES = 500
FAKE_SAMPLES = 500


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("Device:", DEVICE)


# ============================================================
# CNN MODEL
# ============================================================

class VoiceCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1, 16,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                16, 32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                32, 64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                64, 128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
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

        x = self.classifier(x)

        return x


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading CNN...")

model = VoiceCNN().to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Model loaded:", MODEL_PATH)

print(
    "Saved validation F1:",
    f'{checkpoint["val_f1"] * 100:.2f}%'
)


# ============================================================
# AUDIO -> LOG-MEL
# ============================================================

def audio_to_mel(audio_bytes):

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

    # Resample
    if sample_rate != SAMPLE_RATE:

        waveform = librosa.resample(
            waveform,
            orig_sr=sample_rate,
            target_sr=SAMPLE_RATE
        )

    waveform = waveform.astype(
        np.float32
    )

    # Center crop / zero pad
    if len(waveform) >= WINDOW_SIZE:

        start = (
            len(waveform) - WINDOW_SIZE
        ) // 2

        waveform = waveform[
            start:start + WINDOW_SIZE
        ]

    else:

        padded = np.zeros(
            WINDOW_SIZE,
            dtype=np.float32
        )

        padded[:len(waveform)] = waveform

        waveform = padded

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0
    )

    # Log scale
    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # Same normalization used during training
    mel_db = (
        mel_db - mel_db.mean()
    ) / (
        mel_db.std() + 1e-8
    )

    # [1, 64, 251]
    tensor = torch.tensor(
        mel_db,
        dtype=torch.float32
    ).unsqueeze(0)

    return tensor


# ============================================================
# PREDICTION
# ============================================================

@torch.no_grad()
def predict_sample(sample):

    audio_bytes = sample["audio"]["bytes"]

    mel = audio_to_mel(
        audio_bytes
    )

    mel = mel.unsqueeze(0)

    mel = mel.to(DEVICE)

    logits = model(mel)

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    fake_probability = (
        probabilities[0, 1]
        .item()
    )

    prediction = (
        1 if fake_probability >= 0.5
        else 0
    )

    return prediction, fake_probability


# ============================================================
# LOAD ASVSPOOF2021-DF
# ============================================================

print("\nLoading ASVspoof2021-DF...")

ds = load_dataset(
    "SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF",
    split="test"
)

ds = ds.cast_column(
    "audio",
    Audio(decode=False)
)

print(
    "Total recordings:",
    len(ds)
)


# ============================================================
# GET REAL / FAKE INDICES
# ============================================================

real_indices = []
fake_indices = []

for i, label in enumerate(ds["label"]):

    if label == 0:
        real_indices.append(i)

    elif label == 1:
        fake_indices.append(i)

print(
    "Available real:",
    len(real_indices)
)

print(
    "Available fake:",
    len(fake_indices)
)


# ============================================================
# BALANCED TEST SET
# ============================================================

rng = np.random.default_rng(SEED)

selected_real = rng.choice(
    real_indices,
    size=REAL_SAMPLES,
    replace=False
)

selected_fake = rng.choice(
    fake_indices,
    size=FAKE_SAMPLES,
    replace=False
)

selected_indices = np.concatenate([
    selected_real,
    selected_fake
])

rng.shuffle(selected_indices)


# ============================================================
# EVALUATION
# ============================================================

targets = []
predictions = []
fake_probabilities = []

print("\nStarting external evaluation...\n")

for count, index in enumerate(
    selected_indices,
    start=1
):

    sample = ds[int(index)]

    true_label = int(
        sample["label"]
    )

    prediction, fake_probability = (
        predict_sample(sample)
    )

    targets.append(
        true_label
    )

    predictions.append(
        prediction
    )

    fake_probabilities.append(
        fake_probability
    )

    if count % 100 == 0:

        print(
            f"Processed {count}/"
            f"{len(selected_indices)}"
        )


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    targets,
    predictions
)

precision = precision_score(
    targets,
    predictions,
    zero_division=0
)

recall = recall_score(
    targets,
    predictions,
    zero_division=0
)

f1 = f1_score(
    targets,
    predictions,
    zero_division=0
)

cm = confusion_matrix(
    targets,
    predictions
)


# ============================================================
# RESULTS
# ============================================================

print("\n====================================")
print("ASVSPOOF2021-DF EXTERNAL TEST")
print("====================================")

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

print("\nConfusion Matrix:")

print(cm)

print("\n====================================")
print("INTERPRETATION")
print("====================================")

print(
    "0 = bonafide / real"
)

print(
    "1 = spoof / fake"
)
