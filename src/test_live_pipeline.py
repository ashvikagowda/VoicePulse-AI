from pathlib import Path

import librosa
import numpy as np
import sounddevice as sd
import torch
import torch.nn as nn


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "cnn"
    / "voicepulse_ai_cnn.pth"
)

SAMPLE_RATE = 16000
MIC_SAMPLE_RATE = 48000

RECORD_SECONDS = 6

WINDOW_SECONDS = 4
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SECONDS

N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# ============================================================
# MODEL
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

            nn.ReLU()
        )

        self.global_pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.classifier = nn.Sequential(

            nn.Dropout(0.4),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.Dropout(0.4),

            nn.Linear(
                64,
                2
            )
        )


    def forward(self, x):

        x = self.features(x)

        x = self.global_pool(x)

        x = torch.flatten(
            x,
            1
        )

        x = self.classifier(x)

        return x


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading CNN...")

model = VoiceCNN()

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        state_dict = checkpoint

else:

    state_dict = checkpoint


model.load_state_dict(
    state_dict
)

model = model.to(
    DEVICE
)

model.eval()

print("CNN loaded successfully.")


# ============================================================
# RECORD USING THE EXACT WORKING INPUTSTREAM SETTINGS
# ============================================================

print("\n----------------------------------------")
print("MICROPHONE TEST")
print("----------------------------------------")

print("Device: 0")
print("Sample rate: 48000 Hz")
print("Channels: 1")
print("Recording:", RECORD_SECONDS, "seconds")

print("\nSpeak normally after recording starts.")

    int(
        RECORD_SECONDS
        * MIC_SAMPLE_RATE
    ),
    samplerate=MIC_SAMPLE_RATE,
    channels=1,
    dtype="float32",
    device=2
)recording = sd.rec(
    int(
        RECORD_SECONDS
        * MIC_SAMPLE_RATE
    ),
    samplerate=MIC_SAMPLE_RATE,
    channels=1,
    dtype="float32",
    device=2
)

sd.wait()

print("\nRecording completed.")


# ============================================================
# CONVERT TO MONO
# ============================================================

audio = recording[:, 0].astype(
    np.float32
)


print(
    "Recorded samples:",
    len(audio)
)

print(
    "Maximum amplitude:",
    float(np.max(np.abs(audio)))
)

print(
    "RMS:",
    float(np.sqrt(np.mean(audio ** 2)))
)


# ============================================================
# RESAMPLE 48 kHz → 16 kHz
# ============================================================

print("\nResampling 48000 Hz → 16000 Hz...")

audio = librosa.resample(
    audio,
    orig_sr=MIC_SAMPLE_RATE,
    target_sr=SAMPLE_RATE
)

audio = np.asarray(
    audio,
    dtype=np.float32
)


print(
    "Resampled samples:",
    len(audio)
)


# ============================================================
# TAKE LAST 4 SECONDS
# ============================================================

if len(audio) < WINDOW_SAMPLES:

    audio = np.pad(
        audio,
        (
            0,
            WINDOW_SAMPLES - len(audio)
        )
    )

else:

    audio = audio[
        -WINDOW_SAMPLES:
    ]


# ============================================================
# SAME PREPROCESSING AS DASHBOARD
# ============================================================

mel = librosa.feature.melspectrogram(
    y=audio,
    sr=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS
)

mel = librosa.power_to_db(
    mel,
    ref=np.max
)

mean = np.mean(mel)

std = np.std(mel)

if std > 1e-8:

    mel = (
        mel - mean
    ) / std

else:

    mel = (
        mel - mean
    )


input_tensor = torch.tensor(
    mel,
    dtype=torch.float32
).unsqueeze(0).unsqueeze(0)


input_tensor = input_tensor.to(
    DEVICE
)


# ============================================================
# CNN PREDICTION
# ============================================================

print("\nRunning CNN prediction...")

with torch.no_grad():

    logits = model(
        input_tensor
    )

    probabilities = torch.softmax(
        logits,
        dim=1
    )


real_probability = float(
    probabilities[0, 0].item()
)

fake_probability = float(
    probabilities[0, 1].item()
)


# ============================================================
# RESULT
# ============================================================

if fake_probability >= 0.5:

    prediction = "AI-GENERATED"

else:

    prediction = "REAL"


print("\n========================================")
print("VOICEGUARD RESULT")
print("========================================")

print(
    f"Real probability: "
    f"{real_probability * 100:.2f}%"
)

print(
    f"AI-generated probability: "
    f"{fake_probability * 100:.2f}%"
)

print(
    f"Prediction: {prediction}"
)

print("========================================\n")
