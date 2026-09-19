import time
from collections import deque

import librosa
import numpy as np
import sounddevice as sd
import torch
import torch.nn as nn


# ============================================================
# CONFIG
# ============================================================

SAMPLE_RATE = 16000

WINDOW_SECONDS = 4
UPDATE_SECONDS = 2

WINDOW_SIZE = SAMPLE_RATE * WINDOW_SECONDS
UPDATE_SIZE = SAMPLE_RATE * UPDATE_SECONDS

N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256

MODEL_PATH = (
    "models/cnn/voice_cnn_local_finetuned_best.pth"
)

# Keep the latest 3 predictions.
# 3 x 2-second updates = about 6 seconds of history.
HISTORY_SIZE = 3


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print("Device:", DEVICE)


# ============================================================
# CNN
# ============================================================

class VoiceCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
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

        return self.classifier(x)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading fine-tuned CNN...")

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

print("Model loaded:")
print(MODEL_PATH)

if "val_f1" in checkpoint:
    print(
        "Saved validation F1:",
        f'{checkpoint["val_f1"] * 100:.2f}%'
    )


# ============================================================
# PREPROCESSING
# ============================================================

def prepare_audio(waveform):

    # Exactly 4 seconds
    if len(waveform) >= WINDOW_SIZE:

        waveform = waveform[:WINDOW_SIZE]

    else:

        padded = np.zeros(
            WINDOW_SIZE,
            dtype=np.float32
        )

        padded[:len(waveform)] = waveform

        waveform = padded

    # Same Mel preprocessing used during training
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0
    )

    mel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    mel = (
        mel - mel.mean()
    ) / (
        mel.std() + 1e-8
    )

    tensor = torch.tensor(
        mel,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)

    return tensor


# ============================================================
# PREDICTION
# ============================================================

@torch.no_grad()
def predict_fake_probability(waveform):

    x = prepare_audio(
        waveform
    ).to(DEVICE)

    logits = model(x)

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    # Class 1 = fake
    fake_probability = float(
        probabilities[0, 1].item()
    )

    return fake_probability


# ============================================================
# RISK CLASSIFICATION
# ============================================================

def classify_risk(smoothed_score):

    if smoothed_score >= 0.75:

        return (
            "HIGH",
            "POSSIBLE AI VOICE"
        )

    elif smoothed_score >= 0.50:

        return (
            "MEDIUM",
            "SUSPICIOUS"
        )

    else:

        return (
            "LOW",
            "LIKELY REAL"
        )


# ============================================================
# MICROPHONE
# ============================================================

print("\nAvailable audio devices:")
print(sd.query_devices())

print("\n======================================")
print("        VOICEGUARD LIVE DETECTOR")
print("======================================")

print(
    f"Analysis window : {WINDOW_SECONDS} sec"
)

print(
    f"Update interval : {UPDATE_SECONDS} sec"
)

print(
    f"History windows : {HISTORY_SIZE}"
)

print("\nSpeak into the microphone.")
print("Press Ctrl+C to stop.\n")


audio_buffer = deque(
    maxlen=WINDOW_SIZE
)

score_history = deque(
    maxlen=HISTORY_SIZE
)


first_message = True

try:

    while True:

        # ----------------------------------------------------
        # Capture next 2 seconds
        # ----------------------------------------------------

        audio = sd.rec(
            UPDATE_SIZE,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32"
        )

        sd.wait()

        audio = audio[:, 0]

        audio_buffer.extend(audio)

        # ----------------------------------------------------
        # Wait until first 4-second window exists
        # ----------------------------------------------------

        if len(audio_buffer) < WINDOW_SIZE:

            collected = (
                len(audio_buffer)
                / SAMPLE_RATE
            )

            print(
                f"\rCollecting audio: "
                f"{collected:.1f}/"
                f"{WINDOW_SECONDS:.1f} sec",
                end=""
            )

            continue

        # ----------------------------------------------------
        # Current 4-second window
        # ----------------------------------------------------

        waveform = np.asarray(
            audio_buffer,
            dtype=np.float32
        )

        start_time = time.time()

        fake_score = predict_fake_probability(
            waveform
        )

        processing_time = (
            time.time() - start_time
        )

        # ----------------------------------------------------
        # Temporal smoothing
        # ----------------------------------------------------

        score_history.append(
            fake_score
        )

        smoothed_score = float(
            np.mean(score_history)
        )

        risk, decision = classify_risk(
            smoothed_score
        )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        print("\n")
        print("=" * 58)

        print(
            time.strftime("%H:%M:%S"),
            "| VoiceGuard analysis"
        )

        print("-" * 58)

        print(
            f"Current fake probability : "
            f"{fake_score * 100:6.2f}%"
        )

        print(
            f"Smoothed risk score      : "
            f"{smoothed_score * 100:6.2f}%"
        )

        print(
            f"Risk                     : {risk}"
        )

        print(
            f"Decision                 : {decision}"
        )

        print(
            f"Processing time          : "
            f"{processing_time:.2f} sec"
        )

        if processing_time < UPDATE_SECONDS:

            print(
                "Real-time status         : ✓"
            )

        else:

            print(
                "Real-time status         : "
                "SLOW"
            )

        print("=" * 58)

except KeyboardInterrupt:

    print("\n\nVoiceGuard stopped.")
