from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import glob
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn


SAMPLE_RATE = 16000
WINDOW_SIZE = SAMPLE_RATE * 4
HOP_SIZE = SAMPLE_RATE * 2

N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256

MODEL_PATH = (
    "models/cnn/voice_cnn_local_finetuned_best.pth"
)


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print("Device:", DEVICE)


# ============================================================
# MODEL
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
        return self.classifier(
            self.features(x)
        )


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

print("Fine-tuned model loaded.")


# ============================================================
# AUDIO
# ============================================================

def load_audio(path):

    waveform, sr = librosa.load(
        path,
        sr=SAMPLE_RATE,
        mono=True
    )

    return waveform.astype(np.float32)


# ============================================================
# WINDOW PREDICTION
# ============================================================

@torch.no_grad()
def predict_window(waveform):

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

    x = torch.tensor(
        mel,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)

    logits = model(
        x.to(DEVICE)
    )

    probs = torch.softmax(
        logits,
        dim=1
    )

    # class 1 = fake
    return float(
        probs[0, 1].item()
    )


# ============================================================
# RECORDING PREDICTION
# ============================================================

def predict_recording(path):

    try:
        waveform = load_audio(path)

    except Exception as e:
        print(
            f"SKIP {path}: {e}"
        )
        return None

    if len(waveform) < WINDOW_SIZE:

        padded = np.zeros(
            WINDOW_SIZE,
            dtype=np.float32
        )

        padded[:len(waveform)] = waveform
        waveform = padded

    scores = []

    starts = range(
        0,
        max(
            1,
            len(waveform) - WINDOW_SIZE + 1
        ),
        HOP_SIZE
    )

    for start in starts:

        window = waveform[
            start:start + WINDOW_SIZE
        ]

        if len(window) < WINDOW_SIZE:
            padded = np.zeros(
                WINDOW_SIZE,
                dtype=np.float32
            )
            padded[:len(window)] = window
            window = padded

        scores.append(
            predict_window(window)
        )

    # Include final window when it is not already covered.
    last_start = max(
        0,
        len(waveform) - WINDOW_SIZE
    )

    if last_start not in list(starts):

        window = waveform[
            last_start:last_start + WINDOW_SIZE
        ]

        if len(window) < WINDOW_SIZE:

            padded = np.zeros(
                WINDOW_SIZE,
                dtype=np.float32
            )

            padded[:len(window)] = window
            window = padded

        scores.append(
            predict_window(window)
        )

    avg_score = float(
        np.mean(scores)
    )

    prediction = (
        1 if avg_score >= 0.5
        else 0
    )

    return prediction, avg_score, len(scores)


# ============================================================
# REPRODUCE THE SAME RECORDING SPLIT
# ============================================================

# The fine-tuning script uses seed 42 and separately shuffles
# fake and real lists before creating their 70/15/15 split.

import random

SEED = 42
random.seed(SEED)

fake_files = sorted(
    glob.glob(str(PROJECT_ROOT / "data" / "fake_wav" / "*.wav")))

real_files = sorted(
    glob.glob(str(PROJECT_ROOT / "data" / "real_wav" / "*.wav")))


def split_files(files):

    files = files.copy()

    random.shuffle(files)

    n = len(files)

    n_test = max(
        1,
        round(n * 0.15)
    )

    n_val = max(
        1,
        round(n * 0.15)
    )

    test = files[:n_test]

    val = files[
        n_test:n_test + n_val
    ]

    train = files[
        n_test + n_val:
    ]

    return train, val, test


_, _, fake_test = split_files(fake_files)
_, _, real_test = split_files(real_files)

test_files = (
    [(p, 1) for p in fake_test] +
    [(p, 0) for p in real_test]
)

random.shuffle(test_files)


# ============================================================
# RECORDING-LEVEL TEST
# ============================================================

print("\n====================================")
print("RECORDING-LEVEL HELD-OUT TEST")
print("====================================")

print(
    "Test recordings:",
    len(test_files)
)

targets = []
predictions = []

for path, label in test_files:

    result = predict_recording(path)

    if result is None:
        continue

    prediction, score, n_windows = result

    targets.append(label)
    predictions.append(prediction)

    name = Path(path).name

    print(
        f"{'FAKE' if label else 'REAL':4s} "
        f"{name[:38]:38s} "
        f"score={score * 100:6.2f}% "
        f"prediction={'FAKE' if prediction else 'REAL':4s} "
        f"windows={n_windows}"
    )


# ============================================================
# METRICS
# ============================================================

tp = tn = fp = fn = 0

for target, prediction in zip(
    targets,
    predictions
):

    if target == 1 and prediction == 1:
        tp += 1

    elif target == 0 and prediction == 0:
        tn += 1

    elif target == 0 and prediction == 1:
        fp += 1

    elif target == 1 and prediction == 0:
        fn += 1


total = tp + tn + fp + fn

accuracy = (
    tp + tn
) / max(total, 1)

precision = tp / max(
    tp + fp,
    1
)

recall = tp / max(
    tp + fn,
    1
)

f1 = (
    2 * precision * recall
) / max(
    precision + recall,
    1e-8
)


print("\n====================================")
print("FINAL RECORDING-LEVEL RESULTS")
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
print(
    f"[[{tn:3d} {fp:3d}]"
)
print(
    f" [{fn:3d} {tp:3d}]]"
)
