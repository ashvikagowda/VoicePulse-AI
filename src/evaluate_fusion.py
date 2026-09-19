from io import BytesIO
import random

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import onnxruntime as ort
from datasets import Audio, load_dataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# CONFIG
# ============================================================

SEED = 42

SAMPLE_RATE = 16000

AASIST_WINDOW = 64600
CNN_WINDOW = 64000

N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256

CNN_MODEL_PATH = (
    "models/cnn/voice_cnn_asvspoof2019_best.pth"
)

AASIST_MODEL_PATH = (
    "models/aasist/aasist.onnx"
)

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
# LOAD CNN
# ============================================================

print("\nLoading CNN...")

cnn = VoiceCNN().to(DEVICE)

checkpoint = torch.load(
    CNN_MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

cnn.load_state_dict(
    checkpoint["model_state_dict"]
)

cnn.eval()

print("CNN loaded")


# ============================================================
# LOAD AASIST
# ============================================================

print("Loading AASIST...")

aasist_session = ort.InferenceSession(
    AASIST_MODEL_PATH,
    providers=[
        "CoreMLExecutionProvider",
        "CPUExecutionProvider"
    ]
)

aasist_input_name = (
    aasist_session.get_inputs()[0].name
)

print(
    "AASIST providers:",
    aasist_session.get_providers()
)


# ============================================================
# AUDIO
# ============================================================

def decode_audio(audio_bytes):

    waveform, sample_rate = sf.read(
        BytesIO(audio_bytes),
        dtype="float32"
    )

    if waveform.ndim > 1:
        waveform = np.mean(
            waveform,
            axis=1
        )

    if sample_rate != SAMPLE_RATE:

        waveform = librosa.resample(
            waveform,
            orig_sr=sample_rate,
            target_sr=SAMPLE_RATE
        )

    return waveform.astype(np.float32)


# ============================================================
# CNN PREPROCESSING
# ============================================================

def prepare_cnn(waveform):

    if len(waveform) >= CNN_WINDOW:

        start = (
            len(waveform) - CNN_WINDOW
        ) // 2

        waveform = waveform[
            start:start + CNN_WINDOW
        ]

    else:

        padded = np.zeros(
            CNN_WINDOW,
            dtype=np.float32
        )

        padded[:len(waveform)] = waveform
        waveform = padded

    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    mel_db = (
        mel_db - mel_db.mean()
    ) / (
        mel_db.std() + 1e-8
    )

    return torch.tensor(
        mel_db,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)


# ============================================================
# AASIST PREPROCESSING
# ============================================================
def prepare_aasist(waveform):

    # Match the original AASIST evaluation:
    # use the FIRST 64600 samples.

    if len(waveform) >= AASIST_WINDOW:

        waveform = waveform[:AASIST_WINDOW]

    else:

        padded = np.zeros(
            AASIST_WINDOW,
            dtype=np.float32
        )

        padded[:len(waveform)] = waveform
        waveform = padded

    return waveform.reshape(
        1, -1
    ).astype(np.float32)


# ============================================================
# PREDICT BOTH
# ============================================================

@torch.no_grad()
def predict_models(audio_bytes):

    waveform = decode_audio(audio_bytes)

    # -------------------------
    # CNN
    # -------------------------

    cnn_input = prepare_cnn(
        waveform
    ).to(DEVICE)

    cnn_logits = cnn(cnn_input)

    cnn_probs = torch.softmax(
        cnn_logits,
        dim=1
    )

    cnn_fake = cnn_probs[0, 1].item()

    # -------------------------
    # AASIST
    # -------------------------

    aasist_input = prepare_aasist(
        waveform
    )

    output = aasist_session.run(
        None,
        {
            aasist_input_name:
            aasist_input
        }
    )[0][0]

    exp_output = np.exp(
        output - np.max(output)
    )

    aasist_probs = (
        exp_output /
        exp_output.sum()
    )

    aasist_fake = float(
        aasist_probs[0]
    )

    return aasist_fake, cnn_fake


# ============================================================
# METRICS
# ============================================================

def metrics(y_true, scores, threshold=0.5):

    y_pred = [
        1 if score >= threshold else 0
        for score in scores
    ]

    return (
        accuracy_score(
            y_true,
            y_pred
        ),
        precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        f1_score(
            y_true,
            y_pred,
            zero_division=0
        )
    )


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
# GET BALANCED SAMPLE
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
# RUN EVALUATION
# ============================================================

targets = []

aasist_scores = []
cnn_scores = []

print(
    "\nEvaluating "
    f"{len(selected_indices)} recordings..."
)

for count, index in enumerate(
    selected_indices,
    start=1
):

    sample = ds[int(index)]

    target = int(
        sample["label"]
    )

    aasist_fake, cnn_fake = (
        predict_models(
            sample["audio"]["bytes"]
        )
    )

    targets.append(target)

    aasist_scores.append(
        aasist_fake
    )

    cnn_scores.append(
        cnn_fake
    )

    if count % 100 == 0:

        print(
            f"Processed {count}/"
            f"{len(selected_indices)}"
        )


# ============================================================
# EVALUATE DIFFERENT FUSION WEIGHTS
# ============================================================

weights = [
    (1.00, 0.00),
    (0.90, 0.10),
    (0.80, 0.20),
    (0.70, 0.30),
    (0.60, 0.40),
    (0.50, 0.50),
    (0.40, 0.60),
    (0.30, 0.70),
    (0.20, 0.80),
    (0.10, 0.90),
    (0.00, 1.00),
]


print("\n")
print("=" * 78)
print("AASIST + CNN FUSION RESULTS")
print("=" * 78)

print(
    f"{'AASIST':>10} "
    f"{'CNN':>10} "
    f"{'Accuracy':>12} "
    f"{'Precision':>12} "
    f"{'Recall':>12} "
    f"{'F1':>12}"
)

print("-" * 78)

results = []

for aasist_weight, cnn_weight in weights:

    fusion_scores = [
        (
            aasist_weight * a +
            cnn_weight * c
        )
        for a, c in zip(
            aasist_scores,
            cnn_scores
        )
    ]

    acc, prec, rec, f1 = metrics(
        targets,
        fusion_scores
    )

    results.append(
        (
            f1,
            aasist_weight,
            cnn_weight,
            acc,
            prec,
            rec
        )
    )

    print(
        f"{aasist_weight:>9.0%} "
        f"{cnn_weight:>9.0%} "
        f"{acc * 100:>11.2f}% "
        f"{prec * 100:>11.2f}% "
        f"{rec * 100:>11.2f}% "
        f"{f1 * 100:>11.2f}%"
    )


# ============================================================
# BEST FUSION
# ============================================================

best = max(
    results,
    key=lambda x: x[0]
)

best_f1, best_aasist, best_cnn, best_acc, best_prec, best_rec = best

print("\n")
print("=" * 78)
print("BEST RESULT")
print("=" * 78)

print(
    f"AASIST weight : {best_aasist:.0%}"
)

print(
    f"CNN weight    : {best_cnn:.0%}"
)

print(
    f"Accuracy      : {best_acc * 100:.2f}%"
)

print(
    f"Precision     : {best_prec * 100:.2f}%"
)

print(
    f"Recall        : {best_rec * 100:.2f}%"
)

print(
    f"F1 Score      : {best_f1 * 100:.2f}%"
)

print("=" * 78)
