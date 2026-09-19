from io import BytesIO
import glob

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import onnxruntime as ort
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


SAMPLE_RATE = 16000

AASIST_WINDOW = 64600
CNN_WINDOW = 64000

N_MELS = 64

CNN_MODEL_PATH = (
    "models/cnn/voice_cnn_asvspoof2019_best.pth"
)

AASIST_MODEL_PATH = (
    "models/aasist/aasist.onnx"
)


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
        return self.classifier(
            self.features(x)
        )


if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


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


# ============================================================
# AASIST
# ============================================================

aasist = ort.InferenceSession(
    AASIST_MODEL_PATH,
    providers=[
        "CoreMLExecutionProvider",
        "CPUExecutionProvider"
    ]
)

aasist_input = aasist.get_inputs()[0].name


# ============================================================
# AUDIO
# ============================================================

def load_audio(path):

    try:
        with open(path, "rb") as f:
            data = f.read()

        waveform, sr = sf.read(
            BytesIO(data),
            dtype="float32"
        )

    except Exception as e:

        print(
            f"\nSKIPPING unreadable file: {path}"
        )

        print(
            f"Reason: {e}"
        )

        return None

    if waveform.ndim > 1:
        waveform = np.mean(
            waveform,
            axis=1
        )

    if sr != SAMPLE_RATE:
        waveform = librosa.resample(
            waveform,
            orig_sr=sr,
            target_sr=SAMPLE_RATE
        )

    return waveform.astype(np.float32)


# ============================================================
# AASIST
# ============================================================

def aasist_score(waveform):

    if len(waveform) >= AASIST_WINDOW:
        x = waveform[:AASIST_WINDOW]

    else:
        x = np.zeros(
            AASIST_WINDOW,
            dtype=np.float32
        )

        x[:len(waveform)] = waveform

    x = x.reshape(
        1, -1
    ).astype(np.float32)

    logits = aasist.run(
        None,
        {
            aasist_input: x
        }
    )[0][0]

    p = np.exp(
        logits - np.max(logits)
    )

    p = p / p.sum()

    return float(p[1])


# ============================================================
# CNN
# ============================================================

@torch.no_grad()
def cnn_score(waveform):

    if len(waveform) >= CNN_WINDOW:
        x = waveform[:CNN_WINDOW]

    else:
        x = np.zeros(
            CNN_WINDOW,
            dtype=np.float32
        )

        x[:len(waveform)] = waveform

    mel = librosa.feature.melspectrogram(
        y=x,
        sr=SAMPLE_RATE,
        n_fft=1024,
        hop_length=256,
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

    logits = cnn(
        tensor.to(DEVICE)
    )

    probs = torch.softmax(
        logits,
        dim=1
    )

    return float(
        probs[0, 1].item()
    )


# ============================================================
# COLLECT SCORES
# ============================================================

fake_files = sorted(
    glob.glob("data/fake_wav/*.wav")
)

real_files = sorted(
    glob.glob("data/real_wav/*.wav")
)

print("Fake recordings:", len(fake_files))
print("Real recordings:", len(real_files))


targets = []
aasist_scores = []
cnn_scores = []


def process_file(path, label):

    waveform = load_audio(path)

    if waveform is None:
        return None

    a = aasist_score(waveform)
    c = cnn_score(waveform)

    targets.append(label)
    aasist_scores.append(a)
    cnn_scores.append(c)

    return a, c


print("\nProcessing files...\n")
for path in real_files:

    result = process_file(
        path,
        0
    )

    if result is None:
        continue

    a, c = result

    print(
        f"REAL  {path.split('/')[-1][:35]:35s} "
        f"AASIST={a*100:6.2f}% "
        f"CNN={c*100:6.2f}%"
    )


for path in fake_files:

    result = process_file(
        path,
        1
    )

    if result is None:
        continue

    a, c = result

    print(
        f"FAKE  {path.split('/')[-1][:35]:35s} "
        f"AASIST={a*100:6.2f}% "
        f"CNN={c*100:6.2f}%"
    )


# ============================================================
# EVALUATE THREE DETECTORS
# ============================================================

def evaluate(name, scores):

    predictions = [
        1 if s >= 0.5 else 0
        for s in scores
    ]

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

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print(
        f"Accuracy : {accuracy*100:.2f}%"
    )

    print(
        f"Precision: {precision*100:.2f}%"
    )

    print(
        f"Recall   : {recall*100:.2f}%"
    )

    print(
        f"F1 Score : {f1*100:.2f}%"
    )

    print("\nConfusion matrix:")

    print(
        confusion_matrix(
            targets,
            predictions
        )
    )


fusion_scores = [
    0.5 * a + 0.5 * c
    for a, c in zip(
        aasist_scores,
        cnn_scores
    )
]


evaluate(
    "AASIST",
    aasist_scores
)

evaluate(
    "CNN",
    cnn_scores
)

evaluate(
    "50/50 FUSION",
    fusion_scores
)
