from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import glob
import random

import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIG
# ============================================================

SEED = 42

SAMPLE_RATE = 16000

WINDOW_SECONDS = 4
WINDOW_SIZE = SAMPLE_RATE * WINDOW_SECONDS
HOP_SIZE = SAMPLE_RATE * 2

N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256

BATCH_SIZE = 16

LEARNING_RATE = 1e-4
EPOCHS = 15

PRETRAINED_MODEL = (
    "models/cnn/voice_cnn_asvspoof2019_best.pth"
)

OUTPUT_MODEL = (
    "models/cnn/voice_cnn_local_finetuned_best.pth"
)


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
        return self.classifier(
            self.features(x)
        )


# ============================================================
# LOAD PRETRAINED MODEL
# ============================================================

print("\nLoading ASVspoof-trained CNN...")

model = VoiceCNN().to(DEVICE)

checkpoint = torch.load(
    PRETRAINED_MODEL,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print("Pretrained model loaded.")


# ============================================================
# LOCAL RECORDINGS
# ============================================================

fake_files = sorted(
    glob.glob(str(PROJECT_ROOT / "data" / "fake_wav" / "*.wav")))

real_files = sorted(
    glob.glob(str(PROJECT_ROOT / "data" / "real_wav" / "*.wav")))

print("\nLocal dataset:")
print("Fake recordings:", len(fake_files))
print("Real recordings:", len(real_files))


# ============================================================
# RECORDING-LEVEL SPLIT
# ============================================================

def split_files(files):

    files = files.copy()

    random.shuffle(files)

    n = len(files)

    n_test = max(1, round(n * 0.15))
    n_val = max(1, round(n * 0.15))

    test = files[:n_test]

    val = files[
        n_test:n_test + n_val
    ]

    train = files[
        n_test + n_val:
    ]

    return train, val, test


fake_train, fake_val, fake_test = split_files(
    fake_files
)

real_train, real_val, real_test = split_files(
    real_files
)


train_files = (
    [(p, 1) for p in fake_train] +
    [(p, 0) for p in real_train]
)

val_files = (
    [(p, 1) for p in fake_val] +
    [(p, 0) for p in real_val]
)

test_files = (
    [(p, 1) for p in fake_test] +
    [(p, 0) for p in real_test]
)

random.shuffle(train_files)
random.shuffle(val_files)
random.shuffle(test_files)


print("\nRecording-level split:")

print(
    "Train:",
    len(train_files),
    "(fake:",
    len(fake_train),
    "real:",
    len(real_train),
    ")"
)

print(
    "Validation:",
    len(val_files),
    "(fake:",
    len(fake_val),
    "real:",
    len(real_val),
    ")"
)

print(
    "Test:",
    len(test_files),
    "(fake:",
    len(fake_test),
    "real:",
    len(real_test),
    ")"
)


# ============================================================
# AUDIO LOADER
# ============================================================

def load_audio(path):

    waveform, sr = librosa.load(
        path,
        sr=SAMPLE_RATE,
        mono=True
    )

    return waveform.astype(
        np.float32
    )


# ============================================================
# MEL EXTRACTION
# ============================================================

def audio_to_mel(waveform):

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

    return mel.astype(
        np.float32
    )


# ============================================================
# DATASET
# ============================================================

class LocalVoiceDataset(Dataset):

    def __init__(
        self,
        files,
        training=False
    ):
        self.files = files
        self.training = training

        self.samples = []

        for path, label in files:

            try:
                waveform = load_audio(path)

            except Exception as e:

                print(
                    "Skipping:",
                    path,
                    e
                )

                continue

            if len(waveform) < WINDOW_SIZE:

                self.samples.append(
                    (
                        path,
                        label,
                        0
                    )
                )

            else:

                starts = list(
                    range(
                        0,
                        len(waveform)
                        - WINDOW_SIZE
                        + 1,
                        HOP_SIZE
                    )
                )

                if not starts:
                    starts = [0]

                for start in starts:

                    self.samples.append(
                        (
                            path,
                            label,
                            start
                        )
                    )

        print(
            "Generated windows:",
            len(self.samples)
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        path, label, start = (
            self.samples[index]
        )

        waveform = load_audio(path)

        # ----------------------------------------------------
        # Select window
        # ----------------------------------------------------

        if len(waveform) >= WINDOW_SIZE:

            if self.training:

                max_start = (
                    len(waveform)
                    - WINDOW_SIZE
                )

                start = random.randint(
                    0,
                    max_start
                )

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

        # ----------------------------------------------------
        # Small augmentation during training
        # ----------------------------------------------------

        if self.training:

            gain = np.random.uniform(
                0.9,
                1.1
            )

            waveform = waveform * gain

        mel = audio_to_mel(
            waveform
        )

        tensor = torch.tensor(
            mel,
            dtype=torch.float32
        ).unsqueeze(0)

        return (
            tensor,
            torch.tensor(
                label,
                dtype=torch.long
            )
        )


# ============================================================
# CREATE DATASETS
# ============================================================

print("\nCreating training dataset...")

train_dataset = LocalVoiceDataset(
    train_files,
    training=True
)

print("\nCreating validation dataset...")

val_dataset = LocalVoiceDataset(
    val_files,
    training=False
)

print("\nCreating test dataset...")

test_dataset = LocalVoiceDataset(
    test_files,
    training=False
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    targets,
    predictions
):

    tp = tn = fp = fn = 0

    for t, p in zip(
        targets,
        predictions
    ):

        if t == 1 and p == 1:
            tp += 1

        elif t == 0 and p == 0:
            tn += 1

        elif t == 0 and p == 1:
            fp += 1

        elif t == 1 and p == 0:
            fn += 1

    accuracy = (
        tp + tn
    ) / max(
        tp + tn + fp + fn,
        1
    )

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
        / max(
            precision + recall,
            1e-8
        )
    )

    return (
        accuracy,
        precision,
        recall,
        f1
    )


# ============================================================
# TRAIN
# ============================================================

def train_epoch(
    model,
    loader,
    optimizer,
    criterion
):

    model.train()

    total_loss = 0

    targets = []
    predictions = []

    for x, y in loader:

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        logits = model(x)

        loss = criterion(
            logits,
            y
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        pred = torch.argmax(
            logits,
            dim=1
        )

        targets.extend(
            y.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        predictions.extend(
            pred.detach()
            .cpu()
            .numpy()
            .tolist()
        )

    return (
        total_loss / len(loader),
        *calculate_metrics(
            targets,
            predictions
        )
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0

    targets = []
    predictions = []

    for x, y in loader:

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x)

        loss = criterion(
            logits,
            y
        )

        total_loss += loss.item()

        pred = torch.argmax(
            logits,
            dim=1
        )

        targets.extend(
            y.cpu()
            .numpy()
            .tolist()
        )

        predictions.extend(
            pred.cpu()
            .numpy()
            .tolist()
        )

    return (
        total_loss / len(loader),
        *calculate_metrics(
            targets,
            predictions
        )
    )


# ============================================================
# OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# ============================================================
# FINE-TUNING
# ============================================================

best_val_f1 = 0.0

print("\nStarting local fine-tuning...\n")

for epoch in range(
    1,
    EPOCHS + 1
):

    print(
        f"========== EPOCH "
        f"{epoch}/{EPOCHS} =========="
    )

    (
        train_loss,
        train_acc,
        train_prec,
        train_rec,
        train_f1
    ) = train_epoch(
        model,
        train_loader,
        optimizer,
        criterion
    )

    (
        val_loss,
        val_acc,
        val_prec,
        val_rec,
        val_f1
    ) = evaluate(
        model,
        val_loader,
        criterion
    )

    print(
        f"\nTrain Loss : {train_loss:.4f}"
    )

    print(
        f"Train Acc  : "
        f"{train_acc * 100:.2f}%"
    )

    print(
        f"Train F1   : "
        f"{train_f1 * 100:.2f}%"
    )

    print(
        f"\nVal Loss   : {val_loss:.4f}"
    )

    print(
        f"Val Acc    : "
        f"{val_acc * 100:.2f}%"
    )

    print(
        f"Val Prec   : "
        f"{val_prec * 100:.2f}%"
    )

    print(
        f"Val Recall : "
        f"{val_rec * 100:.2f}%"
    )

    print(
        f"Val F1     : "
        f"{val_f1 * 100:.2f}%"
    )

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "val_f1":
                    val_f1,

                "val_accuracy":
                    val_acc,

                "val_precision":
                    val_prec,

                "val_recall":
                    val_rec,
            },
            OUTPUT_MODEL
        )

        print(
            "\n*** BEST LOCAL MODEL SAVED ***"
        )

    print()


# ============================================================
# LOAD BEST MODEL
# ============================================================

print(
    "\nLoading best local model..."
)

best_checkpoint = torch.load(
    OUTPUT_MODEL,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    best_checkpoint["model_state_dict"]
)


# ============================================================
# FINAL TEST
# ============================================================

print(
    "\n===================================="
)

print(
    "HELD-OUT LOCAL TEST"
)

print(
    "===================================="
)

(
    test_loss,
    test_acc,
    test_prec,
    test_rec,
    test_f1
) = evaluate(
    model,
    test_loader,
    criterion
)

print(
    f"Accuracy : {test_acc * 100:.2f}%"
)

print(
    f"Precision: {test_prec * 100:.2f}%"
)

print(
    f"Recall   : {test_rec * 100:.2f}%"
)

print(
    f"F1 Score : {test_f1 * 100:.2f}%"
)

print(
    "\nBest model:",
    OUTPUT_MODEL
)
