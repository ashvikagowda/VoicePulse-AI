from pathlib import Path
import random

import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

FAKE_DIR = PROJECT_DIR / "data" / "fake_wav"
REAL_DIR = PROJECT_DIR / "data" / "real_wav"

MODEL_DIR = PROJECT_DIR / "models" / "cnn"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_RATE = 16000

WINDOW_SECONDS = 4

WINDOW_SIZE = SAMPLE_RATE * WINDOW_SECONDS

HOP_SECONDS = 2

HOP_SIZE = SAMPLE_RATE * HOP_SECONDS

N_MELS = 64

N_FFT = 1024

HOP_LENGTH = 256

BATCH_SIZE = 16

EPOCHS = 20

LEARNING_RATE = 0.001

RANDOM_SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)

np.random.seed(RANDOM_SEED)

torch.manual_seed(RANDOM_SEED)


# ============================================================
# FIND RECORDINGS
# ============================================================

fake_files = sorted(
    FAKE_DIR.glob("*.wav")
)

real_files = sorted(
    REAL_DIR.glob("*.wav")
)


print("Fake recordings:", len(fake_files))
print("Real recordings:", len(real_files))


# ============================================================
# RECORDING-LEVEL SPLIT
# ============================================================

def split_files(files):

    files = files.copy()

    random.shuffle(files)

    total = len(files)

    train_end = int(total * 0.70)

    val_end = int(total * 0.85)

    train = files[:train_end]

    validation = files[train_end:val_end]

    test = files[val_end:]

    return train, validation, test


fake_train, fake_val, fake_test = split_files(
    fake_files
)

real_train, real_val, real_test = split_files(
    real_files
)


# ============================================================
# CREATE RECORDING LABEL LISTS
# ============================================================

train_recordings = (
    [(file, 1) for file in fake_train]
    +
    [(file, 0) for file in real_train]
)

val_recordings = (
    [(file, 1) for file in fake_val]
    +
    [(file, 0) for file in real_val]
)

test_recordings = (
    [(file, 1) for file in fake_test]
    +
    [(file, 0) for file in real_test]
)


print()
print("========================================")
print("RECORDING-LEVEL SPLIT")
print("========================================")

print(
    "Train recordings:",
    len(train_recordings)
)

print(
    "Validation recordings:",
    len(val_recordings)
)

print(
    "Test recordings:",
    len(test_recordings)
)


# ============================================================
# LOAD AUDIO
# ============================================================

def load_audio(audio_path):

    waveform, sample_rate = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    waveform = waveform.astype(
        np.float32
    )

    return waveform


# ============================================================
# CREATE WINDOWS
# ============================================================

def create_windows(waveform):

    windows = []

    # Recording shorter than 4 seconds
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

    # Multiple overlapping windows
    start = 0

    while start + WINDOW_SIZE <= len(waveform):

        window = waveform[
            start:start + WINDOW_SIZE
        ]

        windows.append(window)

        start += HOP_SIZE

    # Include final part of the recording
    if start < len(waveform):

        final_window = waveform[-WINDOW_SIZE:]

        windows.append(final_window)

    return windows


# ============================================================
# WAVEFORM → MEL SPECTROGRAM
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

    # Normalize spectrogram
    mean = mel_db.mean()

    std = mel_db.std()

    mel_db = (
        mel_db - mean
    ) / (std + 1e-8)

    return mel_db.astype(
        np.float32
    )


# ============================================================
# BUILD WINDOW DATASET
# ============================================================

def build_examples(recordings):

    examples = []

    for audio_path, label in recordings:

        waveform = load_audio(
            audio_path
        )

        windows = create_windows(
            waveform
        )

        for window in windows:

            mel = waveform_to_mel(
                window
            )

            examples.append(
                (
                    mel,
                    label,
                    audio_path.name
                )
            )

    return examples


print()
print("Creating training windows...")

train_examples = build_examples(
    train_recordings
)

print(
    "Training windows:",
    len(train_examples)
)


print()
print("Creating validation windows...")

val_examples = build_examples(
    val_recordings
)

print(
    "Validation windows:",
    len(val_examples)
)


print()
print("Creating test windows...")

test_examples = build_examples(
    test_recordings
)

print(
    "Test windows:",
    len(test_examples)
)


# ============================================================
# PYTORCH DATASET
# ============================================================

class MelDataset(Dataset):

    def __init__(self, examples):

        self.examples = examples


    def __len__(self):

        return len(self.examples)


    def __getitem__(self, index):

        mel, label, filename = (
            self.examples[index]
        )

        # (64, 251)
        # →
        # (1, 64, 251)

        mel = np.expand_dims(
            mel,
            axis=0
        )

        mel_tensor = torch.tensor(
            mel,
            dtype=torch.float32
        )

        label_tensor = torch.tensor(
            label,
            dtype=torch.long
        )

        return (
            mel_tensor,
            label_tensor
        )


# ============================================================
# DATA LOADERS
# ============================================================

train_dataset = MelDataset(
    train_examples
)

val_dataset = MelDataset(
    val_examples
)

test_dataset = MelDataset(
    test_examples
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# CNN
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

        x = self.classifier(x)

        return x


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():

    device = torch.device("mps")

else:

    device = torch.device("cpu")


print()
print("Device:", device)


# ============================================================
# MODEL
# ============================================================

model = VoiceCNN().to(device)


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAIN
# ============================================================

def train_one_epoch():

    model.train()

    total_loss = 0

    correct = 0

    total = 0

    for inputs, labels in train_loader:

        inputs = inputs.to(device)

        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
            * inputs.size(0)
        )

        predictions = (
            outputs.argmax(dim=1)
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    return (
        total_loss / total,
        correct / total
    )


# ============================================================
# VALIDATE
# ============================================================

def evaluate(loader):

    model.eval()

    total_loss = 0

    correct = 0

    total = 0

    with torch.no_grad():

        for inputs, labels in loader:

            inputs = inputs.to(device)

            labels = labels.to(device)

            outputs = model(inputs)

            loss = criterion(
                outputs,
                labels
            )

            total_loss += (
                loss.item()
                * inputs.size(0)
            )

            predictions = (
                outputs.argmax(dim=1)
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return (
        total_loss / total,
        correct / total
    )


# ============================================================
# TRAINING LOOP
# ============================================================

best_val_accuracy = 0.0

best_model_path = (
    MODEL_DIR /
    "voice_cnn_windows_best.pth"
)


print()
print("========================================")
print("TRAINING CNN")
print("========================================")


for epoch in range(
    1,
    EPOCHS + 1
):

    train_loss, train_accuracy = (
        train_one_epoch()
    )

    val_loss, val_accuracy = (
        evaluate(val_loader)
    )

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: "
        f"{train_accuracy * 100:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: "
        f"{val_accuracy * 100:.2f}%"
    )

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            best_model_path
        )


# ============================================================
# LOAD BEST MODEL
# ============================================================

print()
print("Loading best CNN model...")

model.load_state_dict(
    torch.load(
        best_model_path,
        map_location=device
    )
)


# ============================================================
# TEST
# ============================================================

test_loss, test_accuracy = evaluate(
    test_loader
)


print()
print("========================================")
print("FINAL CNN RESULT")
print("========================================")

print(
    f"Test Loss: {test_loss:.4f}"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print()
print("Best model:")

print(best_model_path)