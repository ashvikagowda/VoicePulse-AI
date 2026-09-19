import argparse
import random
from io import BytesIO

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from datasets import Audio, load_dataset
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. CONFIG
# ============================================================

SEED = 42
SAMPLE_RATE = 16000
WINDOW_SECONDS = 4
WINDOW_SIZE = SAMPLE_RATE * WINDOW_SECONDS

N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256

BATCH_SIZE = 16
LEARNING_RATE = 1e-3

MODEL_PATH = "models/cnn/voice_cnn_asvspoof2019_best.pth"


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# 3. DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("Device:", DEVICE)


# ============================================================
# 4. LOAD DATASETS
# ============================================================

print("\nLoading ASVspoof2019 LA...")

train_ds = load_dataset(
    "Bisher/ASVspoof_2019_LA",
    split="train"
)

val_ds = load_dataset(
    "Bisher/ASVspoof_2019_LA",
    split="validation"
)

train_ds = train_ds.cast_column(
    "audio",
    Audio(decode=False)
)

val_ds = val_ds.cast_column(
    "audio",
    Audio(decode=False)
)

print("Train recordings:", len(train_ds))
print("Validation recordings:", len(val_ds))


# ============================================================
# 5. CREATE BALANCED SUBSETS
# ============================================================

def get_balanced_indices(dataset, samples_per_class):
    real_indices = []
    fake_indices = []

    for i, label in enumerate(dataset["key"]):
        if label == 0:
            real_indices.append(i)
        elif label == 1:
            fake_indices.append(i)

    print("Available real :", len(real_indices))
    print("Available fake :", len(fake_indices))

    if samples_per_class > len(real_indices):
        raise ValueError(
            f"Requested {samples_per_class} real samples, "
            f"but only {len(real_indices)} available."
        )

    if samples_per_class > len(fake_indices):
        raise ValueError(
            f"Requested {samples_per_class} fake samples, "
            f"but only {len(fake_indices)} available."
        )

    rng = np.random.default_rng(SEED)

    real_selected = rng.choice(
        real_indices,
        size=samples_per_class,
        replace=False
    )

    fake_selected = rng.choice(
        fake_indices,
        size=samples_per_class,
        replace=False
    )

    indices = np.concatenate([
        real_selected,
        fake_selected
    ])

    rng.shuffle(indices)

    return indices.tolist()


# ============================================================
# 6. AUDIO -> LOG-MEL
# ============================================================

def audio_to_mel(audio_bytes, training=True):
    waveform, sample_rate = sf.read(
        BytesIO(audio_bytes),
        dtype="float32"
    )

    # Convert stereo -> mono
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)

    # Resample if required
    if sample_rate != SAMPLE_RATE:
        waveform = librosa.resample(
            waveform,
            orig_sr=sample_rate,
            target_sr=SAMPLE_RATE
        )

    waveform = waveform.astype(np.float32)

    # --------------------------------------------------------
    # Crop / pad to exactly 4 seconds
    # --------------------------------------------------------

    if len(waveform) >= WINDOW_SIZE:

        if training:
            # Random crop during training
            start = np.random.randint(
                0,
                len(waveform) - WINDOW_SIZE + 1
            )
        else:
            # Center crop during validation
            start = (len(waveform) - WINDOW_SIZE) // 2

        waveform = waveform[
            start:start + WINDOW_SIZE
        ]

    else:
        # Zero-pad short recordings
        padded = np.zeros(
            WINDOW_SIZE,
            dtype=np.float32
        )

        padded[:len(waveform)] = waveform
        waveform = padded

    # --------------------------------------------------------
    # Log-Mel spectrogram
    # --------------------------------------------------------

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

    # Per-sample normalization
    mel_db = (
        mel_db - mel_db.mean()
    ) / (
        mel_db.std() + 1e-8
    )

    return mel_db.astype(np.float32)


# ============================================================
# 7. PYTORCH DATASET
# ============================================================

class ASVspoofDataset(Dataset):

    def __init__(
        self,
        hf_dataset,
        indices,
        training=False
    ):
        self.dataset = hf_dataset
        self.indices = indices
        self.training = training

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        real_idx = self.indices[idx]

        sample = self.dataset[real_idx]

        audio_bytes = sample["audio"]["bytes"]
        label = int(sample["key"])

        mel = audio_to_mel(
            audio_bytes,
            training=self.training
        )

        # Add channel dimension
        mel = torch.tensor(
            mel,
            dtype=torch.float32
        ).unsqueeze(0)

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return mel, label


# ============================================================
# 8. CNN MODEL
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
# 9. METRICS
# ============================================================

def calculate_metrics(all_targets, all_predictions):

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for target, prediction in zip(
        all_targets,
        all_predictions
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

    accuracy = (tp + tn) / max(total, 1)

    precision = tp / max(tp + fp, 1)

    recall = tp / max(tp + fn, 1)

    f1 = (
        2 * precision * recall /
        max(precision + recall, 1e-8)
    )

    return accuracy, precision, recall, f1


# ============================================================
# 10. TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion
):

    model.train()

    total_loss = 0.0

    all_targets = []
    all_predictions = []

    for batch_idx, (x, y) in enumerate(loader):

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

        predictions = torch.argmax(
            logits,
            dim=1
        )

        all_targets.extend(
            y.detach().cpu().numpy().tolist()
        )

        all_predictions.extend(
            predictions.detach().cpu().numpy().tolist()
        )

        if (batch_idx + 1) % 50 == 0:
            print(
                f"  Batch {batch_idx + 1}/{len(loader)}"
            )

    accuracy, precision, recall, f1 = calculate_metrics(
        all_targets,
        all_predictions
    )

    avg_loss = total_loss / len(loader)

    return avg_loss, accuracy, precision, recall, f1


# ============================================================
# 11. VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0.0

    all_targets = []
    all_predictions = []

    for x, y in loader:

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x)

        loss = criterion(
            logits,
            y
        )

        total_loss += loss.item()

        predictions = torch.argmax(
            logits,
            dim=1
        )

        all_targets.extend(
            y.cpu().numpy().tolist()
        )

        all_predictions.extend(
            predictions.cpu().numpy().tolist()
        )

    accuracy, precision, recall, f1 = calculate_metrics(
        all_targets,
        all_predictions
    )

    avg_loss = total_loss / len(loader)

    return avg_loss, accuracy, precision, recall, f1


# ============================================================
# 12. MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-per-class",
        type=int,
        default=2580
    )

    parser.add_argument(
        "--val-per-class",
        type=int,
        default=2548
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10
    )

    args = parser.parse_args()

    print("\nCreating balanced training subset...")

    train_indices = get_balanced_indices(
        train_ds,
        args.train_per_class
    )

    print("\nCreating balanced validation subset...")

    val_indices = get_balanced_indices(
        val_ds,
        args.val_per_class
    )

    print(
        f"\nTraining recordings: "
        f"{len(train_indices)}"
    )

    print(
        f"Validation recordings: "
        f"{len(val_indices)}"
    )

    # --------------------------------------------------------
    # PyTorch datasets
    # --------------------------------------------------------

    train_data = ASVspoofDataset(
        train_ds,
        train_indices,
        training=True
    )

    val_data = ASVspoofDataset(
        val_ds,
        val_indices,
        training=False
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------------
    # Check one sample
    # --------------------------------------------------------

    print("\nChecking one processed sample...")

    sample_x, sample_y = train_data[0]

    print(
        "Mel tensor shape:",
        tuple(sample_x.shape)
    )

    print(
        "Label:",
        sample_y.item()
    )

    # Expected:
    # [1, 64, 251]

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = VoiceCNN().to(DEVICE)

    print("\nModel:")
    print(model)

    # --------------------------------------------------------
    # Loss + optimizer
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_val_f1 = 0.0

    print("\nStarting training...\n")

    for epoch in range(1, args.epochs + 1):

        print(
            f"========== EPOCH {epoch}/{args.epochs} =========="
        )

        train_loss, train_acc, train_prec, train_rec, train_f1 = (
            train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion
            )
        )

        print(
            f"\nTrain Loss      : {train_loss:.4f}"
        )

        print(
            f"Train Accuracy  : {train_acc * 100:.2f}%"
        )

        print(
            f"Train Precision : {train_prec * 100:.2f}%"
        )

        print(
            f"Train Recall    : {train_rec * 100:.2f}%"
        )

        print(
            f"Train F1        : {train_f1 * 100:.2f}%"
        )

        print("\nRunning validation...")

        val_loss, val_acc, val_prec, val_rec, val_f1 = (
            validate(
                model,
                val_loader,
                criterion
            )
        )

        print(
            f"\nVal Loss        : {val_loss:.4f}"
        )

        print(
            f"Val Accuracy    : {val_acc * 100:.2f}%"
        )

        print(
            f"Val Precision   : {val_prec * 100:.2f}%"
        )

        print(
            f"Val Recall      : {val_rec * 100:.2f}%"
        )

        print(
            f"Val F1          : {val_f1 * 100:.2f}%"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_f1 > best_val_f1:

            best_val_f1 = val_f1

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "val_f1": val_f1,
                    "val_accuracy": val_acc,
                    "val_precision": val_prec,
                    "val_recall": val_rec,
                    "sample_rate": SAMPLE_RATE,
                    "window_seconds": WINDOW_SECONDS,
                    "n_mels": N_MELS,
                    "n_fft": N_FFT,
                    "hop_length": HOP_LENGTH,
                },
                MODEL_PATH
            )

            print(
                f"\n*** BEST MODEL SAVED ***"
            )

            print(
                f"Path: {MODEL_PATH}"
            )

        print()

    print("\n====================================")
    print("TRAINING COMPLETE")
    print("====================================")

    print(
        f"Best validation F1: "
        f"{best_val_f1 * 100:.2f}%"
    )

    print(
        f"Saved model: {MODEL_PATH}"
    )


if __name__ == "__main__":
    main()
