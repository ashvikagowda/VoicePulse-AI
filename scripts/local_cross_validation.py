from pathlib import Path
import copy
import random

import librosa
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import (
    Dataset,
    DataLoader,
    WeightedRandomSampler
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "real_wav"
)

FAKE_DIR = (
    PROJECT_ROOT
    / "data"
    / "fake_wav"
)

# IMPORTANT:
# Start every fold from the ASVspoof2019-trained CNN.
# DO NOT use voicepulse_ai_cnn.pth here because that model
# has already been fine-tuned on the local 66 recordings.

BASE_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "cnn"
    / "voice_cnn_asvspoof2019_best.pth"
)

SAMPLE_RATE = 16000

WINDOW_SECONDS = 4

WINDOW_SAMPLES = (
    SAMPLE_RATE
    * WINDOW_SECONDS
)

HOP_SECONDS = 2

HOP_SAMPLES = (
    SAMPLE_RATE
    * HOP_SECONDS
)

N_MELS = 64

N_FFT = 1024

HOP_LENGTH = 256

N_FOLDS = 5

EPOCHS = 12

BATCH_SIZE = 8

LEARNING_RATE = 1e-4

RANDOM_SEED = 42


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():

    DEVICE = torch.device("mps")

else:

    DEVICE = torch.device("cpu")


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(
    RANDOM_SEED
)

np.random.seed(
    RANDOM_SEED
)

torch.manual_seed(
    RANDOM_SEED
)


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
# CHECKPOINT LOADING
# ============================================================

def load_checkpoint(model):

    if not BASE_MODEL_PATH.exists():

        raise FileNotFoundError(
            "ASVspoof2019 checkpoint not found:\n"
            f"{BASE_MODEL_PATH}"
        )


    checkpoint = torch.load(
        BASE_MODEL_PATH,
        map_location="cpu"
    )


    if isinstance(
        checkpoint,
        dict
    ):

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


    return model


# ============================================================
# AUDIO LOADING
# ============================================================

def load_audio(path):

    audio, sr = librosa.load(
        path,
        sr=SAMPLE_RATE,
        mono=True
    )


    audio = np.asarray(
        audio,
        dtype=np.float32
    )


    audio = np.nan_to_num(
        audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )


    return audio


# ============================================================
# TRAINING WINDOW
# ============================================================

def make_training_window(audio):

    if len(audio) < WINDOW_SAMPLES:

        audio = np.pad(
            audio,
            (
                0,
                WINDOW_SAMPLES
                - len(audio)
            )
        )

    elif len(audio) > WINDOW_SAMPLES:

        start = random.randint(
            0,
            len(audio)
            - WINDOW_SAMPLES
        )


        audio = audio[
            start:
            start + WINDOW_SAMPLES
        ]


    return audio


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def preprocess_audio(audio):

    if len(audio) < WINDOW_SAMPLES:

        audio = np.pad(
            audio,
            (
                0,
                WINDOW_SAMPLES
                - len(audio)
            )
        )

    elif len(audio) > WINDOW_SAMPLES:

        audio = audio[
            -WINDOW_SAMPLES:
        ]


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


    mean = np.mean(
        mel
    )


    std = np.std(
        mel
    )


    if std > 1e-8:

        mel = (
            mel
            - mean
        ) / std

    else:

        mel = (
            mel
            - mean
        )


    tensor = torch.tensor(
        mel,
        dtype=torch.float32
    ).unsqueeze(0)


    return tensor


# ============================================================
# DATASET
# ============================================================

class RecordingDataset(Dataset):

    def __init__(
        self,
        records
    ):

        self.records = records


    def __len__(self):

        return len(
            self.records
        )


    def __getitem__(
        self,
        index
    ):

        path, label = (
            self.records[index]
        )


        audio = load_audio(
            path
        )


        audio = make_training_window(
            audio
        )


        features = preprocess_audio(
            audio
        )


        label_tensor = torch.tensor(
            label,
            dtype=torch.long
        )


        return (
            features,
            label_tensor
        )


# ============================================================
# COLLECT RECORDINGS
# ============================================================

def collect_records():

    records = []


    if not REAL_DIR.exists():

        raise FileNotFoundError(
            f"Real directory not found:\n"
            f"{REAL_DIR}"
        )


    if not FAKE_DIR.exists():

        raise FileNotFoundError(
            f"Fake directory not found:\n"
            f"{FAKE_DIR}"
        )


    real_files = sorted(
        REAL_DIR.glob(
            "*.wav"
        )
    )


    fake_files = sorted(
        FAKE_DIR.glob(
            "*.wav"
        )
    )


    print(
        f"Real recordings found: "
        f"{len(real_files)}"
    )


    print(
        f"Fake recordings found: "
        f"{len(fake_files)}"
    )


    for path in real_files:

        records.append(
            (
                path,
                0
            )
        )


    for path in fake_files:

        records.append(
            (
                path,
                1
            )
        )


    return records


# ============================================================
# CREATE STRATIFIED RECORDING FOLDS
# ============================================================

def create_folds(records):

    real_records = [

        item

        for item in records

        if item[1] == 0

    ]


    fake_records = [

        item

        for item in records

        if item[1] == 1

    ]


    random.shuffle(
        real_records
    )


    random.shuffle(
        fake_records
    )


    real_folds = np.array_split(
        real_records,
        N_FOLDS
    )


    fake_folds = np.array_split(
        fake_records,
        N_FOLDS
    )


    folds = []


    for fold_index in range(
        N_FOLDS
    ):

        test_records = (

            list(
                real_folds[
                    fold_index
                ]
            )

            +

            list(
                fake_folds[
                    fold_index
                ]
            )

        )


        train_records = []


        for other_index in range(
            N_FOLDS
        ):

            if (
                other_index
                == fold_index
            ):

                continue


            train_records.extend(
                list(
                    real_folds[
                        other_index
                    ]
                )
            )


            train_records.extend(
                list(
                    fake_folds[
                        other_index
                    ]
                )
            )


        random.shuffle(
            train_records
        )


        random.shuffle(
            test_records
        )


        folds.append(
            (
                train_records,
                test_records
            )
        )


    return folds


# ============================================================
# TRAIN ONE FOLD
# ============================================================

def train_fold(
    train_records
):

    model = VoiceCNN()


    model = load_checkpoint(
        model
    )


    model = model.to(
        DEVICE
    )


    model.train()


    dataset = RecordingDataset(
        train_records
    )


    labels = [

        label

        for _, label in train_records

    ]


    class_counts = np.bincount(
        labels,
        minlength=2
    )


    weights = np.array(

        [

            1.0
            / class_counts[label]

            for label in labels

        ],

        dtype=np.float64

    )


    sampler = WeightedRandomSampler(

        weights=torch.tensor(
            weights,
            dtype=torch.double
        ),

        num_samples=len(
            weights
        ),

        replacement=True

    )


    loader = DataLoader(

        dataset,

        batch_size=BATCH_SIZE,

        sampler=sampler,

        num_workers=0

    )


    criterion = nn.CrossEntropyLoss()


    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=LEARNING_RATE

    )


    best_state = None

    best_loss = float(
        "inf"
    )


    for epoch in range(
        EPOCHS
    ):

        model.train()


        running_loss = 0.0


        for (
            features,
            labels_batch
        ) in loader:

            features = features.to(
                DEVICE
            )


            labels_batch = (
                labels_batch.to(
                    DEVICE
                )
            )


            optimizer.zero_grad()


            logits = model(
                features
            )


            loss = criterion(
                logits,
                labels_batch
            )


            loss.backward()


            optimizer.step()


            running_loss += (

                loss.item()

                *

                len(
                    labels_batch
                )

            )


        epoch_loss = (

            running_loss
            /
            len(dataset)

        )


        if epoch_loss < best_loss:

            best_loss = epoch_loss


            best_state = copy.deepcopy(
                model.state_dict()
            )


        print(

            f"    Epoch "
            f"{epoch + 1:02d}"
            f"/{EPOCHS} "
            f"loss={epoch_loss:.4f}"

        )


    if best_state is not None:

        model.load_state_dict(
            best_state
        )


    return model


# ============================================================
# CREATE TEST WINDOWS
# ============================================================

def make_test_windows(
    audio
):

    if len(audio) <= WINDOW_SAMPLES:

        return [
            audio
        ]


    windows = []


    start = 0


    while (

        start + WINDOW_SAMPLES
        <= len(audio)

    ):

        windows.append(

            audio[
                start:
                start + WINDOW_SAMPLES
            ]

        )


        start += HOP_SAMPLES


    final_window = audio[
        -WINDOW_SAMPLES:
    ]


    if not windows:

        windows.append(
            final_window
        )

    elif not np.array_equal(

        windows[-1],

        final_window

    ):

        windows.append(
            final_window
        )


    return windows


# ============================================================
# RECORDING-LEVEL PREDICTION
# ============================================================

def predict_recording(
    model,
    path
):

    model.eval()


    audio = load_audio(
        path
    )


    windows = make_test_windows(
        audio
    )


    probabilities = []


    with torch.no_grad():

        for window in windows:

            features = (
                preprocess_audio(
                    window
                )
            )


            features = (
                features
                .unsqueeze(0)
                .to(DEVICE)
            )


            logits = model(
                features
            )


            probs = torch.softmax(
                logits,
                dim=1
            )


            fake_probability = float(

                probs[
                    0,
                    1
                ].item()

            )


            probabilities.append(
                fake_probability
            )


    fake_score = float(
        np.mean(
            probabilities
        )
    )


    prediction = (

        1

        if fake_score >= 0.5

        else 0

    )


    return (
        prediction,
        fake_score
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n========================================"
    )


    print(
        "VOICEPULSE AI LOCAL CROSS-VALIDATION"
    )


    print(
        "========================================"
    )


    print(
        f"Device: {DEVICE}"
    )


    print(
        "\nBase checkpoint:"
    )


    print(
        BASE_MODEL_PATH
    )


    records = collect_records()


    print(
        f"\nTotal recordings: "
        f"{len(records)}"
    )


    print(
        "Real:",
        sum(
            label == 0
            for _, label in records
        )
    )


    print(
        "Fake:",
        sum(
            label == 1
            for _, label in records
        )
    )


    folds = create_folds(
        records
    )


    # ========================================================
    # GLOBAL OUT-OF-FOLD RESULTS
    # ========================================================

    all_true = []

    all_pred = []

    all_scores = []

    all_names = []

    all_fold_ids = []

    fold_results = []


    # ========================================================
    # FIVE FOLDS
    # ========================================================

    for fold_index, (

        train_records,

        test_records

    ) in enumerate(

        folds,

        start=1

    ):

        print(
            "\n----------------------------------------"
        )


        print(
            f"FOLD {fold_index}/{N_FOLDS}"
        )


        print(
            "----------------------------------------"
        )


        print(
            f"Training recordings: "
            f"{len(train_records)}"
        )


        print(
            f"Test recordings: "
            f"{len(test_records)}"
        )


        train_real = sum(

            label == 0

            for _, label
            in train_records

        )


        train_fake = sum(

            label == 1

            for _, label
            in train_records

        )


        test_real = sum(

            label == 0

            for _, label
            in test_records

        )


        test_fake = sum(

            label == 1

            for _, label
            in test_records

        )


        print(
            f"Train: "
            f"{train_real} real / "
            f"{train_fake} fake"
        )


        print(
            f"Test: "
            f"{test_real} real / "
            f"{test_fake} fake"
        )


        print(
            "\nFine-tuning fold..."
        )


        model = train_fold(
            train_records
        )


        fold_true = []

        fold_pred = []

        fold_scores = []


        print(
            "\nTesting unseen recordings..."
        )


        for path, true_label in (
            test_records
        ):

            (
                prediction,
                fake_score
            ) = predict_recording(

                model,

                path

            )


            fold_true.append(
                true_label
            )


            fold_pred.append(
                prediction
            )


            fold_scores.append(
                fake_score
            )


            all_true.append(
                true_label
            )


            all_pred.append(
                prediction
            )


            all_scores.append(
                fake_score
            )


            all_names.append(
                path.name
            )


            all_fold_ids.append(
                fold_index
            )


            print(

                f"{path.name}: "
                f"true={true_label} "
                f"fake_score={fake_score:.4f} "
                f"pred={prediction}"

            )


        # ====================================================
        # FOLD METRICS
        # ====================================================

        fold_accuracy = (
            accuracy_score(
                fold_true,
                fold_pred
            )
        )


        fold_precision = (
            precision_score(
                fold_true,
                fold_pred,
                zero_division=0
            )
        )


        fold_recall = (
            recall_score(
                fold_true,
                fold_pred,
                zero_division=0
            )
        )


        fold_f1 = (
            f1_score(
                fold_true,
                fold_pred,
                zero_division=0
            )
        )


        print(
            "\nFold metrics:"
        )


        print(
            f"Accuracy : "
            f"{fold_accuracy:.4f}"
        )


        print(
            f"Precision: "
            f"{fold_precision:.4f}"
        )


        print(
            f"Recall   : "
            f"{fold_recall:.4f}"
        )


        print(
            f"F1       : "
            f"{fold_f1:.4f}"
        )


        fold_results.append(

            {

                "accuracy":
                fold_accuracy,

                "precision":
                fold_precision,

                "recall":
                fold_recall,

                "f1":
                fold_f1

            }

        )


    # ========================================================
    # OVERALL RECORDING-LEVEL RESULTS
    # ========================================================

    print(
        "\n========================================"
    )


    print(
        "OVERALL RECORDING-LEVEL RESULTS"
    )


    print(
        "========================================"
    )


    overall_accuracy = (
        accuracy_score(
            all_true,
            all_pred
        )
    )


    overall_precision = (
        precision_score(
            all_true,
            all_pred,
            zero_division=0
        )
    )


    overall_recall = (
        recall_score(
            all_true,
            all_pred,
            zero_division=0
        )
    )


    overall_f1 = (
        f1_score(
            all_true,
            all_pred,
            zero_division=0
        )
    )


    print(
        f"Accuracy : "
        f"{overall_accuracy * 100:.2f}%"
    )


    print(
        f"Precision: "
        f"{overall_precision * 100:.2f}%"
    )


    print(
        f"Recall   : "
        f"{overall_recall * 100:.2f}%"
    )


    print(
        f"F1 Score : "
        f"{overall_f1 * 100:.2f}%"
    )


    print(
        "\nConfusion matrix:"
    )


    print(
        confusion_matrix(
            all_true,
            all_pred
        )
    )


    # ========================================================
    # FIVE-FOLD MEAN ± STANDARD DEVIATION
    # ========================================================

    print(
        "\nFive-fold mean ± standard deviation:"
    )


    for metric in [

        "accuracy",

        "precision",

        "recall",

        "f1"

    ]:

        values = np.array(

            [

                result[
                    metric
                ]

                for result
                in fold_results

            ]

        )


        print(

            f"{metric.capitalize():10s}: "

            f"{values.mean() * 100:.2f}% "

            f"± "

            f"{values.std() * 100:.2f}%"

        )


    # ========================================================
    # THRESHOLD ANALYSIS
    #
    # These are OUT-OF-FOLD probabilities, meaning every
    # recording's score came from a model that did not train
    # on that recording.
    # ========================================================

    print(
        "\n========================================"
    )


    print(
        "THRESHOLD ANALYSIS"
    )


    print(
        "========================================"
    )


    thresholds = [

        0.50,

        0.55,

        0.60,

        0.65,

        0.70,

        0.75,

        0.80,

        0.85,

        0.90

    ]


    print(
        "\nThreshold | Accuracy | Precision | Recall | F1"
    )


    print(
        "----------|----------|-----------|--------|--------"
    )


    for threshold in thresholds:

        threshold_predictions = [

            1
            if score >= threshold
            else 0

            for score
            in all_scores

        ]


        threshold_accuracy = (
            accuracy_score(
                all_true,
                threshold_predictions
            )
        )


        threshold_precision = (
            precision_score(
                all_true,
                threshold_predictions,
                zero_division=0
            )
        )


        threshold_recall = (
            recall_score(
                all_true,
                threshold_predictions,
                zero_division=0
            )
        )


        threshold_f1 = (
            f1_score(
                all_true,
                threshold_predictions,
                zero_division=0
            )
        )


        print(

            f"{threshold:9.2f} | "

            f"{threshold_accuracy * 100:8.2f}% | "

            f"{threshold_precision * 100:9.2f}% | "

            f"{threshold_recall * 100:6.2f}% | "

            f"{threshold_f1 * 100:6.2f}%"

        )


    # ========================================================
    # INDIVIDUAL OUT-OF-FOLD SCORES
    # ========================================================

    print(
        "\n========================================"
    )


    print(
        "OUT-OF-FOLD RECORDING SCORES"
    )


    print(
        "========================================"
    )


    results = sorted(

        zip(
            all_fold_ids,
            all_names,
            all_true,
            all_scores,
            all_pred
        ),

        key=lambda x: (
            x[0],
            x[2],
            x[1]
        )

    )


    for (

        fold_id,
        name,
        true_label,
        fake_score,
        original_prediction

    ) in results:

        print(

            f"Fold={fold_id} | "
            f"true={true_label} | "
            f"fake_score={fake_score:.4f} | "
            f"pred@0.50={original_prediction} | "
            f"{name}"

        )


    # ========================================================
    # DONE
    # ========================================================

    print(
        "\n========================================"
    )


    print(
        "DONE"
    )


    print(
        "========================================"
    )


if __name__ == "__main__":

    main()
