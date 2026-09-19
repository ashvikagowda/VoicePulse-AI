import time
from collections import deque

import librosa
import numpy as np
import sounddevice as sd
import torch
import torch.nn as nn
import onnxruntime as ort


# ============================================================
# CONFIG
# ============================================================

SAMPLE_RATE = 16000

WINDOW_SECONDS = 4
HOP_SECONDS = 2

WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SECONDS
HOP_SAMPLES = SAMPLE_RATE * HOP_SECONDS

AASIST_WINDOW = 64600
CNN_WINDOW = 64000

CNN_MODEL_PATH = (
    "models/cnn/voice_cnn_asvspoof2019_best.pth"
)

AASIST_MODEL_PATH = (
    "models/aasist/aasist.onnx"
)

# Final fusion found from your external benchmark
AASIST_WEIGHT = 0.50
CNN_WEIGHT = 0.50


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print("Torch device:", DEVICE)


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
# PREPROCESS
# ============================================================

def prepare_aasist(waveform):

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


def prepare_cnn(waveform):

    if len(waveform) >= CNN_WINDOW:

        waveform = waveform[:CNN_WINDOW]

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
        n_fft=1024,
        hop_length=256,
        n_mels=64,
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
# DETECTOR
# ============================================================

@torch.no_grad()
def detect(waveform):

    # --------------------------------------------------------
    # AASIST
    # --------------------------------------------------------

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

    probabilities = (
        exp_output /
        exp_output.sum()
    )

    # Class 0 = fake for your current AASIST setup
    aasist_fake = float(
        probabilities[0]
    )

    # --------------------------------------------------------
    # CNN
    # --------------------------------------------------------

    cnn_input = prepare_cnn(
        waveform
    ).to(DEVICE)

    logits = cnn(
        cnn_input
    )

    cnn_probabilities = torch.softmax(
        logits,
        dim=1
    )

    # CNN: class 1 = fake
    cnn_fake = float(
        cnn_probabilities[0, 1]
        .item()
    )

    # --------------------------------------------------------
    # FUSION
    # --------------------------------------------------------

    fusion_score = (
        AASIST_WEIGHT * aasist_fake
        +
        CNN_WEIGHT * cnn_fake
    )

    # --------------------------------------------------------
    # RISK
    # --------------------------------------------------------

    if fusion_score >= 0.75:
        risk = "HIGH"
        decision = "POSSIBLE AI VOICE"

    elif fusion_score >= 0.50:
        risk = "MEDIUM"
        decision = "SUSPICIOUS"

    elif fusion_score >= 0.30:
        risk = "LOW"
        decision = "LIKELY REAL"

    else:
        risk = "LOW"
        decision = "LIKELY REAL"

    return (
        aasist_fake,
        cnn_fake,
        fusion_score,
        risk,
        decision
    )


# ============================================================
# MICROPHONE
# ============================================================

print("\nAvailable audio devices:")
print(sd.query_devices())

print("\nStarting microphone...")
print(
    f"Window: {WINDOW_SECONDS} seconds"
)

print(
    f"Update interval: {HOP_SECONDS} seconds"
)

print("\nSpeak into the microphone.")
print("Press Ctrl+C to stop.\n")


buffer = deque(
    maxlen=WINDOW_SAMPLES
)

first_window = True

try:

    while True:

        # Capture 2 seconds
        audio = sd.rec(
            HOP_SAMPLES,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32"
        )

        sd.wait()

        audio = audio[:, 0]

        # Add to rolling buffer
        buffer.extend(audio)

        # Wait until we have 4 seconds
        if len(buffer) < WINDOW_SAMPLES:

            if first_window:

                remaining = (
                    WINDOW_SAMPLES -
                    len(buffer)
                )

                print(
                    f"Collecting audio... "
                    f"{len(buffer) / SAMPLE_RATE:.1f}/"
                    f"{WINDOW_SECONDS:.1f} sec"
                )

                first_window = False

            continue

        # Convert rolling buffer to array
        waveform = np.asarray(
            buffer,
            dtype=np.float32
        )

        print("\n" + "=" * 55)

        print(
            time.strftime("%H:%M:%S"),
            "Analyzing 4-second window..."
        )

        start_time = time.time()

        (
            aasist_fake,
            cnn_fake,
            fusion_score,
            risk,
            decision
        ) = detect(waveform)

        processing_time = (
            time.time() - start_time
        )

        print(
            f"AASIST fake score : "
            f"{aasist_fake * 100:.2f}%"
        )

        print(
            f"CNN fake score    : "
            f"{cnn_fake * 100:.2f}%"
        )

        print(
            f"Fusion score      : "
            f"{fusion_score * 100:.2f}%"
        )

        print(
            f"Risk              : {risk}"
        )

        print(
            f"Decision          : {decision}"
        )

        print(
            f"Processing time   : "
            f"{processing_time:.2f} sec"
        )

        if processing_time < HOP_SECONDS:

            print(
                "Status            : REAL-TIME ✓"
            )

        else:

            print(
                "Status            : "
                "Processing slower than update interval"
            )

except KeyboardInterrupt:

    print("\n\nVoiceGuard stopped.")
