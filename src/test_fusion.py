from io import BytesIO

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import onnxruntime as ort


# ============================================================
# CONFIG
# ============================================================

SAMPLE_RATE = 16000
WINDOW_SIZE = 64600          # AASIST input
CNN_WINDOW_SIZE = 64000      # 4 seconds

CNN_MODEL_PATH = "models/cnn/voice_cnn_asvspoof2019_best.pth"
AASIST_MODEL_PATH = "models/aasist/aasist.onnx"


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

cnn = VoiceCNN().to(DEVICE)

checkpoint = torch.load(
    CNN_MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

cnn.load_state_dict(checkpoint["model_state_dict"])
cnn.eval()

print("CNN loaded")


# ============================================================
# LOAD AASIST
# ============================================================

providers = [
    "CoreMLExecutionProvider",
    "CPUExecutionProvider"
]

aasist_session = ort.InferenceSession(
    AASIST_MODEL_PATH,
    providers=providers
)

print(
    "AASIST providers:",
    aasist_session.get_providers()
)


# ============================================================
# AUDIO DECODING
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

def prepare_cnn_audio(waveform):

    if len(waveform) >= CNN_WINDOW_SIZE:

        start = (
            len(waveform) - CNN_WINDOW_SIZE
        ) // 2

        waveform = waveform[
            start:start + CNN_WINDOW_SIZE
        ]

    else:

        padded = np.zeros(
            CNN_WINDOW_SIZE,
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

    tensor = torch.tensor(
        mel_db,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)

    return tensor


# ============================================================
# AASIST PREPROCESSING
# ============================================================

def prepare_aasist_audio(waveform):

    if len(waveform) >= WINDOW_SIZE:

        start = (
            len(waveform) - WINDOW_SIZE
        ) // 2

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

    return waveform.reshape(
        1, -1
    ).astype(np.float32)


# ============================================================
# PREDICTION
# ============================================================

def predict(audio_bytes):

    waveform = decode_audio(audio_bytes)

    # --------------------------------------------------------
    # CNN
    # --------------------------------------------------------

    cnn_input = prepare_cnn_audio(
        waveform
    ).to(DEVICE)

    with torch.no_grad():

        cnn_logits = cnn(cnn_input)

        cnn_probs = torch.softmax(
            cnn_logits,
            dim=1
        )

    cnn_fake = cnn_probs[
        0, 1
    ].item()

    # --------------------------------------------------------
    # AASIST
    # --------------------------------------------------------

    aasist_input = prepare_aasist_audio(
        waveform
    )

    input_name = aasist_session.get_inputs()[0].name

    output = aasist_session.run(
        None,
        {
            input_name: aasist_input
        }
    )[0][0]

    aasist_exp = np.exp(
        output - np.max(output)
    )

    aasist_probs = (
        aasist_exp /
        aasist_exp.sum()
    )

    aasist_fake = float(
        aasist_probs[1]
    )

    # --------------------------------------------------------
    # SIMPLE FUSION
    # --------------------------------------------------------

    fusion_score = (
        0.6 * aasist_fake +
        0.4 * cnn_fake
    )

    if fusion_score >= 0.70:
        decision = "FAKE / HIGH RISK"

    elif fusion_score >= 0.40:
        decision = "SUSPICIOUS"

    else:
        decision = "LIKELY REAL"

    return (
        aasist_fake,
        cnn_fake,
        fusion_score,
        decision
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    import glob

    files = glob.glob(
        "data/fake_wav/*.wav"
    )

    if not files:
        raise FileNotFoundError(
            "No WAV files found in data/fake_wav"
        )

    test_file = files[0]

    print("\nTesting:", test_file)

    with open(
        test_file,
        "rb"
    ) as f:
        audio_bytes = f.read()

    (
        aasist_fake,
        cnn_fake,
        fusion_score,
        decision
    ) = predict(audio_bytes)

    print(
        f"\nAASIST fake probability : "
        f"{aasist_fake * 100:.2f}%"
    )

    print(
        f"CNN fake probability    : "
        f"{cnn_fake * 100:.2f}%"
    )

    print(
        f"Fusion score            : "
        f"{fusion_score * 100:.2f}%"
    )

    print(
        f"Decision                : "
        f"{decision}"
    )
