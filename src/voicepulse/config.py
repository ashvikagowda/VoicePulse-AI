from pathlib import Path
import torch


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "cnn"
    / "voicepulse_ai_cnn.pth"
)


# ============================================================
# AUDIO CONFIGURATION
# ============================================================

SAMPLE_RATE = 16_000
MIC_SAMPLE_RATE = 48_000

WINDOW_SECONDS = 4
UPDATE_SECONDS = 1

WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SECONDS

N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256


# ============================================================
# DETECTION CONFIGURATION
# ============================================================

FAKE_DETECTION_THRESHOLD = 0.85

SMOOTHING_WINDOWS = 2


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
