from io import BytesIO

import librosa
import soundfile as sf
import numpy as np

from datasets import load_dataset, Audio


print("Loading one ASVspoof2019 sample...")

ds = load_dataset(
    "Bisher/ASVspoof_2019_LA",
    split="train[:1]"
)

ds = ds.cast_column(
    "audio",
    Audio(decode=False)
)

sample = ds[0]

print()
print("File:", sample["audio_file_name"])
print("Speaker:", sample["speaker_id"])
print("Label:", sample["key"])
print("System:", sample["system_id"])


# ============================================================
# READ RAW AUDIO BYTES
# ============================================================

audio_bytes = sample["audio"]["bytes"]

waveform, sample_rate = sf.read(
    BytesIO(audio_bytes),
    dtype="float32"
)

print()
print("Original sample rate:", sample_rate)
print("Original shape:", waveform.shape)


# ============================================================
# CONVERT TO MONO
# ============================================================

if waveform.ndim > 1:

    waveform = np.mean(
        waveform,
        axis=1
    )


# ============================================================
# RESAMPLE TO 16 kHz
# ============================================================

if sample_rate != 16000:

    waveform = librosa.resample(
        waveform,
        orig_sr=sample_rate,
        target_sr=16000
    )

    sample_rate = 16000


print()
print("Processed sample rate:", sample_rate)
print("Processed samples:", len(waveform))


# ============================================================
# FOUR-SECOND INPUT
# ============================================================

target_length = 16000 * 4

if len(waveform) < target_length:

    waveform = np.pad(
        waveform,
        (
            0,
            target_length - len(waveform)
        ),
        mode="constant"
    )

else:

    waveform = waveform[:target_length]


print()
print("Final waveform shape:", waveform.shape)


# ============================================================
# MEL SPECTROGRAM
# ============================================================

mel = librosa.feature.melspectrogram(
    y=waveform,
    sr=16000,
    n_fft=1024,
    hop_length=256,
    n_mels=64
)

mel_db = librosa.power_to_db(
    mel,
    ref=np.max
)

print()
print("Mel spectrogram shape:", mel_db.shape)

print()
print("========================================")
print("ASVSPOOF AUDIO TEST SUCCESSFUL")
print("========================================")