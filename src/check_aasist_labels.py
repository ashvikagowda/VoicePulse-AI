from io import BytesIO
import glob

import librosa
import numpy as np
import soundfile as sf
import onnxruntime as ort


SAMPLE_RATE = 16000
WINDOW_SIZE = 64600

MODEL_PATH = "models/aasist/aasist.onnx"

session = ort.InferenceSession(
    MODEL_PATH,
    providers=[
        "CoreMLExecutionProvider",
        "CPUExecutionProvider"
    ]
)

input_name = session.get_inputs()[0].name


def load_audio(path):

    with open(path, "rb") as f:
        audio_bytes = f.read()

    waveform, sr = sf.read(
        BytesIO(audio_bytes),
        dtype="float32"
    )

    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)

    if sr != SAMPLE_RATE:
        waveform = librosa.resample(
            waveform,
            orig_sr=sr,
            target_sr=SAMPLE_RATE
        )

    if len(waveform) >= WINDOW_SIZE:
        waveform = waveform[:WINDOW_SIZE]
    else:
        padded = np.zeros(
            WINDOW_SIZE,
            dtype=np.float32
        )
        padded[:len(waveform)] = waveform
        waveform = padded

    return waveform.reshape(1, -1).astype(np.float32)


def predict(path):

    waveform = load_audio(path)

    logits = session.run(
        None,
        {
            input_name: waveform
        }
    )[0][0]

    exp_logits = np.exp(
        logits - np.max(logits)
    )

    probs = exp_logits / exp_logits.sum()

    return logits, probs


fake_files = sorted(
    glob.glob("data/fake_wav/*.wav")
)

real_files = sorted(
    glob.glob("data/real_wav/*.wav")
)

print("\n========== KNOWN FAKE ==========")

fake_path = fake_files[0]

logits, probs = predict(fake_path)

print("File:", fake_path)
print("Logits:", logits)
print(
    f"Class 0 probability: {probs[0] * 100:.2f}%"
)
print(
    f"Class 1 probability: {probs[1] * 100:.2f}%"
)
print(
    "Predicted class:",
    int(np.argmax(probs))
)

print("\n========== KNOWN REAL ==========")

real_path = real_files[0]

logits, probs = predict(real_path)

print("File:", real_path)
print("Logits:", logits)
print(
    f"Class 0 probability: {probs[0] * 100:.2f}%"
)
print(
    f"Class 1 probability: {probs[1] * 100:.2f}%"
)
print(
    "Predicted class:",
    int(np.argmax(probs))
)
