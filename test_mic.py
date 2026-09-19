import sounddevice as sd
import numpy as np

print("Default device:")
print(sd.default.device)

print("\nInput devices:")
print(sd.query_devices(kind="input"))

print("\nTesting microphone...")

try:
    recording = sd.rec(
        int(3 * 16000),
        samplerate=16000,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    print("\nSUCCESS")
    print("Shape:", recording.shape)
    print("Maximum:", np.max(np.abs(recording)))

except Exception as e:
    print("\nFAILED")
    print(type(e).__name__)
    print(e)
