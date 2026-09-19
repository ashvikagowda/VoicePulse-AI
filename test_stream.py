import sounddevice as sd
import numpy as np
import time

print("Input devices:")
print(sd.query_devices())

print("\nDefault input:")
print(sd.query_devices(kind="input"))

print("\nChecking 48 kHz...")
sd.check_input_settings(
    device=0,
    channels=1,
    samplerate=48000,
    dtype="float32"
)

print("48 kHz settings OK")

print("\nOpening InputStream...")

stream = sd.InputStream(
    device=0,
    samplerate=48000,
    channels=1,
    dtype=np.float32,
    blocksize=0
)

stream.start()

print("INPUT STREAM STARTED")
print("Recording for 5 seconds...")

time.sleep(5)

stream.stop()
stream.close()

print("INPUT STREAM CLOSED")
print("SUCCESS")
