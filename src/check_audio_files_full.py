from pathlib import Path

import soundfile as sf
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DIRECTORIES = [
    PROJECT_ROOT / "data" / "real_wav",
    PROJECT_ROOT / "data" / "fake_wav",
]


print("\n========================================")
print("VOICEPULSE AI FULL AUDIO DECODE CHECK")
print("========================================")


bad_files = []


for directory in DIRECTORIES:

    print(f"\nChecking: {directory}")

    files = sorted(
        directory.glob("*.wav")
    )

    print(
        f"Found {len(files)} WAV files"
    )


    for path in files:

        try:

            # IMPORTANT:
            # This actually reads the entire audio file.

            audio, sample_rate = sf.read(
                path,
                dtype="float32",
                always_2d=False
            )


            audio = np.asarray(
                audio
            )


            if audio.size == 0:

                raise ValueError(
                    "Audio file contains zero samples."
                )


            if not np.isfinite(audio).all():

                raise ValueError(
                    "Audio contains NaN or infinite values."
                )


            duration = (
                audio.shape[0]
                / sample_rate
            )


            print(
                f"✅ {path.name} "
                f"| {sample_rate} Hz "
                f"| {duration:.2f} sec"
            )


        except Exception as e:

            print(
                f"\n❌ BAD FILE: {path.name}"
            )

            print(
                f"Error: {e}"
            )

            bad_files.append(
                path
            )


print("\n========================================")
print("SUMMARY")
print("========================================")

print(
    f"Bad files found: {len(bad_files)}"
)


if bad_files:

    print("\nFiles that failed full decoding:")

    for path in bad_files:

        print(path)

else:

    print(
        "All WAV files passed full decoding."
    )
