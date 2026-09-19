from pathlib import Path
import subprocess
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DIRECTORIES = [
    PROJECT_ROOT / "data" / "real_wav",
    PROJECT_ROOT / "data" / "fake_wav",
]


print("\n========================================")
print("VOICEPULSE AI AUDIO FILE CHECK")
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

        soundfile_ok = True

        ffmpeg_ok = True

        soundfile_error = ""

        ffmpeg_error = ""


        # -----------------------------------------------
        # Check with SoundFile
        # -----------------------------------------------

        try:

            info = sf.info(path)

            if info.frames <= 0:

                soundfile_ok = False

                soundfile_error = (
                    "File contains no audio frames."
                )

        except Exception as e:

            soundfile_ok = False

            soundfile_error = str(e)


        # -----------------------------------------------
        # Check with FFmpeg
        # -----------------------------------------------

        try:

            result = subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-i",
                    str(path),
                    "-f",
                    "null",
                    "-"
                ],
                capture_output=True,
                text=True
            )


            if result.returncode != 0:

                ffmpeg_ok = False

                ffmpeg_error = (
                    result.stderr.strip()
                )

        except FileNotFoundError:

            print(
                "\nWARNING: ffmpeg was not found."
            )

            ffmpeg_ok = True


        # -----------------------------------------------
        # Report
        # -----------------------------------------------

        if not soundfile_ok or not ffmpeg_ok:

            print("\n❌ BAD FILE")

            print(
                f"File: {path.name}"
            )

            print(
                f"Path: {path}"
            )


            if not soundfile_ok:

                print(
                    f"SoundFile error: "
                    f"{soundfile_error}"
                )


            if not ffmpeg_ok:

                print(
                    f"FFmpeg error: "
                    f"{ffmpeg_error}"
                )


            bad_files.append(
                path
            )

        else:

            print(
                f"✅ {path.name}"
            )


print("\n========================================")
print("SUMMARY")
print("========================================")

print(
    f"Bad files found: {len(bad_files)}"
)


if bad_files:

    print("\nFiles that need repair:")

    for path in bad_files:

        print(
            path
        )

else:

    print(
        "All WAV files passed both checks."
    )
