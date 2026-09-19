import queue

import numpy as np
import sounddevice as sd


def find_macbook_microphone():
    """
    Find the MacBook Pro built-in microphone.
    """

    devices = sd.query_devices()

    for index, device in enumerate(devices):

        name = str(
            device.get("name", "")
        )

        max_input_channels = int(
            device.get(
                "max_input_channels",
                0,
            )
        )

        if max_input_channels <= 0:
            continue

        if "MacBook Pro Microphone" in name:
            return index

    for index, device in enumerate(devices):

        name = str(
            device.get("name", "")
        )

        max_input_channels = int(
            device.get(
                "max_input_channels",
                0,
            )
        )

        if max_input_channels <= 0:
            continue

        if (
            "MacBook M3" in name
            and "Microphone" in name
        ):
            return index

    raise RuntimeError(
        "MacBook M3 Pro Microphone was not found."
    )


class MicrophoneRecorder:

    def __init__(self):
        self.audio_queue = queue.Queue()
        self.stream = None
        self.device_index = None

    def audio_callback(
        self,
        indata,
        frames,
        time_info,
        status,
    ):
        if status:
            print(
                f"Audio status: {status}"
            )

        try:

            if indata.ndim > 1:
                chunk = indata[:, 0].copy()
            else:
                chunk = indata.copy()

            self.audio_queue.put(chunk)

        except Exception as exc:
            print(
                f"Audio callback error: {exc}"
            )

    def start(
        self,
        sample_rate: int,
    ):
        if self.stream is not None:

            try:
                if self.stream.active:
                    return
            except Exception:
                pass

            try:
                self.stream.stop()
            except Exception:
                pass

            try:
                self.stream.close()
            except Exception:
                pass

            self.stream = None

        self.device_index = (
            find_macbook_microphone()
        )

        sd.check_input_settings(
            device=self.device_index,
            channels=1,
            samplerate=sample_rate,
            dtype="float32",
        )

        try:

            stream = sd.InputStream(
                device=self.device_index,
                samplerate=sample_rate,
                channels=1,
                dtype=np.float32,
                callback=self.audio_callback,
                blocksize=0,
            )

            stream.start()

            self.stream = stream

        except Exception as exc:

            self.stream = None

            raise RuntimeError(
                f"Error opening MacBook microphone:\n{exc}"
            ) from exc

    def stop(self):

        if self.stream is not None:

            try:
                self.stream.stop()
            except Exception:
                pass

            try:
                self.stream.close()
            except Exception:
                pass

            self.stream = None

    def clear_queue(self):

        while True:

            try:
                self.audio_queue.get_nowait()

            except queue.Empty:
                break
