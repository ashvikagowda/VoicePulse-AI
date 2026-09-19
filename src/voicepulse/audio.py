import librosa
import numpy as np
import torch

from .config import (
    HOP_LENGTH,
    MIC_SAMPLE_RATE,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
    WINDOW_SAMPLES,
)


def resample_microphone_audio(audio):
    """
    Convert microphone audio from 48 kHz to
    the model's 16 kHz sampling rate.
    """

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    if audio.size == 0:
        return audio

    resampled = librosa.resample(
        audio,
        orig_sr=MIC_SAMPLE_RATE,
        target_sr=SAMPLE_RATE,
    )

    return np.asarray(
        resampled,
        dtype=np.float32,
    )


def prepare_audio_window(audio):
    """
    Ensure the model receives exactly one
    WINDOW_SECONDS audio segment.
    """

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    audio = np.nan_to_num(
        audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if len(audio) < WINDOW_SAMPLES:

        audio = np.pad(
            audio,
            (
                0,
                WINDOW_SAMPLES - len(audio),
            ),
        )

    elif len(audio) > WINDOW_SAMPLES:

        audio = audio[-WINDOW_SAMPLES:]

    return audio


def preprocess_audio(audio):
    """
    Convert waveform into a normalized Mel spectrogram tensor.

    Output shape:
        [batch, channel, mel, time]
    """

    audio = prepare_audio_window(audio)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )

    mel = librosa.power_to_db(
        mel,
        ref=np.max,
    )

    mean = np.mean(mel)
    std = np.std(mel)

    if std > 1e-8:
        mel = (mel - mean) / std
    else:
        mel = mel - mean

    tensor = torch.tensor(
        mel,
        dtype=torch.float32,
    ).unsqueeze(0).unsqueeze(0)

    return tensor
