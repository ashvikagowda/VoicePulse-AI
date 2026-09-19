import numpy as np
import torch

from .audio import preprocess_audio
from .config import DEVICE
from .model import VoiceCNN


def predict_audio(
    model: VoiceCNN,
    audio,
):
    """
    Run CNN inference on one audio window.

    Returns:
        real_probability
        fake_probability
    """

    input_tensor = preprocess_audio(audio)
    input_tensor = input_tensor.to(DEVICE)

    with torch.no_grad():

        logits = model(input_tensor)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

    real_probability = float(
        probabilities[0, 0].item()
    )

    fake_probability = float(
        probabilities[0, 1].item()
    )

    return (
        real_probability,
        fake_probability,
    )


class ProbabilitySmoother:
    """
    Moving-average smoothing for live AI probability.
    """

    def __init__(self, window_size: int):
        self.window_size = window_size
        self.history = []

    def update(self, probability: float) -> float:

        self.history.append(
            float(probability)
        )

        if len(self.history) > self.window_size:
            self.history = self.history[
                -self.window_size:
            ]

        return float(
            np.mean(self.history)
        )

    def reset(self):
        self.history.clear()
