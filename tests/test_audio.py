import unittest

import numpy as np

from voicepulse.audio import prepare_audio_window
from voicepulse.config import WINDOW_SAMPLES


class TestAudioProcessing(unittest.TestCase):

    def test_short_audio_is_padded(self):
        audio = np.zeros(
            WINDOW_SAMPLES // 2,
            dtype=np.float32,
        )

        result = prepare_audio_window(audio)

        self.assertEqual(
            len(result),
            WINDOW_SAMPLES,
        )

    def test_long_audio_is_trimmed(self):
        audio = np.zeros(
            WINDOW_SAMPLES * 2,
            dtype=np.float32,
        )

        result = prepare_audio_window(audio)

        self.assertEqual(
            len(result),
            WINDOW_SAMPLES,
        )

    def test_exact_window_is_unchanged(self):
        audio = np.ones(
            WINDOW_SAMPLES,
            dtype=np.float32,
        )

        result = prepare_audio_window(audio)

        self.assertEqual(
            len(result),
            WINDOW_SAMPLES,
        )

        np.testing.assert_array_equal(
            result,
            audio,
        )


if __name__ == "__main__":
    unittest.main()
