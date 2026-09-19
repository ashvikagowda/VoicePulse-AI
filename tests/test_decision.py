import unittest

from voicepulse.config import FAKE_DETECTION_THRESHOLD
from voicepulse.decision import (
    classify_probability,
    get_risk_level,
)


class TestDecisionLogic(unittest.TestCase):

    def test_below_threshold_is_real(self):
        probability = FAKE_DETECTION_THRESHOLD - 0.01

        self.assertEqual(
            classify_probability(probability),
            "REAL",
        )

    def test_at_threshold_is_ai_generated(self):
        probability = FAKE_DETECTION_THRESHOLD

        self.assertEqual(
            classify_probability(probability),
            "AI-GENERATED",
        )

    def test_above_threshold_is_ai_generated(self):
        probability = 0.95

        self.assertEqual(
            classify_probability(probability),
            "AI-GENERATED",
        )

    def test_high_risk_at_threshold(self):
        probability = FAKE_DETECTION_THRESHOLD

        self.assertEqual(
            get_risk_level(probability),
            "HIGH",
        )


if __name__ == "__main__":
    unittest.main()
