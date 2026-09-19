from .config import FAKE_DETECTION_THRESHOLD


def classify_probability(
    fake_probability: float,
) -> str:
    """
    Convert AI-generation probability into
    the displayed detection decision.
    """

    if fake_probability >= FAKE_DETECTION_THRESHOLD:
        return "AI-GENERATED"

    return "REAL"


def get_risk_level(
    fake_probability: float,
) -> str:
    """
    Risk interpretation based on the same
    AI-generation probability.
    """

    if fake_probability >= FAKE_DETECTION_THRESHOLD:
        return "HIGH"

    if fake_probability >= 0.50:
        return "MEDIUM"

    return "LOW"
