import numpy as np
import streamlit as st

from .audio import resample_microphone_audio
from .config import (
    DEVICE,
    MIC_SAMPLE_RATE,
    SAMPLE_RATE,
    SMOOTHING_WINDOWS,
    UPDATE_SECONDS,
    WINDOW_SECONDS,
    WINDOW_SAMPLES,
    FAKE_DETECTION_THRESHOLD,
)
from .decision import (
    classify_probability,
    get_risk_level,
)
from .inference import (
    ProbabilitySmoother,
    predict_audio,
)
from .microphone import MicrophoneRecorder
from .model import load_model


def initialize_session_state():

    defaults = {
        "audio_buffer": np.array(
            [],
            dtype=np.float32,
        ),
        "running": False,
        "last_prediction": "Waiting",
        "last_fake_probability": 0.0,
        "last_real_probability": 0.0,
        "last_risk": "LOW",
    }

    for key, value in defaults.items():

        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource
def get_model():
    return load_model()


@st.cache_resource
def get_microphone():
    return MicrophoneRecorder()


@st.cache_resource
def get_smoother():
    return ProbabilitySmoother(
        SMOOTHING_WINDOWS
    )


def stop_microphone(microphone):

    microphone.stop()
    microphone.clear_queue()

    st.session_state.audio_buffer = np.array(
        [],
        dtype=np.float32,
    )

    get_smoother().reset()


def render_live_detection(
    model,
    microphone,
):

    status_col1, status_col2, status_col3 = (
        st.columns(3)
    )

    with status_col1:

        if st.session_state.running:
            st.success(
                "🟢 MICROPHONE ACTIVE"
            )
        else:
            st.warning(
                "⚪ MICROPHONE STOPPED"
            )

    with status_col2:

        prediction = (
            st.session_state.last_prediction
        )

        if prediction == "AI-GENERATED":
            st.error("⚠️ AI-GENERATED")

        elif prediction == "REAL":
            st.success("✅ REAL")

        elif prediction == "Listening":
            st.info("🎙️ LISTENING")

        elif prediction == "Stopped":
            st.warning("⏹️ STOPPED")

        else:
            st.info("WAITING")

    with status_col3:

        risk = st.session_state.last_risk

        if risk == "HIGH":
            st.error("🔴 HIGH RISK")

        elif risk == "MEDIUM":
            st.warning("🟠 MEDIUM RISK")

        else:
            st.success("🟢 LOW RISK")

    st.divider()

    st.subheader("Current Analysis")

    result_col1, result_col2 = (
        st.columns(2)
    )

    with result_col1:
        st.metric(
            "AI-Generated Probability",
            f"{st.session_state.last_fake_probability * 100:.2f}%",
        )

    with result_col2:
        st.metric(
            "Real Probability",
            f"{st.session_state.last_real_probability * 100:.2f}%",
        )

    st.divider()

    progress = st.progress(0)
    message = st.empty()
    detection_box = st.empty()
    probability_box = st.empty()

    if not st.session_state.running:

        progress.progress(0)

        message.info(
            "Microphone is stopped."
        )

        detection_box.empty()
        probability_box.empty()

        return

    message.info(
        "🎙️ Listening for speech..."
    )

    new_audio = []

    while True:

        try:
            chunk = (
                microphone.audio_queue.get_nowait()
            )

            new_audio.append(chunk)

        except Exception:
            break

    if new_audio:

        new_audio = np.concatenate(
            new_audio
        )

        new_audio = (
            resample_microphone_audio(
                new_audio
            )
        )

        audio_buffer = np.concatenate(
            [
                st.session_state.audio_buffer,
                new_audio,
            ]
        )

        max_buffer_samples = SAMPLE_RATE * 10

        if len(audio_buffer) > max_buffer_samples:

            audio_buffer = audio_buffer[
                -max_buffer_samples:
            ]

        st.session_state.audio_buffer = (
            audio_buffer
        )

    else:

        audio_buffer = (
            st.session_state.audio_buffer
        )

    if len(audio_buffer) < WINDOW_SAMPLES:

        seconds_available = (
            len(audio_buffer)
            / SAMPLE_RATE
        )

        progress_value = min(
            seconds_available / WINDOW_SECONDS,
            1.0,
        )

        progress.progress(
            progress_value
        )

        message.info(
            f"Collecting audio... "
            f"{seconds_available:.1f} / "
            f"{WINDOW_SECONDS} seconds"
        )

        detection_box.info(
            "Waiting for enough speech "
            "for analysis."
        )

        probability_box.empty()

        return

    current_audio = audio_buffer[
        -WINDOW_SAMPLES:
    ]

    progress.progress(1.0)

    try:

        (
            _,
            fake_probability,
        ) = predict_audio(
            model,
            current_audio,
        )

    except Exception as exc:

        detection_box.error(
            "Model prediction failed."
        )

        probability_box.code(
            str(exc)
        )

        return

    smoother = get_smoother()

    smoothed_probability = (
        smoother.update(
            fake_probability
        )
    )

    prediction = classify_probability(
        smoothed_probability
    )

    risk = get_risk_level(
        smoothed_probability
    )

    st.session_state.last_prediction = (
        prediction
    )

    st.session_state.last_fake_probability = (
        smoothed_probability
    )

    st.session_state.last_real_probability = (
        1.0 - smoothed_probability
    )

    st.session_state.last_risk = risk

    if prediction == "AI-GENERATED":

        detection_box.error(
            f"⚠️ POSSIBLE AI-GENERATED VOICE\n\n"
            f"AI probability: "
            f"{smoothed_probability * 100:.2f}%"
        )

    else:

        detection_box.success(
            f"✅ VOICE CLASSIFIED AS REAL\n\n"
            f"AI probability: "
            f"{smoothed_probability * 100:.2f}%"
        )

    probability_box.write(
        f"**Real probability:** "
        f"{(1.0 - smoothed_probability) * 100:.2f}%"
    )

    probability_box.write(
        f"**AI-generated probability:** "
        f"{smoothed_probability * 100:.2f}%"
    )

    probability_box.write(
        f"**Decision threshold:** "
        f"{FAKE_DETECTION_THRESHOLD * 100:.0f}%"
    )

    probability_box.write(
        f"**Risk level:** {risk}"
    )


def run_dashboard():

    st.set_page_config(
        page_title="VoicePulse AI",
        page_icon="🔊",
        layout="wide",
    )

    initialize_session_state()

    st.title(
        "🔉 VoicePulse AI"
    )

    st.subheader(
        "AI-Powered Real-Time Voice Cloning Detection"
    )

    st.write(
        "**Model:** `VoicePulse AI CNN`"
    )

    st.write(
        "Detect potentially AI-generated or cloned "
        "speech using the VoicePulse AI CNN model."
    )

    with st.sidebar:

        st.header(
            "System Information"
        )

        st.write(
            "**Model:** `VoicePulse AI CNN`"
        )

        st.write(
            f"**Model sample rate:** "
            f"`{SAMPLE_RATE} Hz`"
        )

        st.write(
            f"**Microphone sample rate:** "
            f"`{MIC_SAMPLE_RATE} Hz`"
        )

        st.write(
            f"**Analysis window:** "
            f"`{WINDOW_SECONDS} seconds`"
        )

        st.write(
            f"**Update interval:** "
            f"`{UPDATE_SECONDS} second`"
        )

        st.write(
            f"**Detection threshold:** "
            f"`{FAKE_DETECTION_THRESHOLD * 100:.0f}%`"
        )

        st.write(
            "**Microphone:** "
            "`MacBook M3 Pro Microphone`"
        )

        st.write(
            f"**PyTorch device:** `{DEVICE}`"
        )

        st.divider()

        st.write(
            "The MacBook microphone captures at "
            "48 kHz and audio is resampled to "
            "16 kHz before CNN analysis."
        )

    try:

        model = get_model()

        st.success(
            "CNN model loaded successfully."
        )

    except Exception as exc:

        st.error(
            "Failed to load CNN model."
        )

        st.code(str(exc))

        st.stop()

    microphone = get_microphone()

    col1, col2 = st.columns(2)

    with col1:

        start_clicked = st.button(
            "🎙️ Start Microphone",
            use_container_width=True,
        )

    with col2:

        stop_clicked = st.button(
            "⏹️ Stop Microphone",
            use_container_width=True,
        )

    if start_clicked:

        try:

            microphone.start(
                MIC_SAMPLE_RATE
            )

            st.session_state.running = True
            st.session_state.last_prediction = (
                "Listening"
            )
            st.session_state.last_risk = "LOW"

            st.success(
                "MacBook Pro microphone started."
            )

        except Exception as exc:

            st.session_state.running = False

            st.error(
                "Could not start MacBook microphone."
            )

            st.code(str(exc))

    if stop_clicked:

        stop_microphone(
            microphone
        )

        st.session_state.running = False

        st.session_state.last_prediction = (
            "Stopped"
        )

        st.session_state.last_fake_probability = 0.0
        st.session_state.last_real_probability = 0.0
        st.session_state.last_risk = "LOW"

    @st.fragment(
        run_every=f"{UPDATE_SECONDS}s"
    )
    def live_fragment():

        render_live_detection(
            model,
            microphone,
        )

    live_fragment()

    st.divider()

    st.caption(
        "VoicePulse AI uses a custom-trained CNN classifier "
        "to estimate whether the current 4-second speech "
        "segment resembles real or AI-generated speech. "
        "The displayed probability is a model estimate "
        "and should not be treated as definitive proof "
        "of voice cloning."
    )
