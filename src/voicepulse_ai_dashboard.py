
from pathlib import Path

import queue

import librosa

import numpy as np

import sounddevice as sd

import streamlit as st

import torch

import torch.nn as nn


# ============================================================

# PAGE CONFIGURATION

# ============================================================

st.set_page_config(

    page_title="VoicePulse AI",

    page_icon="🔊️",

    layout="wide"

)


# ============================================================

# PROJECT CONFIGURATION

# ============================================================

# ============================================================

# PROJECT CONFIGURATION

# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (

    PROJECT_ROOT

    / "models"

    / "cnn"

    / "voicepulse_ai_cnn.pth"

)

SAMPLE_RATE = 16000

MIC_SAMPLE_RATE = 48000

UPDATE_SECONDS = 1

SMOOTHING_WINDOWS =2

WINDOW_SECONDS =3

FAKE_DETECTION_THRESHOLD = 0.70

UPDATE_SECONDS = 1

WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SECONDS

N_MELS = 64

N_FFT = 1024

HOP_LENGTH = 256

SMOOTHING_WINDOWS = 2

# ============================================================

# DEVICE

# ============================================================

if torch.backends.mps.is_available():

    DEVICE = torch.device("mps")

else:

    DEVICE = torch.device("cpu")


# ============================================================

# PERSISTENT AUDIO STATE

# ============================================================

if "audio_buffer" not in st.session_state:

    st.session_state.audio_buffer = np.array(

        [],

        dtype=np.float32

    )


if "score_history" not in st.session_state:

    st.session_state.score_history = []


if "running" not in st.session_state:

    st.session_state.running = False


if "last_prediction" not in st.session_state:

    st.session_state.last_prediction = "Waiting"


if "last_fake_probability" not in st.session_state:

    st.session_state.last_fake_probability = 0.0


if "last_real_probability" not in st.session_state:

    st.session_state.last_real_probability = 0.0


if "last_risk" not in st.session_state:

    st.session_state.last_risk = "LOW"


# ============================================================

# FIND MACBOOK MICROPHONE

# ============================================================

def find_macbook_microphone():

    devices = sd.query_devices()

    for index, device in enumerate(devices):

        name = str(

            device.get("name", "")

        )

        max_input_channels = int(

            device.get(

                "max_input_channels",

                0

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

                0

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


# ============================================================

# CNN MODEL

# ============================================================

class VoiceCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(

                1,

                16,

                kernel_size=3,

                padding=1

            ),

            nn.BatchNorm2d(16),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(

                16,

                32,

                kernel_size=3,

                padding=1

            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(

                32,

                64,

                kernel_size=3,

                padding=1

            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(

                64,

                128,

                kernel_size=3,

                padding=1

            ),

            nn.BatchNorm2d(128),

            nn.ReLU()

        )


        self.global_pool = nn.AdaptiveAvgPool2d(

            (1, 1)

        )


        # Matches voiceguard_cnn.pth

        self.classifier = nn.Sequential(

            nn.Dropout(0.4),

            nn.Linear(

                128,

                64

            ),

            nn.ReLU(),

            nn.Dropout(0.4),

            nn.Linear(

                64,

                2

            )

        )


    def forward(self, x):

        x = self.features(x)

        x = self.global_pool(x)

        x = torch.flatten(

            x,

            1

        )

        x = self.classifier(x)

        return x


# ============================================================

# LOAD MODEL

# ============================================================

@st.cache_resource

def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(

            f"Model checkpoint not found:\n{MODEL_PATH}"

        )


    model = VoiceCNN()


    checkpoint = torch.load(

        MODEL_PATH,

        map_location="cpu"

    )


    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[

                "model_state_dict"

            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[

                "state_dict"

            ]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint


    model.load_state_dict(

        state_dict

    )


    model = model.to(

        DEVICE

    )


    model.eval()


    return model


# ============================================================

# AUDIO PREPROCESSING

# ============================================================

def preprocess_audio(audio):

    audio = np.asarray(

        audio,

        dtype=np.float32

    )


    audio = np.nan_to_num(

        audio,

        nan=0.0,

        posinf=0.0,

        neginf=0.0

    )


    # Make exactly 4 seconds

    if len(audio) < WINDOW_SAMPLES:

        audio = np.pad(

            audio,

            (

                0,

                WINDOW_SAMPLES - len(audio)

            )

        )

    elif len(audio) > WINDOW_SAMPLES:

        audio = audio[

            -WINDOW_SAMPLES:

        ]


    # Create Mel spectrogram

    mel = librosa.feature.melspectrogram(

        y=audio,

        sr=SAMPLE_RATE,

        n_fft=N_FFT,

        hop_length=HOP_LENGTH,

        n_mels=N_MELS

    )


    # Convert to dB

    mel = librosa.power_to_db(

        mel,

        ref=np.max

    )


    # Normalize

    mean = np.mean(mel)

    std = np.std(mel)


    if std > 1e-8:

        mel = (

            mel - mean

        ) / std

    else:

        mel = (

            mel - mean

        )


    # [mel, time]

    #

    # becomes

    #

    # [batch, channel, mel, time]

    tensor = torch.tensor(

        mel,

        dtype=torch.float32

    ).unsqueeze(0).unsqueeze(0)


    return tensor


# ============================================================

# PREDICTION

# ============================================================

def predict_audio(

    model,

    audio

):

    input_tensor = preprocess_audio(

        audio

    )


    input_tensor = input_tensor.to(

        DEVICE

    )


    with torch.no_grad():

        logits = model(

            input_tensor

        )


        probabilities = torch.softmax(

            logits,

            dim=1

        )


    # Class 0 = Real

    # Class 1 = AI-generated

    real_probability = float(

        probabilities[0, 0].item()

    )


    fake_probability = float(

        probabilities[0, 1].item()

    )


    prediction = "AI-GENERATED"

    if fake_probability >= FAKE_DETECTION_THRESHOLD:

        prediction = "AI-GENERATED"

    else:

        prediction = "REAL"


    return (

        prediction,

        real_probability,

        fake_probability

    )


# ============================================================

# RISK LEVEL

# ============================================================

def get_risk_level(

    fake_probability

):

    if fake_probability >= 0.75:

        return "HIGH"

    elif fake_probability >= 0.50:

        return "MEDIUM"

    else:

        return "LOW"


# ============================================================

# MICROPHONE MANAGER

# ============================================================

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

        status

    ):

        if status:

            print(

                f"Audio status: {status}"

            )


        try:

            if indata.ndim > 1:

                chunk = indata[

                    :,

                    0

                ].copy()

            else:

                chunk = indata.copy()


            self.audio_queue.put(

                chunk

            )


        except Exception as e:

            print(

                f"Audio callback error: {e}"

            )


    def start(self):

        # Already running

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


        # Find MacBook microphone

        self.device_index = (

            find_macbook_microphone()

        )


        # Check requested settings

        sd.check_input_settings(

            device=self.device_index,

            channels=1,

            samplerate=MIC_SAMPLE_RATE,

            dtype="float32"

        )


        try:

            stream = sd.InputStream(

                device=self.device_index,

                samplerate=MIC_SAMPLE_RATE,

                channels=1,

                dtype=np.float32,

                callback=self.audio_callback,

                blocksize=0

            )


            stream.start()


            self.stream = stream


        except Exception as e:

            self.stream = None

            raise RuntimeError(

                f"Error opening MacBook microphone:\n{e}"

            )


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


# ============================================================

# PERSISTENT MICROPHONE RESOURCE

# ============================================================

@st.cache_resource

def get_microphone():

    return MicrophoneRecorder()


microphone = get_microphone()

audio_queue = microphone.audio_queue


# ============================================================

# START MICROPHONE

# ============================================================

def start_microphone():

    microphone.start()


# ============================================================

# STOP MICROPHONE

# ============================================================

def stop_microphone():

    microphone.stop()

    microphone.clear_queue()


    st.session_state.audio_buffer = np.array(

        [],

        dtype=np.float32

    )


    st.session_state.score_history = []


# ============================================================

# TITLE

# ============================================================

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

    "Detect potentially AI-generated or cloned speech "

    "using the VoicePulse AI CNN model."

)


# ============================================================

# SIDEBAR

# ============================================================

with st.sidebar:

    st.header(

        "System Information"

    )


    st.write(

        "**Model:** `VoicePulse AI `"

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

        "**Microphone:** `MacBook M3 Pro Microphone`"

    )


    st.write(

        f"**PyTorch device:** `{DEVICE}`"

    )


    st.divider()


    st.write(

        "The MacBook M3 Pro microphone captures at "

        "48 kHz and the audio is resampled to 16 kHz "

        "before CNN analysis."

    )


# ============================================================

# LOAD MODEL

# ============================================================

try:

    model = load_model()


    st.success(

        "CNN model loaded successfully."

    )


except Exception as e:

    st.error(

        "Failed to load CNN model."

    )


    st.code(

        str(e)

    )


    st.stop()


# ============================================================

# CONTROL BUTTONS

# ============================================================

col1, col2 = st.columns(2)


with col1:

    start_clicked = st.button(

        "🎙️ Start Microphone",

        use_container_width=True

    )


with col2:

    stop_clicked = st.button(

        "⏹️ Stop Microphone",

        use_container_width=True

    )


# ============================================================

# START BUTTON

# ============================================================

if start_clicked:

    try:

        start_microphone()

        st.session_state.running = True

        st.session_state.last_prediction = (

            "Listening"

        )

        st.session_state.last_risk = "LOW"

        st.success(

            "MacBook Pro microphone started."

        )


    except Exception as e:

        st.session_state.running = False

        st.error(

            "Could not start MacBook microphone."

        )

        st.code(

            str(e)

        )


# ============================================================

# STOP BUTTON

# ============================================================

if stop_clicked:

    stop_microphone()

    st.session_state.running = False

    st.session_state.last_prediction = (

        "Stopped"

    )

    st.session_state.last_fake_probability = 0.0

    st.session_state.last_real_probability = 0.0

    st.session_state.last_risk = "LOW"


# ============================================================

# LIVE DETECTION

# ============================================================

@st.fragment(

    run_every="0.5s"

)

def live_detection():

    # ========================================================

    # STATUS CARDS

    # ========================================================

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

        current_prediction = (

            st.session_state.last_prediction

        )


        if current_prediction == "AI-GENERATED":

            st.error(

                "⚠️ AI-GENERATED"

            )

        elif current_prediction == "REAL":

            st.success(

                "✅ REAL"

            )

        elif current_prediction == "Listening":

            st.info(

                "🎙️ LISTENING"

            )

        elif current_prediction == "Stopped":

            st.warning(

                "⏹️ STOPPED"

            )

        else:

            st.info(

                "WAITING"

            )


    with status_col3:

        current_risk = (

            st.session_state.last_risk

        )


        if current_risk == "HIGH":

            st.error(

                "🔴 HIGH RISK"

            )

        elif current_risk == "MEDIUM":

            st.warning(

                "🟠 MEDIUM RISK"

            )

        else:

            st.success(

                "🟢 LOW RISK"

            )


    # ========================================================

    # CURRENT ANALYSIS

    #

    # KEEP THIS ABOVE THE VARIABLE-HEIGHT LIVE AREA.

    # ========================================================

    st.divider()

    st.subheader(

        "Current Analysis"

    )


    result_col1, result_col2 = (

        st.columns(2)

    )


    with result_col1:

        st.metric(

            "AI-Generated Probability",

            f"{st.session_state.last_fake_probability * 100:.2f}%"

        )


    with result_col2:

        st.metric(

            "Real Probability",

            f"{st.session_state.last_real_probability * 100:.2f}%"

        )


    # ========================================================

    # LIVE DETECTION AREA

    # ========================================================

    st.divider()

    progress = st.progress(

        0

    )


    message = st.empty()


    detection_box = st.empty()


    probability_box = st.empty()


    # ========================================================

    # NOT RUNNING

    # ========================================================

    if not st.session_state.running:

        progress.progress(

            0

        )


        message.info(

            "Microphone is stopped."

        )


        detection_box.empty()

        probability_box.empty()

        return


    # ========================================================

    # RUNNING

    # ========================================================

    message.info(

        "🎙️ Listening for speech..."

    )


    # ========================================================

    # COLLECT NEW AUDIO

    # ========================================================

    new_audio = []


    while True:

        try:

            chunk = (

                audio_queue.get_nowait()

            )


            new_audio.append(

                chunk

            )


        except queue.Empty:

            break


    # ========================================================

    # PROCESS NEW AUDIO

    # ========================================================

    if new_audio:

        new_audio = np.concatenate(

            new_audio

        )


        # ====================================================

        # 48 kHz MICROPHONE

        #

        #          ↓

        #

        #       RESAMPLE

        #

        #          ↓

        #

        # 16 kHz MODEL AUDIO

        # ====================================================

        new_audio = librosa.resample(

            new_audio,

            orig_sr=MIC_SAMPLE_RATE,

            target_sr=SAMPLE_RATE

        )


        new_audio = np.asarray(

            new_audio,

            dtype=np.float32

        )


        audio_buffer = np.concatenate(

            [

                st.session_state.audio_buffer,

                new_audio

            ]

        )


        # Keep only last 10 seconds

        max_buffer_samples = (

            SAMPLE_RATE * 10

        )


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


    # ========================================================

    # WAIT FOR 4 SECONDS

    # ========================================================

    if len(audio_buffer) < WINDOW_SAMPLES:

        seconds_available = (

            len(audio_buffer)

            / SAMPLE_RATE

        )


        progress_value = min(

            seconds_available

            / WINDOW_SECONDS,

            1.0

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


    # ========================================================

    # TAKE LAST 4 SECONDS

    # ========================================================

    current_audio = audio_buffer[

        -WINDOW_SAMPLES:

    ]


    progress.progress(

        1.0

    )


    # ========================================================

    # MODEL PREDICTION

    # ========================================================

    try:

        (

            prediction,

            real_probability,

            fake_probability

        ) = predict_audio(

            model,

            current_audio

        )


    except Exception as e:

        detection_box.error(

            "Model prediction failed."

        )


        probability_box.code(

            str(e)

        )

        return


    # ========================================================

    # SMOOTHING

    # ========================================================

    score_history = (

        st.session_state.score_history

    )


    score_history.append(

        fake_probability

    )


    if len(score_history) > SMOOTHING_WINDOWS:

        score_history = score_history[

            -SMOOTHING_WINDOWS:

        ]


    st.session_state.score_history = (

        score_history

    )


    smoothed_fake_probability = float(

        np.mean(

            score_history

        )

    )


    # ========================================================

    # FINAL PREDICTION

    # ========================================================


    final_prediction = (

        "AI-GENERATED"

    )

    if smoothed_fake_probability >= FAKE_DETECTION_THRESHOLD:

        final_prediction = (

            "AI-GENERATED"

        )

    else:

        final_prediction = (

            "REAL"

        )


    final_risk = get_risk_level(

        smoothed_fake_probability

    )


    # ========================================================

    # SAVE RESULTS

    # ========================================================

    st.session_state.last_prediction = (

        final_prediction

    )


    st.session_state.last_fake_probability = (

        smoothed_fake_probability

    )


    st.session_state.last_real_probability = (

        1.0 - smoothed_fake_probability

    )


    st.session_state.last_risk = (

        final_risk

    )


    # ========================================================

    # DISPLAY DETECTION

    # ========================================================

    if final_prediction == "AI-GENERATED":

        detection_box.error(

            f"⚠️ POSSIBLE AI-GENERATED VOICE\n\n"

            f"Fake probability: "

            f"{smoothed_fake_probability * 100:.2f}%"

        )


    else:

        detection_box.success(

            f"✅ VOICE CLASSIFIED AS REAL\n\n"

            f"Fake probability: "

            f"{smoothed_fake_probability * 100:.2f}%"

        )


    # ========================================================

    # DISPLAY PROBABILITIES

    # ========================================================

    probability_box.write(

        f"**Real probability:** "

        f"{(1.0 - smoothed_fake_probability) * 100:.2f}%"

    )


    probability_box.write(

        f"**AI-generated probability:** "

        f"{smoothed_fake_probability * 100:.2f}%"

    )


    probability_box.write(

        f"**Risk level:** "

        f"{final_risk}"

    )


# ============================================================

# RUN LIVE DETECTION

# ============================================================

live_detection()


# ============================================================

# INFORMATION

# ============================================================

st.divider()


st.caption(

    "VoicePulse AI uses a Custom trained CNN classifier to "

    "estimate whether the current 4-second speech segment "

    "resembles real or AI-generated speech. "

    "The displayed probability is a model estimate and "

    "should not be treated as definitive proof of voice cloning."

)
