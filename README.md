# VoicePulse AI

Real-time AI-generated voice detection system designed to identify synthetic and cloned speech from live microphone input.

## Overview

VoicePulse AI analyzes short audio windows and estimates whether the speech is likely real or AI-generated.

The project combines:

- CNN-based audio classification
- ASVspoof2019 training data
- ASVspoof2021 external evaluation
- Local-domain fine-tuning
- Recording-level cross-validation
- Threshold calibration
- Real-time microphone inference
- Streamlit dashboard visualization

## System Pipeline

```text
Microphone
    ↓
Audio Capture
    ↓
4-second Sliding Window
    ↓
Audio Preprocessing
    ↓
CNN Classifier
    ↓
AI Probability
    ↓
Threshold Decision
    ↓
REAL / AI-GENERATED
    ↓
Streamlit Dashboard

## Current Decision Logic

The live system uses a calibrated decision threshold:

```text
AI probability >= 0.85
        ↓
AI-GENERATED

AI probability < 0.85
        ↓
REAL
```

Live processing configuration:

```text
Window size: 4 seconds
Update interval: 1 second
```

## Model Development

The development pipeline includes:

1. CNN training using ASVspoof2019 data
2. External evaluation using ASVspoof2021 data
3. Local recording evaluation
4. Local-domain fine-tuning
5. Recording-level 5-fold cross-validation
6. Threshold analysis and calibration
7. Real-time microphone inference

## Repository Structure

```text
voicepulse_ai/
│
├── src/
│   ├── voicepulse_ai_dashboard.py
│   ├── train_cnn.py
│   ├── train_cnn_asvspoof2019.py
│   ├── finetune_cnn_local.py
│   ├── local_cross_validation.py
│   ├── evaluate_asvspoof.py
│   ├── evaluate_cnn_asvspoof.py
│   ├── evaluate_cnn_recording.py
│   ├── evaluate_finetuned_recording.py
│   ├── evaluate_local_domain.py
│   ├── evaluate_fusion.py
│   └── ...
│
├── test_mic.py
├── test_stream.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset and Model Files

Audio recordings, datasets, and trained model checkpoints are intentionally excluded from this repository.

The repository contains the implementation and evaluation pipeline without distributing the underlying audio recordings or trained model weights.

## Running the Dashboard

Create and activate a Python virtual environment, install the dependencies, then run:

```bash
streamlit run src/voicepulse_ai_dashboard.py
```

The dashboard uses the MacBook microphone for live inference.

## Project Status

* CNN training: complete
* ASVspoof2019 evaluation: complete
* ASVspoof2021 external evaluation: complete
* Local-domain fine-tuning: complete
* Recording-level validation: complete
* Threshold calibration: complete
* Real-time microphone inference: complete
* Streamlit dashboard: complete

## Future Scope

* Speaker verification
* Mobile deployment
* Telephony and VoIP integration
* Larger cross-speaker evaluation
* Model compression and edge deployment

## Author

Ashvika Gowda

B.E. — Internet of Things, Cybersecurity with Blockchain
