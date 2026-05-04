# ASL Hand Sign Recognition

A Streamlit-based ASL hand sign recognition app that uses MediaPipe and an ensemble of TensorFlow models for real-time webcam and image input.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Language: Python](https://img.shields.io/badge/language-Python-3.11-blue)](https://www.python.org/)
[![GitHub Repo Size](https://img.shields.io/github/repo-size/ali-ezz/asl-hand-sign-recognition)](https://github.com/ali-ezz/asl-hand-sign-recognition)

## Overview

This repository contains an ASL alphabet recognition system built on MediaPipe hand detection, OpenCV preprocessing, and a multi-model TensorFlow ensemble for reliable real-time inference.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Features

- Real-time ASL alphabet recognition using webcam input
- MediaPipe hand tracking with robust fallback handling
- Ensemble inference with multiple TensorFlow models
- Image upload support and confidence scoring
- Designed for accuracy and usability with a lightweight UI

## Requirements

- Python 3.11+ recommended
- `requirements.txt` includes:
  - `streamlit`
  - `streamlit-webrtc`
  - `tensorflow-cpu`
  - `opencv-python-headless`
  - `mediapipe`
  - `av`
  - `Pillow`
  - `numpy`
  - `huggingface-hub`

## Usage

1. Create and activate a Python virtual environment.
2. Install dependencies from `requirements.txt`.
3. Launch the app with `streamlit run app.py`.
4. Open the local Streamlit URL displayed in the terminal.

## Repository layout

- `app.py` - main Streamlit application entrypoint
- `requirements.txt` - core Python dependencies
- `packages.txt` - duplicate dependency list included for compatibility
- `class_names.json` - ASL alphabet label definitions
- `models/` - model artifacts and helper files
- `all-train-data/` - training dataset notes and notebooks
- `cheeklist-Ahmed_abobakr/` - project checklists and documentation
- `digrams-Ahmed_abobakr/` - architecture and process diagrams
- `improvmint-ahmed_sabary/` - improved ASL engine implementation and support files

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for issue and pull request guidelines.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Policies

- Contributor behavior is defined in [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
- Security reporting is described in [SECURITY.md](SECURITY.md).
