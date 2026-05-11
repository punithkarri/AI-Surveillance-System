---
title: AI Surveillance System
emoji: 🎥
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
---

# AI Surveillance System

A modular AI deployment-ready surveillance app with Streamlit, YOLO detection, threat scoring, alert management, recording, reporting, and analytics.

## Project Structure

- `app.py` - Streamlit entrypoint and UI orchestration.
- `config/settings.py` - centralized configuration and environment handling.
- `services/` - core modules for detection, alerts, reporting, analytics, and recording.
- `utils/` - UI helpers, folder management, and utility functions.
- `assets/` - shared static assets (alarm sound, logo placeholder).
- `models/` - YOLO model file storage.
- `outputs/` - generated files for videos, reports, and snapshots.
- `screenshots/` - legacy screenshot storage area.

## Installation

1. Create a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Place `yolov8n.pt` in `models/` or let the application auto-download the model.
4. Copy `.env.example` to `.env` and customize deployment settings.

## Running Locally

From the project root:

```bash
cd AI-Surveillance-System
streamlit run app.py
```

If you want local webcam and recording enabled, set:

```bash
DEPLOYMENT_MODE=false
```

## Deployment Mode

Set `DEPLOYMENT_MODE=true` in `.env` to enable cloud-safe mode.
When cloud mode is active:

- Local webcam UI is hidden.
- Recording is disabled.
- Upload Video mode remains available.
- Reports, analytics, screenshots, and threat detection remain functional.

## Hugging Face Spaces

1. Create a new Space using the `streamlit` SDK.
2. Add this repository files, including `requirements.txt`, `.env.example`, and `.streamlit/config.toml`.
3. Set `DEPLOYMENT_MODE=true` in the Space secrets or `.env` file.
4. Ensure the `models/` folder contains `yolov8n.pt` or allow the model to auto-download.
5. Start the app with the default Streamlit web app launcher.

> Note: Spaces does not support webcam access, so Upload Video is the recommended mode.

## Render

1. Create a new Web Service on Render.
2. Set the build command:

```bash
pip install -r requirements.txt
```

3. Set the start command:

```bash
streamlit run app.py --server.port $PORT
```

4. Add environment variables in Render:

- `DEPLOYMENT_MODE=true`
- Other `.env` values as needed.

> Note: Render does not support local webcam access in cloud deployment.

## Output Storage

Generated files are stored in:

- `outputs/videos/`
- `outputs/reports/`
- `outputs/snapshots/`

## Webcam Limitations

- Local webcam mode only works on a machine with an attached webcam.
- In cloud deployment, webcam functionality is disabled and the app falls back to Upload Video.

## Requirements

- Streamlit UI preserved
- Live camera and uploaded video modes
- Alarm sound and screenshot capture
- CSV/TXT report export
- Analytics charts and event summaries
- Centralized config with `.env` support

## Compatibility

Designed for Streamlit Cloud, Render, and Hugging Face Spaces.
