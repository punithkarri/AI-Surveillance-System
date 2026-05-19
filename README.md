---
title: HawkEye – Intelligent Security Surveillance System
emoji: 🦅
colorFrom: indigo
colorTo: red
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
---

<div align="center">

# 🦅 HawkEye – Intelligent Security Surveillance System

### *Real-Time AI-Powered Threat Detection & Monitoring*

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge&logo=opencv&logoColor=black)](https://ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-FFD21E?style=for-the-badge)](https://huggingface.co/spaces)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![Stars](https://img.shields.io/github/stars/punithkarri/AI-Surveillance-System?style=for-the-badge&color=gold)](https://github.com/punithkarri/AI-Surveillance-System/stargazers)

<br/>

> **A production-ready, browser-deployable AI surveillance application** that performs real-time human detection, threat scoring, alert generation, and analytics — powered by YOLOv8 and deployable on Hugging Face Spaces with zero hardware dependencies.

<br/>

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Hugging%20Face%20Spaces-FFD21E?style=for-the-badge)](https://huggingface.co/spaces/punithkarri/AI-Surveillance-System)

</div>

---

## 📌 Table of Contents

- [Problem Statement](#-problem-statement)
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [System Architecture](#-system-architecture)
- [Screenshots](#-screenshots)
- [Installation](#-installation)
- [Running Locally](#-running-locally)
- [Deployment](#-deployment)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [Author](#-author)
- [License](#-license)

---

## 🚨 Problem Statement

Traditional CCTV surveillance systems require 24/7 human monitoring, are expensive to operate, and often fail to flag threats in real time. Security teams face alert fatigue, video backlog, and delayed response times — resulting in missed incidents and post-incident-only evidence.

**HawkEye** solves this by deploying deep learning computer vision directly in the browser — providing automated, real-time suspicious activity detection with intelligent threat scoring, instant alerts, and exportable incident reports. No dedicated hardware. No manual watching.

---

## 🔍 Project Overview

**HawkEye** is a full-stack AI Computer Vision application built with:
- **YOLOv8** (Ultralytics) for state-of-the-art real-time object detection
- **Streamlit** for an interactive, browser-first surveillance dashboard
- **WebRTC** (`streamlit-webrtc`) for zero-plugin live browser camera streaming
- **OpenCV** for frame-level video processing and annotation
- **Plotly** for dynamic threat timelines and risk distribution charts

The system supports two deployment modes:
| Mode | Input | Camera Access | Best For |
|------|-------|--------------|---------|
| 🌐 **Cloud (Hugging Face)** | Browser WebRTC stream | Browser camera via WebRTC | Demos, portfolios, remote access |
| 🖥️ **Local** | OpenCV webcam feed | USB/built-in webcam | Development, on-premise surveillance |

Both modes share the same detection pipeline, analytics engine, alert management, and reporting system.

---

## ✨ Key Features

### 🎯 Core AI Capabilities
- **Real-Time YOLOv8 Detection** — Identifies persons, vehicles, and suspicious objects per video frame with sub-100ms inference latency
- **Intelligent Threat Scoring** — Calculates a dynamic threat percentage (0–100%) per frame based on detection confidence and object class
- **Configurable Alert Thresholds** — Trigger alerts only when the threat score exceeds a user-defined threshold (default: 60%)
- **Motion & Event Analysis** — Tracks unique suspicious events, counts detections, and classifies risk levels (LOW / MEDIUM / HIGH)

### 📊 Monitoring Dashboard
- **Live Threat Gauge** — Real-time animated threat level indicator with colour-coded status (🟢 SAFE / 🔴 HIGH THREAT)
- **Interactive Threat Timeline** — Plotly-powered rolling chart visualising threat score over consecutive frames
- **Risk Distribution Pie Chart** — Breaks down detected event categories and their risk proportions
- **Event Summary Cards** — Detailed cards for each detected class: count, max threat %, and risk classification

### 🔔 Alerting & Reporting
- **Automatic Snapshot Capture** — Auto-captures and stores annotated alert frames when threat threshold is breached
- **Browser Alarm Beep** — Web Audio API-based audible alert fires on threat detection (no file required)
- **CSV & TXT Report Export** — One-click downloadable session reports with timestamped incident logs
- **Alert Screenshot Gallery** — Scrollable grid of all captured threat snapshots with bounding boxes

### 🌐 Cloud-Native Deployment
- **Browser WebRTC Camera** — Stream live video directly from the browser — no RTSP, no IP cameras, no plugins
- **Hugging Face Spaces Support** — Fully deployable on HF Spaces (free tier) with auto-dependency resolution
- **Dual-Mode Architecture** — Auto-detects deployment environment (`SPACE_ID`) and switches between WebRTC and OpenCV
- **Upload Video Mode** — Analyse pre-recorded footage when live camera is unavailable (universal cloud fallback)

---

## 🛠️ Tech Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Language** | Python 3.9+ | Core application logic |
| **AI / Detection** | YOLOv8 (Ultralytics) | Real-time object detection model |
| **Computer Vision** | OpenCV | Frame capture, annotation, video I/O |
| **Deep Learning** | PyTorch (CPU) | YOLOv8 inference backend |
| **Web Framework** | Streamlit | Interactive surveillance dashboard UI |
| **Browser Camera** | streamlit-webrtc + aiortc | WebRTC-based live camera streaming |
| **Video Codec** | av (PyAV) | Efficient video frame decoding |
| **Analytics** | Plotly + Matplotlib | Threat timelines and risk charts |
| **Data** | Pandas + NumPy | Event aggregation and data manipulation |
| **Audio** | Web Audio API (JS) | Browser-native alert beep |
| **Config** | python-dotenv | Environment-based configuration |
| **Deployment** | Hugging Face Spaces | Cloud-hosted live demo |
| **Version Control** | Git + GitHub | Source control and CI/CD |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    HawkEye – System Architecture                │
├──────────────────────┬──────────────────────────────────────────┤
│   INPUT SOURCES      │             PROCESSING PIPELINE          │
│                      │                                          │
│  🌐 Browser Camera   │   ┌─────────────┐   ┌───────────────┐   │
│  (WebRTC Stream)  ───┼──►│ Frame Queue │──►│  YOLOv8       │   │
│                      │   │ (aiortc/av) │   │  Inference    │   │
│  📁 Uploaded Video   │   └─────────────┘   └──────┬────────┘   │
│  (MP4/AVI/MOV)    ───┼──►                          │            │
│                      │                    ┌─────────▼────────┐  │
│  🖥️ Local Webcam     │                    │  Threat Scoring  │  │
│  (OpenCV/USB)     ───┼──►                 │  & Annotation    │  │
│                      │                    └─────────┬────────┘  │
├──────────────────────┤                              │           │
│   INTELLIGENCE       │                    ┌─────────▼────────┐  │
│                      │                    │  Alert Engine    │  │
│  ⚠️ Threat Engine    │◄───────────────────│  (Threshold      │  │
│  📸 Snapshot Capture │                    │   Evaluation)    │  │
│  🔔 Alarm Trigger    │                    └──────────────────┘  │
├──────────────────────┤                                          │
│   DASHBOARD (UI)     │              SharedState                 │
│                      │            (Thread-Safe)                 │
│  📊 Live Metrics     │◄─────────────────────────────────────── │
│  📈 Threat Timeline  │                                          │
│  🗂️ Event Cards      │                                          │
│  📥 Export Reports   │                                          │
└──────────────────────┴──────────────────────────────────────────┘
```

### Dual-Mode Deployment Flow

```
App Startup
    │
    ▼
SPACE_ID detected?
    │
    ├── YES ──► Cloud Mode (Hugging Face)
    │               └── streamlit-webrtc (WebRTC)
    │               └── Browser camera stream
    │               └── SharedState (thread-safe)
    │               └── Auto-refresh analytics
    │
    └── NO ───► Local Mode (Localhost)
                    └── cv2.VideoCapture(0)
                    └── USB/built-in webcam
                    └── Optional video recording
```

---

## 📸 Screenshots

<table>
  <tr>
    <td align="center"><b>🖥️ Live Camera Dashboard</b></td>
    <td align="center"><b>🎯 Real-Time YOLO Detection</b></td>
  </tr>
  <tr>
    <td><img src="screenshots/dashboard.png" alt="Dashboard" width="100%"/></td>
    <td><img src="screenshots/live.png" alt="Live Detection" width="100%"/></td>
  </tr>
  <tr>
    <td align="center"><b>📈 Threat Analytics Timeline</b></td>
    <td align="center"><b>📥 Report Export & Alert Snapshots</b></td>
  </tr>
  <tr>
    <td><img src="screenshots/event.png" alt="Analytics" width="100%"/></td>
    <td><img src="screenshots/export.png" alt="Export" width="100%"/></td>
  </tr>
</table>

---

## ⚙️ Installation

### Prerequisites

- Python 3.9 or higher
- pip / virtualenv
- Git

### Step-by-Step Setup

**1. Clone the repository**
```bash
git clone https://github.com/punithkarri/AI-Surveillance-System.git
cd AI-Surveillance-System
```

**2. Create and activate a virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Configure environment variables**
```bash
cp .env.example .env
# Edit .env to set DEPLOYMENT_MODE and other options
```

**5. (Optional) Pre-download YOLO model**

The app will auto-download `yolov8n.pt` on first run. To use the larger, more accurate model:
```bash
# Place yolov8s.pt or yolov8n.pt in the models/ directory
# or let the app auto-download yolov8n.pt on startup
```

---

## 🚀 Running Locally

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

**To enable local webcam mode (full features):**
```bash
# In your .env file:
DEPLOYMENT_MODE=false
```

**Local mode enables:**
- ✅ Live webcam feed via OpenCV
- ✅ Local video recording (AVI output)
- ✅ Full YOLO detection pipeline
- ✅ All analytics, alerts, and reports

---

## ☁️ Deployment

### 🤗 Hugging Face Spaces (Recommended)

HawkEye is fully optimized for free-tier Hugging Face Spaces deployment.

**Automatic via GitHub Actions:**
1. Fork this repository
2. Create a new Hugging Face Space (SDK: Streamlit)
3. Connect your GitHub repo to the Space
4. Add `DEPLOYMENT_MODE=true` in Space secrets
5. Push to `main` — GitHub Actions auto-deploys

**Manual deployment:**
```bash
# Install Hugging Face CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Push to your Space
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/AI-Surveillance-System
git push space main
```

**Key behaviours in cloud mode:**
| Feature | Cloud (HF Spaces) | Local |
|---------|------------------|-------|
| Live Camera | ✅ WebRTC (browser) | ✅ OpenCV webcam |
| Video Upload | ✅ Yes | ✅ Yes |
| Video Recording | ❌ Disabled | ✅ AVI output |
| Alarm Sound | ✅ Web Audio API | ✅ pygame |
| YOLO Detection | ✅ CPU inference | ✅ CPU/GPU |
| Analytics & Reports | ✅ Full | ✅ Full |

> **Browser Camera Access:** In Hugging Face Spaces, click **START** on the Live Camera panel and allow webcam permission when prompted. HawkEye streams your browser camera via WebRTC — the video never leaves your machine; only detection results are processed server-side.

---

### 🖥️ Render Deployment

1. Create a new **Web Service** on [Render](https://render.com/)
2. Connect your GitHub repository
3. Set build command:
   ```bash
   pip install -r requirements.txt
   ```
4. Set start command:
   ```bash
   streamlit run app.py --server.port $PORT --server.address 0.0.0.0
   ```
5. Add environment variable: `DEPLOYMENT_MODE=true`

> **Note:** Render does not support local webcam access. Use Upload Video mode or browser WebRTC.

---

## 📁 Project Structure

```
AI-Surveillance-System/
│
├── app.py                          # 🚀 Streamlit entry point & UI orchestration
│
├── config/
│   └── settings.py                 # ⚙️ Centralized config, .env handling
│
├── services/
│   ├── detection.py                # 🎯 YOLOv8 model loading & frame inference
│   ├── alerts.py                   # 🔔 Alert engine, snapshot capture, alarm
│   ├── analytics.py                # 📊 Plotly chart builders (timeline, pie)
│   ├── reporting.py                # 📥 CSV & TXT report generation
│   ├── recording.py                # 🎥 Video recorder (local mode)
│   └── webrtc_processor.py         # 🌐 WebRTC video processor & SharedState
│
├── utils/
│   ├── ui.py                       # 🎨 Custom CSS, metric cards, event cards
│   └── helpers.py                  # 🛠️ Folder management, timestamp utilities
│
├── assets/                         # 🗂️ Static assets (alarm.wav, logo)
├── models/                         # 🤖 YOLO model weights storage
├── outputs/
│   ├── videos/                     # 🎬 Recorded surveillance footage
│   ├── reports/                    # 📄 Generated CSV/TXT reports
│   └── snapshots/                  # 📸 Auto-captured alert frames
├── screenshots/                    # 🖼️ App UI screenshots for README
│
├── .env.example                    # 🔐 Environment variables template
├── requirements.txt                # 📦 Python dependencies
├── packages.txt                    # 📦 System-level apt packages (HF Spaces)
├── .streamlit/config.toml          # 🎨 Streamlit theme configuration
└── .github/workflows/              # 🔄 CI/CD deployment pipeline
```

---

## 🔧 Configuration

All settings are managed via `.env` file (copy from `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `DEPLOYMENT_MODE` | `false` | `true` = cloud (WebRTC), `false` = local (cv2) |
| `THREAT_THRESHOLD` | `60` | Threat score % to trigger an alert (0–100) |
| `FRAME_WIDTH` | `640` | Camera frame width in pixels |
| `FRAME_HEIGHT` | `480` | Camera frame height in pixels |
| `ENABLE_RECORDING` | `true` | Enable/disable local video recording |
| `ENABLE_CAMERA` | `true` | Enable/disable webcam UI in local mode |
| `MAX_SCREENSHOTS` | `20` | Maximum alert snapshots per session |
| `YOLO_MODEL` | `yolov8n.pt` | YOLO model variant (`n`, `s`, `m`, `l`, `x`) |

> The app also auto-detects Hugging Face Spaces via the `SPACE_ID` environment variable, automatically enabling cloud mode even if `DEPLOYMENT_MODE` is not set.

---

## 🔮 Future Improvements

| Feature | Description | Priority |
|---------|-------------|----------|
| 🧠 **Custom Model Fine-tuning** | Train on domain-specific surveillance datasets (CCTV footage) | High |
| 🗓️ **Time-Based Analytics** | Historical charts: detections by hour, day, and weekday heatmaps | High |
| 📧 **Email / SMS Alerts** | Push notifications via SendGrid, Twilio, or Telegram on threat detection | Medium |
| 🗺️ **Multi-Camera Grid** | Monitor multiple camera feeds simultaneously in one dashboard | Medium |
| 🔐 **Authentication Layer** | Login/auth system for access-controlled surveillance dashboards | Medium |
| 🏷️ **License Plate Detection** | Extend detection to identify vehicle plates via OCR pipeline | Low |
| 🗃️ **Database Integration** | Persist events and incidents to SQLite / PostgreSQL for long-term records | Low |
| 📱 **Mobile-Responsive UI** | Optimise the Streamlit dashboard layout for tablet and mobile screens | Low |
| ⚡ **GPU Acceleration** | CUDA-enabled inference for edge deployment with NVIDIA hardware | Low |

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/your-feature-name`
3. **Commit** your changes: `git commit -m "feat: add your feature"`
4. **Push** to your branch: `git push origin feature/your-feature-name`
5. **Open a Pull Request** — describe your changes clearly

Please follow conventional commits and keep PRs focused and minimal.

---

## 👨‍💻 Author

<div align="center">

**Punith Karri**  
*AI/ML Engineer | Computer Vision Developer | Full-Stack Builder*

[![GitHub](https://img.shields.io/badge/GitHub-punithkarri-181717?style=for-the-badge&logo=github)](https://github.com/punithkarri)
[![Portfolio](https://img.shields.io/badge/Portfolio-Live-4f46e5?style=for-the-badge&logo=vercel)](https://karri-punith.vercel.app/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-punithkarri-FFD21E?style=for-the-badge)](https://huggingface.co/punithkarri)

</div>

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
MIT License — Free to use, modify, and distribute with attribution.
```

---

<div align="center">

**⭐ If HawkEye helped you, please star the repository!**

*Built with ❤️ using Python, YOLOv8, and Streamlit*

</div>
