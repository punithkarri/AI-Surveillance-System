import tempfile
import time
from pathlib import Path

import cv2
import streamlit as st

from config.settings import (
    ALARM_SOUND_FILE,
    DEPLOYMENT_MODE,
    ENABLE_CAMERA,
    ENABLE_RECORDING,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MAX_SCREENSHOTS,
    REPORT_OUTPUT_DIR,
    SNAPSHOT_OUTPUT_DIR,
    THREAT_THRESHOLD,
    VIDEO_OUTPUT_DIR,
)
from services.analytics import build_risk_pie, build_threat_timeline
from services.alerts import capture_snapshot, play_alarm, should_alert
from services.detection import load_model, process_frame
from services.recording import VideoRecorder
from services.reporting import (
    export_csv_report,
    export_txt_report,
    save_report_files,
    summarize_events,
)
from utils.helpers import ensure_folder, get_timestamp_string
from utils.ui import (
    inject_custom_css,
    render_event_cards,
    render_metric_cards,
    render_screenshot_grid,
)

# ── WebRTC imports (cloud-safe: only used in DEPLOYMENT_MODE) ─────────────────
WEBRTC_AVAILABLE = False
try:
    from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
    from services.webrtc_processor import SurveillanceProcessor, SharedState
    WEBRTC_AVAILABLE = True
except ImportError:
    pass  # streamlit-webrtc not installed (local dev without it) — falls back to cv2

# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="AI Surveillance System", page_icon="🎥", layout="wide")
inject_custom_css()

# ── WebRTC STUN configuration ─────────────────────────────────────────────────
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
) if WEBRTC_AVAILABLE else None


@st.cache_resource(show_spinner=False)
def get_model():
    return load_model()


def init_session_state():
    defaults = {
        "run_camera": False,
        "alarm_enabled": True,
        "alert_count": 0,
        "suspicious_events": [],
        "threat_history": [],
        "screenshots": [],
        "video_path": None,
        "camera_active": False,
        "last_threat_score": 0,
        "analysis_complete": False,
        "upload_results": False,
        # WebRTC shared state (persisted across reruns)
        "webrtc_shared_state": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def reset_session_state():
    st.session_state.alert_count = 0
    st.session_state.suspicious_events = []
    st.session_state.threat_history = []
    st.session_state.screenshots = []
    st.session_state.last_threat_score = 0
    st.session_state.analysis_complete = False
    st.session_state.upload_results = False
    st.session_state.video_path = None


def render_main_header():
    st.markdown("<h1>🎥 AI Surveillance System</h1>", unsafe_allow_html=True)
    st.markdown("<p>Real-time YOLO detection, threat analysis, recording, export and analytics.</p>", unsafe_allow_html=True)
    st.markdown("---")


def render_export_buttons(summary):
    if not summary:
        return
    cols = st.columns(2)
    with cols[0]:
        csv_bytes = export_csv_report(summary, st.session_state.alert_count, len(st.session_state.screenshots))
        st.download_button(
            "⬇ Download CSV Report",
            data=csv_bytes,
            file_name=f"surveillance_report_{get_timestamp_string()}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with cols[1]:
        txt_bytes = export_txt_report(summary, st.session_state.alert_count, len(st.session_state.screenshots))
        st.download_button(
            "⬇ Download TXT Report",
            data=txt_bytes,
            file_name=f"surveillance_report_{get_timestamp_string()}.txt",
            mime="text/plain",
            use_container_width=True,
        )


def render_analysis_panel():
    st.markdown("---")
    st.markdown("## 📊 Analytics")
    render_metric_cards(
        st.session_state.alert_count,
        len(summarize_events(st.session_state.suspicious_events)),
        len(st.session_state.screenshots),
    )
    if st.session_state.threat_history:
        st.plotly_chart(build_threat_timeline(st.session_state.threat_history), use_container_width=True)
    summary = summarize_events(st.session_state.suspicious_events)
    if summary:
        st.plotly_chart(build_risk_pie(summary), use_container_width=True)
        render_event_cards(summary)
    render_export_buttons(summary)
    render_screenshot_grid(st.session_state.screenshots)


def save_video_if_ready(video_path):
    if video_path and Path(video_path).exists():
        with open(video_path, "rb") as f:
            st.download_button(
                "📥 Download Recorded Video",
                data=f.read(),
                file_name=Path(video_path).name,
                mime="video/x-msvideo",
                use_container_width=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# WEBRTC LIVE CAMERA (cloud / Hugging Face Spaces)
# ═══════════════════════════════════════════════════════════════════════════════

def _render_threat_gauge(threat_score: int):
    """Render a color-coded threat level bar."""
    color = "#ef4444" if threat_score >= THREAT_THRESHOLD else "#22c55e"
    st.markdown(
        f"""
        <div style="margin:8px 0 4px 0;">
            <span style="font-size:13px;color:#888;">Live Threat Level</span>
        </div>
        <div style="background:#1e293b;border-radius:8px;height:22px;width:100%;overflow:hidden;">
            <div style="width:{threat_score}%;height:100%;background:{color};
                        border-radius:8px;transition:width 0.3s;"></div>
        </div>
        <div style="font-size:20px;font-weight:700;color:{color};margin-top:4px;">
            {threat_score}%
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_alarm_audio():
    """
    Inject an HTML5 audio element for browser-compatible alarm.
    We use a data-URI beep (generated inline) so no file upload needed.
    The browser may block autoplay — we also show a visual flash.
    """
    st.markdown(
        """
        <script>
        (function() {
            try {
                var ctx = new (window.AudioContext || window.webkitAudioContext)();
                var osc = ctx.createOscillator();
                var gain = ctx.createGain();
                osc.connect(gain);
                gain.connect(ctx.destination);
                osc.frequency.value = 880;
                osc.type = 'square';
                gain.gain.setValueAtTime(0.3, ctx.currentTime);
                osc.start();
                gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.4);
                osc.stop(ctx.currentTime + 0.4);
            } catch(e) {}
        })();
        </script>
        """,
        unsafe_allow_html=True,
    )


def render_webrtc_camera(model):
    """
    Cloud-mode live camera using streamlit-webrtc.
    Frames are processed by SurveillanceProcessor in a background thread.
    The main thread polls SharedState every ~0.5 s to refresh metrics.
    """
    if not WEBRTC_AVAILABLE:
        st.error("⚠️ streamlit-webrtc is not installed. Please add it to requirements.txt.")
        return

    st.markdown("<div class='card'><h2>📹 Live Browser Camera</h2></div>", unsafe_allow_html=True)
    st.info(
        "🌐 **Cloud Camera Mode** — Your browser will request webcam permission. "
        "Detection runs live via YOLO. Allow camera access to start."
    )

    # Initialise / retrieve the shared state from session (persisted across reruns)
    if st.session_state.webrtc_shared_state is None:
        st.session_state.webrtc_shared_state = SharedState()
    shared: SharedState = st.session_state.webrtc_shared_state

    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        if st.button("🔄 Reset Session Data", use_container_width=True):
            shared.reset()
            reset_session_state()
            st.rerun()
    with col_ctrl2:
        alarm_enabled = st.session_state.alarm_enabled

    # ── WebRTC streamer widget ────────────────────────────────────────────────
    ctx = webrtc_streamer(
        key="surveillance-webrtc",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_processor_factory=lambda: SurveillanceProcessor(model, shared),
        media_stream_constraints={"video": {"width": FRAME_WIDTH, "height": FRAME_HEIGHT}, "audio": False},
        async_processing=True,
    )

    # ── Live metrics panel (updates while streaming) ──────────────────────────
    st.markdown("---")
    live_col1, live_col2 = st.columns([2, 1])

    with live_col1:
        threat_placeholder = st.empty()
        alert_placeholder = st.empty()

    with live_col2:
        metric_placeholder = st.empty()

    screenshot_placeholder = st.empty()
    analytics_placeholder = st.empty()

    # ── Polling loop — runs only while the WebRTC stream is active ────────────
    if ctx.state.playing:
        st.session_state.camera_active = True
        poll_interval = 0.5  # seconds

        while ctx.state.playing:
            snap = shared.snapshot()

            # Update threat gauge
            with threat_placeholder.container():
                _render_threat_gauge(snap["last_threat_score"])

            # Flash alert if new alarm triggered
            if snap["new_alert"]:
                with alert_placeholder.container():
                    st.error(f"🚨 ALERT! Threat detected — {snap['alert_count']} total alerts")
                if alarm_enabled:
                    _render_alarm_audio()
                shared.clear_new_alert()
            else:
                with alert_placeholder.container():
                    if snap["alert_count"] > 0:
                        st.warning(f"⚠️ {snap['alert_count']} alerts logged this session")
                    else:
                        st.success("✅ Monitoring active — no threats detected")

            # Sync shared state → session state for analytics panel
            st.session_state.threat_history = snap["threat_history"]
            st.session_state.suspicious_events = snap["suspicious_events"]
            st.session_state.alert_count = snap["alert_count"]
            st.session_state.screenshots = snap["screenshots"]

            # Metric cards
            with metric_placeholder.container():
                render_metric_cards(
                    snap["alert_count"],
                    len(summarize_events(snap["suspicious_events"])),
                    len(snap["screenshots"]),
                )

            # Live screenshots
            with screenshot_placeholder.container():
                render_screenshot_grid(snap["screenshots"])

            # Live analytics (only if there's data)
            if snap["threat_history"]:
                with analytics_placeholder.container():
                    st.plotly_chart(
                        build_threat_timeline(snap["threat_history"]),
                        use_container_width=True,
                    )

            time.sleep(poll_interval)

    else:
        st.session_state.camera_active = False

    # ── Post-stream report ────────────────────────────────────────────────────
    if not ctx.state.playing and st.session_state.alert_count > 0:
        st.markdown("---")
        st.markdown("### 📊 Session Summary")
        render_analysis_panel()


# ═══════════════════════════════════════════════════════════════════════════════
# LOCALHOST LIVE CAMERA (unchanged from original)
# ═══════════════════════════════════════════════════════════════════════════════

def render_live_camera(model):
    """
    Route to the correct camera implementation:
    - DEPLOYMENT_MODE=True  → streamlit-webrtc (browser WebCam, cloud-safe)
    - DEPLOYMENT_MODE=False → cv2.VideoCapture (localhost only)
    """
    if DEPLOYMENT_MODE:
        render_webrtc_camera(model)
        return

    # ── Original localhost cv2 path (100% unchanged) ──────────────────────────
    st.markdown("<div class='card'><h2>📹 Live Camera</h2></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        start_camera = st.button("▶ Start Camera", use_container_width=True)
    with col2:
        stop_camera = st.button("⏹ Stop Camera", use_container_width=True)

    if start_camera:
        st.session_state.run_camera = True
        st.session_state.camera_active = True
        reset_session_state()
        st.session_state.alarm_enabled = st.session_state.alarm_enabled

    if stop_camera:
        st.session_state.run_camera = False
        st.session_state.camera_active = False

    frame_placeholder = st.empty()
    status_placeholder = st.empty()
    if st.session_state.run_camera:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Camera unavailable")
            st.warning("Please verify a local webcam is attached or run in Upload Video mode.")
            st.session_state.run_camera = False
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)

        recorder = None
        if ENABLE_RECORDING:
            path = VIDEO_OUTPUT_DIR / f"surveillance_{get_timestamp_string()}.avi"
            recorder = VideoRecorder(str(path), (FRAME_WIDTH, FRAME_HEIGHT))
            st.session_state.video_path = str(path)

        status_placeholder.info("🔄 Camera active - processing frames...")
        while st.session_state.run_camera:
            success, frame = cap.read()
            if not success or frame is None:
                break

            annotated_frame, detections, threat_score = process_frame(frame.copy(), model)
            st.session_state.threat_history.append(threat_score)
            if should_alert(st.session_state.last_threat_score, threat_score, THREAT_THRESHOLD):
                st.session_state.alert_count += 1
                for det in detections:
                    st.session_state.suspicious_events.append(
                        {
                            "label": det["label"],
                            "confidence": int(det["confidence"] * 100),
                            "threat": int(det["confidence"] * 100),
                        }
                    )
                if st.session_state.alarm_enabled:
                    play_alarm()
                if len(st.session_state.screenshots) < MAX_SCREENSHOTS:
                    path = capture_snapshot(annotated_frame, SNAPSHOT_OUTPUT_DIR, "camera_alert")
                    if path:
                        st.session_state.screenshots.append(str(path))

            st.session_state.last_threat_score = threat_score
            if recorder and recorder.active:
                recorder.write(annotated_frame)

            annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(annotated_rgb, use_container_width=True)
            time.sleep(0.03)

        cap.release()
        if recorder:
            recorder.stop()
            status_placeholder.success("✅ Camera stopped and saved.")
        else:
            status_placeholder.success("✅ Camera stopped.")

    if not st.session_state.run_camera and st.session_state.alert_count > 0:
        render_analysis_panel()
        if ENABLE_RECORDING:
            st.markdown("---")
            st.markdown("## 📹 Recorded Video")
            save_video_if_ready(st.session_state.video_path)
        elif DEPLOYMENT_MODE:
            st.info("Deployment mode active: camera recording is disabled.")


# ═══════════════════════════════════════════════════════════════════════════════
# UPLOAD VIDEO MODE (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

def render_upload_mode(model):
    st.markdown("<div class='card'><h2>📤 Upload Video</h2></div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_file is None:
        return

    if model is None:
        st.warning("Model failed to load. Upload may still work after retry, but accurate detection could be unavailable.")

    if st.button("🔍 Analyze Video", use_container_width=True):
        temp_dir = Path(tempfile.gettempdir())
        temp_video = temp_dir / uploaded_file.name
        temp_video.write_bytes(uploaded_file.getbuffer())

        try:
            cap = cv2.VideoCapture(str(temp_video))
            if not cap.isOpened():
                st.error("❌ Failed to open uploaded video. Please upload a supported file.")
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            progress_bar = st.progress(0)
            status = st.empty()
            status.info("🔄 Processing uploaded video...")

            reset_session_state()
            frame_count = 0
            screenshot_timer = time.time() - 10

            while True:
                success, frame = cap.read()
                if not success or frame is None:
                    break

                frame_count += 1
                annotated_frame, detections, threat_score = process_frame(frame.copy(), model)
                st.session_state.threat_history.append(threat_score)
                if threat_score > THREAT_THRESHOLD:
                    st.session_state.alert_count += 1
                    for det in detections:
                        st.session_state.suspicious_events.append(
                            {
                                "label": det["label"],
                                "confidence": int(det["confidence"] * 100),
                                "threat": int(det["confidence"] * 100),
                            }
                        )
                    if len(st.session_state.screenshots) < MAX_SCREENSHOTS and time.time() - screenshot_timer > 2.0:
                        path = capture_snapshot(annotated_frame, SNAPSHOT_OUTPUT_DIR, "upload_alert")
                        if path:
                            st.session_state.screenshots.append(str(path))
                            screenshot_timer = time.time()

                if total_frames:
                    progress_bar.progress(min(frame_count / total_frames, 1.0))

            st.session_state.analysis_complete = True
            st.session_state.upload_results = True
            status.success("✅ Video analysis complete.")
        except Exception as exc:
            st.error(f"❌ Error processing upload: {exc}")
        finally:
            try:
                cap.release()
            except Exception:
                pass
            if temp_video.exists():
                temp_video.unlink()

    if st.session_state.upload_results and st.session_state.alert_count > 0:
        render_analysis_panel()
    elif st.session_state.upload_results and st.session_state.alert_count == 0:
        st.info("No alerts were detected in the uploaded video.")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    init_session_state()
    render_main_header()
    model = get_model()

    with st.sidebar:
        st.title("⚙️ Controls")
        # In cloud mode Live Camera is now supported via WebRTC
        mode_options = ["Live Camera", "Upload Video"]
        mode = st.radio("Select mode", mode_options)
        st.markdown("---")
        st.markdown("**Model status**")
        if model is not None:
            st.success("✓ YOLO model loaded")
        else:
            st.error("✗ Model failed to load")
        st.markdown("---")
        st.session_state.alarm_enabled = st.checkbox("Enable alarm", value=st.session_state.alarm_enabled)
        st.markdown(f"**Deployment mode:** {'ON ☁️' if DEPLOYMENT_MODE else 'OFF 🖥️'}")
        if DEPLOYMENT_MODE:
            st.info("🌐 Cloud mode — browser WebCam via WebRTC")
        st.markdown("---")
        st.markdown("**Report outputs**")
        st.write(f"Reports: {REPORT_OUTPUT_DIR}")
        st.write(f"Snapshots: {SNAPSHOT_OUTPUT_DIR}")

    if mode == "Live Camera":
        render_live_camera(model)
    else:
        render_upload_mode(model)

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:gray;font-size:12px;'>"
        "Deployment-ready architecture with modular services and centralized config."
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
