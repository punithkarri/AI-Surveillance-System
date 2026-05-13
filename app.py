"""
AI Surveillance System — Streamlit App
=======================================
Entry point for both local and Hugging Face Spaces deployment.
# Triggering HF Space rebuild

Routing logic:
  DEPLOYMENT_MODE=True  (HF Spaces / SPACE_ID env var set)
      → Live camera via streamlit-webrtc (browser WebRTC, non-blocking)
  DEPLOYMENT_MODE=False (localhost)
      → Live camera via cv2.VideoCapture(0)

Upload Video mode works identically in both environments.
"""

import tempfile
import time
from pathlib import Path

import cv2
import streamlit as st

from config.settings import (
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
    summarize_events,
)
from utils.helpers import ensure_folder, get_timestamp_string
from utils.ui import (
    inject_custom_css,
    render_event_cards,
    render_metric_cards,
    render_screenshot_grid,
)

# ── WebRTC (cloud-safe import) ────────────────────────────────────────────────
WEBRTC_AVAILABLE = False
if DEPLOYMENT_MODE:
    try:
        from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
        from services.webrtc_processor import SurveillanceProcessor, SharedState
        WEBRTC_AVAILABLE = True
    except Exception:
        pass  # Graceful fallback if not installed

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Surveillance System",
    page_icon="🎥",
    layout="wide",
)
inject_custom_css()

# ── WebRTC STUN configuration ─────────────────────────────────────────────────
RTC_CONFIG = None
if WEBRTC_AVAILABLE:
    RTC_CONFIG = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_model():
    return load_model()


# ─────────────────────────────────────────────────────────────────────────────
# Session state helpers
# ─────────────────────────────────────────────────────────────────────────────

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
        "webrtc_shared_state": None,  # SharedState instance (cloud mode)
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def reset_session_state():
    st.session_state.alert_count = 0
    st.session_state.suspicious_events = []
    st.session_state.threat_history = []
    st.session_state.screenshots = []
    st.session_state.last_threat_score = 0
    st.session_state.analysis_complete = False
    st.session_state.upload_results = False
    st.session_state.video_path = None


# ─────────────────────────────────────────────────────────────────────────────
# Shared UI helpers
# ─────────────────────────────────────────────────────────────────────────────

def render_main_header():
    st.markdown("<h1>🎥 AI Surveillance System</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p>Real-time YOLO detection · Threat analysis · Analytics · Reports</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")


def render_export_buttons(summary):
    if not summary:
        return
    col1, col2 = st.columns(2)
    with col1:
        csv_bytes = export_csv_report(
            summary, st.session_state.alert_count, len(st.session_state.screenshots)
        )
        st.download_button(
            "⬇ Download CSV Report",
            data=csv_bytes,
            file_name=f"surveillance_report_{get_timestamp_string()}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col2:
        txt_bytes = export_txt_report(
            summary, st.session_state.alert_count, len(st.session_state.screenshots)
        )
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
    summary = summarize_events(st.session_state.suspicious_events)
    render_metric_cards(
        st.session_state.alert_count,
        len(summary),
        len(st.session_state.screenshots),
    )
    if st.session_state.threat_history:
        st.plotly_chart(
            build_threat_timeline(st.session_state.threat_history),
            use_container_width=True,
        )
    if summary:
        pie = build_risk_pie(summary)
        if pie:
            st.plotly_chart(pie, use_container_width=True)
        render_event_cards(summary)
    render_export_buttons(summary)
    render_screenshot_grid(st.session_state.screenshots)


# ─────────────────────────────────────────────────────────────────────────────
# WebRTC helpers
# ─────────────────────────────────────────────────────────────────────────────

def _render_threat_gauge(threat_score: int):
    """Render a colour-coded live threat bar."""
    color = "#ef4444" if threat_score >= THREAT_THRESHOLD else "#22c55e"
    label = "🔴 HIGH THREAT" if threat_score >= THREAT_THRESHOLD else "🟢 SAFE"
    st.markdown(
        f"""
        <div style="margin:8px 0 4px 0;display:flex;justify-content:space-between;">
            <span style="font-size:13px;color:#888;">Live Threat Level</span>
            <span style="font-size:13px;font-weight:600;color:{color};">{label}</span>
        </div>
        <div style="background:#1e293b;border-radius:8px;height:24px;width:100%;overflow:hidden;">
            <div style="width:{threat_score}%;height:100%;background:{color};
                        border-radius:8px;transition:width 0.3s;"></div>
        </div>
        <div style="font-size:28px;font-weight:700;color:{color};margin-top:4px;">
            {threat_score}%
        </div>
        """,
        unsafe_allow_html=True,
    )


def _inject_alarm_beep():
    """Inject a browser-compatible AudioContext beep (will only play after user gesture)."""
    st.markdown(
        """
        <script>
        (function() {
            try {
                var AudioCtx = window.AudioContext || window.webkitAudioContext;
                if (!AudioCtx) return;
                var ctx = new AudioCtx();
                function beep(freq, start, duration) {
                    var osc = ctx.createOscillator();
                    var gain = ctx.createGain();
                    osc.connect(gain);
                    gain.connect(ctx.destination);
                    osc.frequency.value = freq;
                    osc.type = 'square';
                    gain.gain.setValueAtTime(0.25, ctx.currentTime + start);
                    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + start + duration);
                    osc.start(ctx.currentTime + start);
                    osc.stop(ctx.currentTime + start + duration);
                }
                beep(880, 0,    0.2);
                beep(660, 0.25, 0.2);
                beep(880, 0.5,  0.2);
            } catch(e) {}
        })();
        </script>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLOUD CAMERA — streamlit-webrtc (non-blocking, rerun-based)
# ─────────────────────────────────────────────────────────────────────────────

def render_webrtc_camera(model):
    """
    Cloud live camera using streamlit-webrtc.

    Key design: NO blocking loops on the main thread.
    The WebRTC processor runs in its own background thread.
    Each Streamlit rerun reads a snapshot from SharedState and updates the UI.
    st.rerun() is called at the end to keep metrics refreshing while streaming.
    """
    if not WEBRTC_AVAILABLE:
        st.error(
            "⚠️ streamlit-webrtc is not available in this environment. "
            "Please ensure `streamlit-webrtc` and `av` are in requirements.txt."
        )
        return

    st.markdown(
        "<div class='card'><h2>📹 Live Browser Camera</h2></div>",
        unsafe_allow_html=True,
    )
    st.info(
        "🌐 **Cloud Camera Mode** — Click **START** below, then allow webcam access "
        "when your browser asks. YOLO detection runs live on every frame."
    )

    # ── Initialise SharedState (persists across reruns in session) ─────────
    if st.session_state.webrtc_shared_state is None:
        st.session_state.webrtc_shared_state = SharedState()
    shared: SharedState = st.session_state.webrtc_shared_state

    # ── Reset button ───────────────────────────────────────────────────────
    col_r, col_a = st.columns(2)
    with col_r:
        if st.button("🔄 Reset Session", use_container_width=True, key="webrtc_reset"):
            shared.reset()
            reset_session_state()
            st.rerun()
    with col_a:
        st.session_state.alarm_enabled = st.checkbox(
            "🔔 Enable alarm sound",
            value=st.session_state.alarm_enabled,
            key="webrtc_alarm_chk",
        )

    # ── WebRTC streamer widget ─────────────────────────────────────────────
    ctx = webrtc_streamer(
        key="surveillance-webrtc",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_processor_factory=lambda: SurveillanceProcessor(model, shared),
        media_stream_constraints={
            "video": {"width": {"ideal": FRAME_WIDTH}, "height": {"ideal": FRAME_HEIGHT}},
            "audio": False,
        },
        async_processing=True,
    )

    streaming = ctx.state.playing

    # ── Take a thread-safe snapshot of current metrics ─────────────────────
    snap = shared.snapshot()

    # ── Sync shared → session state (for analytics panel) ─────────────────
    st.session_state.threat_history = snap["threat_history"]
    st.session_state.suspicious_events = snap["suspicious_events"]
    st.session_state.alert_count = snap["alert_count"]
    st.session_state.screenshots = snap["screenshots"]

    # ── Live metrics panel ─────────────────────────────────────────────────
    st.markdown("---")

    if streaming:
        st.session_state.camera_active = True

        live_col1, live_col2 = st.columns([3, 2])

        with live_col1:
            _render_threat_gauge(snap["last_threat_score"])

            if snap["new_alert"]:
                st.error(
                    f"🚨 **THREAT DETECTED!** Total alerts this session: {snap['alert_count']}"
                )
                if st.session_state.alarm_enabled:
                    _inject_alarm_beep()
                shared.clear_new_alert()
            elif snap["alert_count"] > 0:
                st.warning(f"⚠️ {snap['alert_count']} alert(s) logged this session")
            else:
                st.success("✅ Monitoring active — no threats detected")

        with live_col2:
            summary_live = summarize_events(snap["suspicious_events"])
            render_metric_cards(
                snap["alert_count"],
                len(summary_live),
                len(snap["screenshots"]),
            )

        # Live threat timeline
        if snap["threat_history"]:
            st.plotly_chart(
                build_threat_timeline(snap["threat_history"]),
                use_container_width=True,
            )

        # Live screenshots
        render_screenshot_grid(snap["screenshots"])

        # ── Refresh every ~1 second while streaming ──────────────────────
        # This is the correct non-blocking pattern for streamlit-webrtc:
        # sleep briefly then rerun; the WebRTC BG thread continues uninterrupted.
        time.sleep(1.0)
        st.rerun()

    else:
        st.session_state.camera_active = False
        if snap["alert_count"] > 0:
            st.markdown("### 📊 Session Summary")
            render_analysis_panel()
        else:
            st.markdown(
                "<div style='text-align:center;padding:40px;color:#888;'>"
                "Click <strong>START</strong> to begin live surveillance</div>",
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# LOCALHOST CAMERA — cv2.VideoCapture (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def render_live_camera(model):
    """Route to WebRTC (cloud) or cv2 (localhost) based on DEPLOYMENT_MODE."""
    if DEPLOYMENT_MODE:
        render_webrtc_camera(model)
        return

    # ── Original localhost cv2 path (100% unchanged) ──────────────────────
    st.markdown(
        "<div class='card'><h2>📹 Live Camera</h2></div>",
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ Start Camera", use_container_width=True, key="cam_start"):
            st.session_state.run_camera = True
            st.session_state.camera_active = True
            reset_session_state()
    with col2:
        if st.button("⏹ Stop Camera", use_container_width=True, key="cam_stop"):
            st.session_state.run_camera = False
            st.session_state.camera_active = False

    frame_placeholder = st.empty()
    status_placeholder = st.empty()

    if st.session_state.run_camera:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Camera unavailable — check that a webcam is connected.")
            st.session_state.run_camera = False
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)

        recorder = None
        if ENABLE_RECORDING:
            vid_path = VIDEO_OUTPUT_DIR / f"surveillance_{get_timestamp_string()}.avi"
            recorder = VideoRecorder(str(vid_path), (FRAME_WIDTH, FRAME_HEIGHT))
            st.session_state.video_path = str(vid_path)

        status_placeholder.info("🔄 Camera active — processing frames…")

        while st.session_state.run_camera:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            annotated, detections, threat_score = process_frame(frame.copy(), model)
            st.session_state.threat_history.append(threat_score)

            if should_alert(st.session_state.last_threat_score, threat_score, THREAT_THRESHOLD):
                st.session_state.alert_count += 1
                for det in detections:
                    st.session_state.suspicious_events.append(
                        {"label": det["label"], "confidence": int(det["confidence"] * 100), "threat": int(det["confidence"] * 100)}
                    )
                if st.session_state.alarm_enabled:
                    play_alarm()
                if len(st.session_state.screenshots) < MAX_SCREENSHOTS:
                    p = capture_snapshot(annotated, SNAPSHOT_OUTPUT_DIR, "camera_alert")
                    if p:
                        st.session_state.screenshots.append(str(p))

            st.session_state.last_threat_score = threat_score
            if recorder and recorder.active:
                recorder.write(annotated)

            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb, use_container_width=True)
            time.sleep(0.03)

        cap.release()
        if recorder:
            recorder.stop()
        status_placeholder.success("✅ Camera stopped.")

    if not st.session_state.run_camera and st.session_state.alert_count > 0:
        render_analysis_panel()
        if ENABLE_RECORDING and st.session_state.video_path:
            st.markdown("---")
            st.markdown("## 📹 Recorded Video")
            vp = Path(st.session_state.video_path)
            if vp.exists():
                with open(str(vp), "rb") as f:
                    st.download_button(
                        "📥 Download Recorded Video",
                        data=f.read(),
                        file_name=vp.name,
                        mime="video/x-msvideo",
                        use_container_width=True,
                    )


# ─────────────────────────────────────────────────────────────────────────────
# UPLOAD VIDEO MODE (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def render_upload_mode(model):
    st.markdown(
        "<div class='card'><h2>📤 Upload Video</h2></div>",
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Choose a video file", type=["mp4", "avi", "mov", "mkv"], key="video_upload"
    )
    if uploaded is None:
        return

    if st.button("🔍 Analyze Video", use_container_width=True, key="analyze_btn"):
        tmp = Path(tempfile.gettempdir()) / uploaded.name
        tmp.write_bytes(uploaded.getbuffer())

        try:
            cap = cv2.VideoCapture(str(tmp))
            if not cap.isOpened():
                st.error("❌ Could not open the uploaded video.")
                return

            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            progress = st.progress(0)
            status = st.empty()
            status.info("🔄 Processing video…")

            reset_session_state()
            count = 0
            last_shot = time.time() - 10

            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                count += 1
                annotated, detections, threat_score = process_frame(frame.copy(), model)
                st.session_state.threat_history.append(threat_score)

                if threat_score > THREAT_THRESHOLD:
                    st.session_state.alert_count += 1
                    for det in detections:
                        st.session_state.suspicious_events.append(
                            {"label": det["label"], "confidence": int(det["confidence"] * 100), "threat": int(det["confidence"] * 100)}
                        )
                    if len(st.session_state.screenshots) < MAX_SCREENSHOTS and time.time() - last_shot > 2.0:
                        p = capture_snapshot(annotated, SNAPSHOT_OUTPUT_DIR, "upload_alert")
                        if p:
                            st.session_state.screenshots.append(str(p))
                            last_shot = time.time()

                if total:
                    progress.progress(min(count / total, 1.0))

            st.session_state.analysis_complete = True
            st.session_state.upload_results = True
            status.success("✅ Video analysis complete.")
        except Exception as exc:
            st.error(f"❌ Error: {exc}")
        finally:
            try:
                cap.release()
            except Exception:
                pass
            if tmp.exists():
                tmp.unlink()

    if st.session_state.upload_results:
        if st.session_state.alert_count > 0:
            render_analysis_panel()
        else:
            st.info("No threats detected in the uploaded video.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    init_session_state()
    render_main_header()
    model = get_model()

    with st.sidebar:
        st.title("⚙️ Controls")
        mode = st.radio("Select mode", ["Live Camera", "Upload Video"], key="mode_radio")
        st.markdown("---")
        st.markdown("**Model**")
        if model is not None:
            st.success("✓ YOLO loaded")
        else:
            st.error("✗ Model failed to load")
        st.markdown("---")
        if not DEPLOYMENT_MODE:
            # In cloud mode the alarm checkbox lives inside render_webrtc_camera
            st.session_state.alarm_enabled = st.checkbox(
                "🔔 Enable alarm",
                value=st.session_state.alarm_enabled,
                key="alarm_sidebar",
            )
        mode_label = "ON ☁️ (WebRTC camera)" if DEPLOYMENT_MODE else "OFF 🖥️ (local webcam)"
        st.markdown(f"**Deployment mode:** {mode_label}")
        st.markdown("---")
        st.caption(f"Snapshots: {SNAPSHOT_OUTPUT_DIR}")
        st.caption(f"Reports: {REPORT_OUTPUT_DIR}")

    if mode == "Live Camera":
        render_live_camera(model)
    else:
        render_upload_mode(model)

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:gray;font-size:12px;'>"
        "AI Surveillance System · YOLOv8 · Streamlit · streamlit-webrtc"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
