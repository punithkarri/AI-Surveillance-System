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


st.set_page_config(page_title="AI Surveillance System", page_icon="🎥", layout="wide")
inject_custom_css()


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


def render_live_camera(model):
    if DEPLOYMENT_MODE:
        st.warning("Live Camera is unavailable in deployment mode. Switch to Upload Video.")
        return

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
            frame_placeholder.image(annotated_rgb, use_column_width=True)
            time.sleep(0.03)

        cap.release()
        if recorder:
            recorder.stop()
            status_placeholder.success("✅ Camera stopped and saved.")
        else:
            status_placeholder.success("✅ Camera stopped.")

    if not st.session_state.run_camera and st.session_state.alert_count > 0:
        render_analysis_panel()
        if ENABLE_CAMERA_RECORDING:
            st.markdown("---")
            st.markdown("## 📹 Recorded Video")
            save_video_if_ready(st.session_state.video_path)
        elif DEPLOYMENT_MODE:
            st.info("Deployment mode active: camera recording is disabled.")


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


def main():
    init_session_state()
    render_main_header()
    model = get_model()

    with st.sidebar:
        st.title("⚙️ Controls")
        mode_options = ["Upload Video"] if DEPLOYMENT_MODE else ["Live Camera", "Upload Video"]
        mode = st.radio("Select mode", mode_options)
        st.markdown("---")
        st.markdown("**Model status**")
        if model is not None:
            st.success("✓ YOLO model loaded")
        else:
            st.error("✗ Model failed to load")
        st.markdown("---")
        st.session_state.alarm_enabled = st.checkbox("Enable alarm", value=st.session_state.alarm_enabled)
        st.markdown(f"**Deployment mode:** {'ON' if DEPLOYMENT_MODE else 'OFF'}")
        if DEPLOYMENT_MODE:
            st.warning("🌐 Cloud mode active — use Upload Video for safer processing.")
            st.info("Local webcam and recording are disabled in deployment mode.")
        st.markdown("---")
        st.markdown("**Report outputs**")
        st.write(f"Videos: {VIDEO_OUTPUT_DIR}")
        st.write(f"Reports: {REPORT_OUTPUT_DIR}")
        st.write(f"Snapshots: {SNAPSHOT_OUTPUT_DIR}")

    if mode == "Live Camera":
        if DEPLOYMENT_MODE:
            st.warning("Live Camera mode is disabled in cloud deployment mode.")
        else:
            render_live_camera(model)
    else:
        render_upload_mode(model)

    st.markdown("---")
    st.markdown("<div style='text-align:center;color:gray;font-size:12px;'>Deployment-ready architecture with modular services and centralized config.</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
