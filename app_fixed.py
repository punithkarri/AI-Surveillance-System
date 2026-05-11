import streamlit as st
import cv2
import tempfile
import threading
import time
import csv
from pathlib import Path
from collections import deque
import numpy as np
from datetime import datetime
from ultralytics import YOLO

try:
    import winsound
except ImportError:
    winsound = None

# ============================================================
# CONFIGURATION CONSTANTS
# ============================================================
TARGET_OBJECTS = {"person", "cell phone", "laptop", "tablet", "book", "bottle", "watch", "remote", "mouse", "keyboard"}
THREAT_OBJECTS = {"cell phone", "laptop", "tablet", "book", "remote", "watch"}
CONFIDENCE_THRESHOLD = 0.5
ALERT_THREAT_THRESHOLD = 60
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MAX_SCREENSHOTS = 10
SCREENSHOT_GAP = 2.0
ALARM_COOLDOWN = 1.5

# ============================================================
# PAGE CONFIG & CUSTOM CSS
# ============================================================
st.set_page_config(
    page_title="AI Surveillance System",
    page_icon="🎥",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
    padding: 0;
}
.stMainBlockContainer {
    padding-top: 1rem;
}
.card {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    margin: 10px 0;
}
.metric {
    font-size: 36px;
    font-weight: bold;
    color: #1f77b4;
}
.header-title {
    font-size: 42px;
    font-weight: bold;
    color: #0d3b66;
    margin-bottom: 5px;
}
.header-subtitle {
    font-size: 16px;
    color: #666;
    margin-bottom: 20px;
}
.threat-safe {
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    padding: 15px;
    border-radius: 8px;
    color: #155724;
    font-weight: bold;
}
.threat-medium {
    background-color: #fff3cd;
    border: 1px solid #ffeaa7;
    padding: 15px;
    border-radius: 8px;
    color: #856404;
    font-weight: bold;
}
.threat-high {
    background-color: #f8d7da;
    border: 1px solid #f5c6cb;
    padding: 15px;
    border-radius: 8px;
    color: #721c24;
    font-weight: bold;
}
.event-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 15px;
    border-radius: 10px;
    margin: 8px 0;
    box-shadow: 0px 4px 8px rgba(0,0,0,0.15);
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header-title">🎥 AI Surveillance System</div>', unsafe_allow_html=True)
st.markdown('<div class="header-subtitle">Real-time Threat Detection using YOLO + Behavior Analysis</div>', unsafe_allow_html=True)
st.markdown("---")

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
if "run_camera" not in st.session_state:
    st.session_state.run_camera = False
if "alarm_on" not in st.session_state:
    st.session_state.alarm_on = True
if "alert_count" not in st.session_state:
    st.session_state.alert_count = 0
if "screenshots" not in st.session_state:
    st.session_state.screenshots = []
if "suspicious_events" not in st.session_state:
    st.session_state.suspicious_events = []
if "threat_history" not in st.session_state:
    st.session_state.threat_history = deque(maxlen=100)
if "model" not in st.session_state:
    try:
        st.session_state.model = YOLO("yolov8n.pt")
    except:
        st.session_state.model = None
if "video_output_path" not in st.session_state:
    st.session_state.video_output_path = None
if "recording_active" not in st.session_state:
    st.session_state.recording_active = False
if "timestamps" not in st.session_state:
    st.session_state.timestamps = []

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def play_alarm():
    """Play alarm sound."""
    if winsound and st.session_state.alarm_on:
        try:
            winsound.Beep(1000, 500)
            time.sleep(0.2)
            winsound.Beep(1200, 500)
        except:
            pass

def calculate_threat_score(detections):
    """Calculate threat score for current frame."""
    threat_score = 0
    for det in detections:
        obj_name = det["object_name"]
        if obj_name in THREAT_OBJECTS:
            if obj_name == "cell phone":
                threat_score += 85
            elif obj_name in {"laptop", "tablet"}:
                threat_score += 65
            elif obj_name == "book":
                threat_score += 55
            elif obj_name == "remote":
                threat_score += 45
            elif obj_name == "watch":
                threat_score += 40
        elif obj_name == "person":
            threat_score += 25
        else:
            threat_score += 10
    return min(threat_score, 100)

def process_frame(frame, model):
    """Process frame with YOLO detection and draw bounding boxes."""
    if frame is None or model is None:
        return frame, [], 0

    # Resize for performance
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    try:
        results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
    except:
        return frame, [], 0

    detections = []
    for result in results:
        if not hasattr(result, "boxes") or len(result.boxes) == 0:
            continue

        for box in result.boxes:
            try:
                xyxy = box.xyxy
                if xyxy is None or len(xyxy) == 0:
                    continue

                x1, y1, x2, y2 = map(int, xyxy[0].tolist())
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                if conf < CONFIDENCE_THRESHOLD:
                    continue

                label = model.names[cls_id] if hasattr(model, "names") else str(cls_id)

                if label not in TARGET_OBJECTS:
                    continue

                # Draw bounding box
                if label in THREAT_OBJECTS:
                    color = (0, 0, 255)  # Red for threats
                else:
                    color = (0, 255, 0)  # Green for normal

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw label
                label_text = f"{label} {int(conf * 100)}%"
                cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                detections.append({
                    "object_name": label,
                    "confidence": conf,
                    "bbox": (x1, y1, x2, y2)
                })

            except:
                continue

    threat_score = calculate_threat_score(detections)

    # Draw threat text
    cv2.putText(frame, f"Threat: {threat_score}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return frame, detections, threat_score

def summarize_events(suspicious_events):
    """Summarize suspicious events."""
    summary = {}
    for event in suspicious_events:
        obj = event.get("object_name", "unknown")
        conf = event.get("confidence", 0)
        if obj not in summary:
            summary[obj] = {"count": 1, "max_threat": conf}
        else:
            summary[obj]["count"] += 1
            summary[obj]["max_threat"] = max(summary[obj]["max_threat"], conf)
    return summary

def export_csv(summary, alert_count, screenshots_count):
    """Export CSV report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"surveillance_report_{timestamp}.csv"
    filepath = Path(filename)

    try:
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["AI SURVEILLANCE REPORT"])
            writer.writerow(["Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            writer.writerow([])
            writer.writerow(["SUMMARY"])
            writer.writerow(["Total Alerts", alert_count])
            writer.writerow(["Screenshots", screenshots_count])
            writer.writerow(["Unique Events", len(summary)])
            writer.writerow([])
            writer.writerow(["EVENTS"])
            writer.writerow(["Object", "Count", "Max Threat %", "Risk Level"])
            for obj, data in sorted(summary.items()):
                risk = "HIGH" if obj in THREAT_OBJECTS else "LOW"
                writer.writerow([obj.title(), data["count"], data["max_threat"], risk])
        return filepath
    except:
        return None

def export_txt(summary, alert_count, screenshots_count):
    """Export TXT report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"surveillance_report_{timestamp}.txt"
    filepath = Path(filename)

    try:
        with open(filepath, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("AI SURVEILLANCE SYSTEM REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("SUMMARY\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Alerts: {alert_count}\n")
            f.write(f"Screenshots: {screenshots_count}\n")
            f.write(f"Unique Events: {len(summary)}\n\n")
            f.write("DETECTED OBJECTS\n")
            f.write("-" * 30 + "\n")
            for obj, data in sorted(summary.items()):
                risk = "HIGH" if obj in THREAT_OBJECTS else "LOW"
                f.write(f"{obj.upper()}\n")
                f.write(f"  Count: {data['count']}\n")
                f.write(f"  Max Threat: {data['max_threat']}%\n")
                f.write(f"  Risk Level: {risk}\n\n")
        return filepath
    except:
        return None

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.title("⚙️ Controls")
    mode = st.radio("📋 Select Mode:", ["📷 Live Camera", "📤 Upload Video"])
    st.markdown("---")
    st.markdown("**System Status:**")
    if st.session_state.model:
        st.success("✓ Model Loaded")
    else:
        st.error("✗ Model Error")
    st.markdown("---")
    st.markdown("**Alarm Settings:**")
    st.session_state.alarm_on = st.checkbox("🔊 Enable Alarm", value=st.session_state.alarm_on)

# ============================================================
# LIVE CAMERA MODE
# ============================================================
if mode == "📷 Live Camera":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📹 Live Camera Feed")
    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("▶ Start Camera", use_container_width=True)
    with col2:
        stop_btn = st.button("⏹ Stop Camera", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if start_btn:
        st.session_state.run_camera = True
        st.session_state.alert_count = 0
        st.session_state.screenshots = []
        st.session_state.suspicious_events = []
        st.session_state.threat_history.clear()
        st.session_state.timestamps.clear()
        st.session_state.recording_active = True

    if stop_btn:
        st.session_state.run_camera = False
        st.session_state.recording_active = False

    frame_placeholder = st.empty()
    status_placeholder = st.empty()
    metrics_placeholder = st.empty()

    if st.session_state.run_camera:
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Camera not available")
                st.session_state.run_camera = False
            else:
                status_placeholder.info("🔄 Camera active - Processing...")

                cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                cap.set(cv2.CAP_PROP_FPS, 30)

                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = None
                output_path = Path("output.avi")
                frame_count = 0
                last_screenshot_time = 0.0
                last_alarm_time = 0.0
                alarm_thread = None

                while st.session_state.run_camera:
                    success, frame = cap.read()
                    if not success or frame is None:
                        break

                    frame_count += 1
                    current_time = time.time()

                    if out is None and st.session_state.recording_active:
                        h, w = frame.shape[:2]
                        out = cv2.VideoWriter(str(output_path), fourcc, 20.0, (w, h))

                    # Process frame
                    processed_frame, detections, threat_score = process_frame(frame.copy(), st.session_state.model)

                    # Update threat history
                    st.session_state.threat_history.append(threat_score)
                    st.session_state.timestamps.append(current_time)

                    # Handle alerts
                    if threat_score > ALERT_THREAT_THRESHOLD:
                        st.session_state.alert_count += 1

                        # Add to suspicious events
                        for det in detections:
                            if det["object_name"] in THREAT_OBJECTS:
                                st.session_state.suspicious_events.append({
                                    "object_name": det["object_name"],
                                    "confidence": int(det["confidence"] * 100),
                                    "time": current_time
                                })

                        # Play alarm
                        if st.session_state.alarm_on and (alarm_thread is None or not alarm_thread.is_alive()):
                            if current_time - last_alarm_time >= ALARM_COOLDOWN:
                                last_alarm_time = current_time
                                alarm_thread = threading.Thread(target=play_alarm, daemon=True)
                                alarm_thread.start()

                        # Capture screenshot
                        if len(st.session_state.screenshots) < MAX_SCREENSHOTS:
                            if current_time - last_screenshot_time >= SCREENSHOT_GAP:
                                screenshot_path = Path("screenshots") / f"alert_{frame_count}_{int(current_time * 1000)}.jpg"
                                Path("screenshots").mkdir(exist_ok=True)
                                try:
                                    cv2.imwrite(str(screenshot_path), processed_frame)
                                    st.session_state.screenshots.append(str(screenshot_path))
                                    last_screenshot_time = current_time
                                except:
                                    pass

                    # Write to video
                    if out is not None:
                        out.write(processed_frame)

                    # Display frame
                    frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame_rgb, use_container_width=True)

                    # Update metrics
                    with metrics_placeholder.container():
                        m1, m2, m3 = st.columns(3)
                        with m1:
                            st.markdown('<div class="card">', unsafe_allow_html=True)
                            st.markdown("### 🚨 Alerts")
                            st.markdown(f'<div class="metric">{st.session_state.alert_count}</div>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        with m2:
                            st.markdown('<div class="card">', unsafe_allow_html=True)
                            st.markdown("### ⚠️ Events")
                            st.markdown(f'<div class="metric">{len(st.session_state.suspicious_events)}</div>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        with m3:
                            st.markdown('<div class="card">', unsafe_allow_html=True)
                            st.markdown("### 📸 Screenshots")
                            st.markdown(f'<div class="metric">{len(st.session_state.screenshots)}</div>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)

                    time.sleep(0.03)  # Small delay for responsiveness

                cap.release()
                if out is not None:
                    out.release()
                st.session_state.video_output_path = str(output_path)
                status_placeholder.success("✅ Camera stopped - Video saved")

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.session_state.run_camera = False

    # Results section
    if not st.session_state.run_camera and st.session_state.alert_count > 0:
        st.markdown("---")

        # Video download
        st.markdown("## 📹 Recorded Video")
        if st.session_state.video_output_path and Path(st.session_state.video_output_path).exists():
            with open(st.session_state.video_output_path, "rb") as f:
                st.download_button(
                    label="📥 Download Recorded Video",
                    data=f.read(),
                    file_name="surveillance_recording.avi",
                    mime="video/x-msvideo",
                    use_container_width=True
                )

        # Analytics
        st.markdown("---")
        st.markdown("## 📊 Analytics Dashboard")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 🚨 Total Alerts")
            st.markdown(f'<div class="metric">{st.session_state.alert_count}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            summary = summarize_events(st.session_state.suspicious_events)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### ⚠️ Unique Events")
            st.markdown(f'<div class="metric">{len(summary)}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 📸 Screenshots")
            st.markdown(f'<div class="metric">{len(st.session_state.screenshots)}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Threat level
        if summary:
            max_threat = max([data["max_threat"] for data in summary.values()], default=0)
            if max_threat >= 70:
                st.markdown('<div class="threat-high">🔴 HIGH THREAT DETECTED</div>', unsafe_allow_html=True)
            elif max_threat >= 50:
                st.markdown('<div class="threat-medium">🟡 MEDIUM RISK</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="threat-safe">🟢 SAFE</div>', unsafe_allow_html=True)

        # Threat graph
        st.markdown("---")
        st.markdown("## 📈 Threat Timeline")
        if st.session_state.threat_history:
            threat_df = {
                "Time": list(range(len(st.session_state.threat_history))),
                "Threat %": list(st.session_state.threat_history)
            }
            st.line_chart(threat_df, x="Time", y="Threat %")
            avg_threat = np.mean(list(st.session_state.threat_history))
            st.markdown(f"**Average Threat:** {avg_threat:.1f}%")

        # Event summary
        if summary:
            st.markdown("---")
            st.markdown("## 📋 Event Summary")
            cols = st.columns(2)
            items = list(summary.items())
            for i in range(0, len(items), 2):
                for j in range(2):
                    idx = i + j
                    if idx < len(items):
                        obj, data = items[idx]
                        with cols[j]:
                            st.markdown(f'''<div class="event-card">
                            <b>{obj.upper()}</b><br>
                            🔍 Count: {data["count"]}<br>
                            ⚡ Max Threat: {data["max_threat"]}%<br>
                            🚨 Risk: {"HIGH" if obj in THREAT_OBJECTS else "LOW"}
                            </div>''', unsafe_allow_html=True)

        # Export reports
        st.markdown("---")
        st.markdown("## 📄 Export Reports")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Export CSV", use_container_width=True):
                if summary:
                    csv_file = export_csv(summary, st.session_state.alert_count, len(st.session_state.screenshots))
                    if csv_file:
                        with open(csv_file, "rb") as f:
                            st.download_button("📥 Download CSV", f.read(), file_name=csv_file.name, mime="text/csv")
        with col2:
            if st.button("📝 Export TXT", use_container_width=True):
                if summary:
                    txt_file = export_txt(summary, st.session_state.alert_count, len(st.session_state.screenshots))
                    if txt_file:
                        with open(txt_file, "rb") as f:
                            st.download_button("📥 Download TXT", f.read(), file_name=txt_file.name, mime="text/plain")

        # Screenshots
        if st.session_state.screenshots:
            st.markdown("---")
            st.markdown("## 📸 Alert Screenshots")
            cols = st.columns(3)
            for i, path in enumerate(st.session_state.screenshots):
                try:
                    img = cv2.imread(path)
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        cols[i % 3].image(img_rgb, caption=f"Alert {i+1}", use_container_width=True)
                except:
                    pass

# ============================================================
# UPLOAD VIDEO MODE
# ============================================================
elif mode == "📤 Upload Video":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📤 Upload & Analyze Video")
    st.markdown('</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose video file", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file is not None:
        if st.button("🔍 Analyze Video", use_container_width=True):
            temp_dir = Path(tempfile.gettempdir())
            temp_video = temp_dir / uploaded_file.name

            with open(temp_video, "wb") as f:
                f.write(uploaded_file.getbuffer())

            status = st.empty()
            progress_bar = st.progress(0)
            status.info("🔄 Processing video...")

            try:
                cap = cv2.VideoCapture(str(temp_video))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                frame_count = 0
                detections_list = []
                threat_history = []
                alert_count = 0
                screenshots = []
                last_screenshot_time = 0.0

                while True:
                    success, frame = cap.read()
                    if not success:
                        break

                    frame_count += 1
                    current_time = time.time()

                    processed_frame, detections, threat_score = process_frame(frame.copy(), st.session_state.model)

                    detections_list.extend(detections)
                    threat_history.append(threat_score)

                    if threat_score > ALERT_THREAT_THRESHOLD:
                        alert_count += 1
                        if len(screenshots) < MAX_SCREENSHOTS and current_time - last_screenshot_time >= SCREENSHOT_GAP:
                            screenshot_path = Path("screenshots") / f"upload_alert_{frame_count}.jpg"
                            Path("screenshots").mkdir(exist_ok=True)
                            try:
                                cv2.imwrite(str(screenshot_path), processed_frame)
                                screenshots.append(str(screenshot_path))
                                last_screenshot_time = current_time
                            except:
                                pass

                    progress_bar.progress(min(frame_count / total_frames, 1.0))

                cap.release()

                # Store results
                st.session_state.alert_count = alert_count
                st.session_state.screenshots = screenshots
                st.session_state.threat_history = deque(threat_history, maxlen=100)
                st.session_state.suspicious_events = [
                    {"object_name": d["object_name"], "confidence": int(d["confidence"] * 100)}
                    for d in detections_list if d["object_name"] in THREAT_OBJECTS
                ]

                status.success("✅ Analysis complete!")

            except Exception as e:
                st.error(f"❌ Error: {e}")
            finally:
                if temp_video.exists():
                    temp_video.unlink()

        # Display results
        if st.session_state.alert_count > 0:
            st.markdown("---")
            st.markdown("## 📊 Analysis Results")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### 🚨 Alerts")
                st.markdown(f'<div class="metric">{st.session_state.alert_count}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                summary = summarize_events(st.session_state.suspicious_events)
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### ⚠️ Events")
                st.markdown(f'<div class="metric">{len(summary)}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with col3:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### 📸 Screenshots")
                st.markdown(f'<div class="metric">{len(st.session_state.screenshots)}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # Threat graph
            st.markdown("---")
            st.markdown("## 📈 Threat Timeline")
            if st.session_state.threat_history:
                threat_df = {
                    "Frame": list(range(len(st.session_state.threat_history))),
                    "Threat %": list(st.session_state.threat_history)
                }
                st.line_chart(threat_df, x="Frame", y="Threat %")
                avg_threat = np.mean(list(st.session_state.threat_history))
                st.markdown(f"**Average Threat:** {avg_threat:.1f}%")

            # Event summary
            if summary:
                st.markdown("---")
                st.markdown("## 📋 Event Summary")
                cols = st.columns(2)
                items = list(summary.items())
                for i in range(0, len(items), 2):
                    for j in range(2):
                        idx = i + j
                        if idx < len(items):
                            obj, data = items[idx]
                            with cols[j]:
                                st.markdown(f'''<div class="event-card">
                                <b>{obj.upper()}</b><br>
                                🔍 Count: {data["count"]}<br>
                                ⚡ Max Threat: {data["max_threat"]}%<br>
                                🚨 Risk: {"HIGH" if obj in THREAT_OBJECTS else "LOW"}
                                </div>''', unsafe_allow_html=True)

            # Export
            st.markdown("---")
            st.markdown("## 📄 Export Reports")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊 Export CSV", use_container_width=True):
                    if summary:
                        csv_file = export_csv(summary, st.session_state.alert_count, len(st.session_state.screenshots))
                        if csv_file:
                            with open(csv_file, "rb") as f:
                                st.download_button("📥 Download CSV", f.read(), file_name=csv_file.name, mime="text/csv")
            with col2:
                if st.button("📝 Export TXT", use_container_width=True):
                    if summary:
                        txt_file = export_txt(summary, st.session_state.alert_count, len(st.session_state.screenshots))
                        if txt_file:
                            with open(txt_file, "rb") as f:
                                st.download_button("📥 Download TXT", f.read(), file_name=txt_file.name, mime="text/plain")

            # Screenshots
            if st.session_state.screenshots:
                st.markdown("---")
                st.markdown("## 📸 Alert Screenshots")
                cols = st.columns(3)
                for i, path in enumerate(st.session_state.screenshots):
                    try:
                        img = cv2.imread(path)
                        if img is not None:
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            cols[i % 3].image(img_rgb, caption=f"Alert {i+1}", use_container_width=True)
                    except:
                        pass
    else:
        st.info("👆 Upload a video file to analyze")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 12px; margin-top: 30px; padding: 20px;'>
        <b>🎥 AI Surveillance System v3.0</b><br>
        Advanced YOLO Detection + Real-time Threat Analysis + Video Recording<br>
        <span style='font-size: 10px;'>© 2026 | Enterprise Security Intelligence</span>
    </div>
    """,
    unsafe_allow_html=True
)