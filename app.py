import streamlit as st
import cv2
import tempfile
import threading
import time
import os
import pandas as pd
from pathlib import Path
from collections import deque
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import io

try:
    import winsound
except ImportError:
    winsound = None

# ============================================================
# CONFIGURATION
# ============================================================
ALLOWED_CLASSES = ["person", "cell phone", "laptop", "book", "tablet", "bottle"]
THREAT_CLASSES = {"cell phone": 85, "laptop": 70, "book": 55, "tablet": 65}
CONFIDENCE_THRESHOLD = 0.6
ALERT_THREAT_THRESHOLD = 70
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="AI Surveillance", page_icon="🎥", layout="wide")

st.markdown("""
<style>
.main { background-color: #f5f7fa; }
.card { background-color: white; padding: 20px; border-radius: 15px; box-shadow: 0px 4px 10px rgba(0,0,0,0.1); margin: 10px 0; }
.metric { font-size: 36px; font-weight: bold; color: #1f77b4; }
.threat-high { background-color: #f8d7da; padding: 15px; border-radius: 8px; color: #721c24; font-weight: bold; }
.threat-medium { background-color: #fff3cd; padding: 15px; border-radius: 8px; color: #856404; font-weight: bold; }
.threat-safe { background-color: #d4edda; padding: 15px; border-radius: 8px; color: #155724; font-weight: bold; }
.event-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; border-radius: 10px; margin: 8px 0; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================
def init_session_state():
    """Initialize all session variables."""
    if "running" not in st.session_state:
        st.session_state.running = False
    if "model" not in st.session_state:
        st.session_state.model = None
    if "alert_count" not in st.session_state:
        st.session_state.alert_count = 0
    if "suspicious_events" not in st.session_state:
        st.session_state.suspicious_events = []
    if "threat_history" not in st.session_state:
        st.session_state.threat_history = deque(maxlen=100)
    if "screenshots" not in st.session_state:
        st.session_state.screenshots = []
    if "video_writer" not in st.session_state:
        st.session_state.video_writer = None
    if "cap" not in st.session_state:
        st.session_state.cap = None
    if "alarm_active" not in st.session_state:
        st.session_state.alarm_active = False
    if "last_threat_score" not in st.session_state:
        st.session_state.last_threat_score = 0

init_session_state()

# ============================================================
# UTILITIES
# ============================================================
def load_model():
    """Load YOLO model."""
    if st.session_state.model is None:
        try:
            st.session_state.model = YOLO("yolov8s.pt")
        except Exception as e:
            st.error(f"❌ Failed to load model: {e}")
    return st.session_state.model

def play_alarm():
    """Play beep sound."""
    if winsound:
        try:
            winsound.Beep(1000, 300)
            time.sleep(0.1)
            winsound.Beep(1200, 300)
        except:
            pass

def calculate_threat(detections):
    """Calculate threat 0-100."""
    threat = 0
    for det in detections:
        label = det["label"]
        if label in THREAT_CLASSES:
            threat += THREAT_CLASSES[label]
        elif label == "person":
            threat += 20
        else:
            threat += 10
    return min(threat, 100)

def draw_boxes(frame, detections):
    """Draw bounding boxes."""
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        conf = det["conf"]
        
        # Color
        if label in THREAT_CLASSES:
            color = (0, 0, 255)  # RED
        elif label == "person":
            color = (0, 255, 0)  # GREEN
        else:
            color = (255, 0, 0)  # BLUE
        
        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # Label
        label_text = f"{label} {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(label_text, font, 0.6, 2)[0]
        cv2.rectangle(frame, (x1, y1 - 30), (x1 + text_size[0] + 10, y1), color, -1)
        cv2.putText(frame, label_text, (x1 + 5, y1 - 8), font, 0.6, (255, 255, 255), 2)
    
    return frame

def process_frame(frame, model):
    """Detect objects in frame."""
    if frame is None or model is None:
        return frame, [], 0
    
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    
    try:
        results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
    except:
        return frame, [], 0
    
    detections = []
    
    for result in results:
        if not hasattr(result, "boxes"):
            continue
        
        for box in result.boxes:
            try:
                xyxy = box.xyxy[0].tolist() if hasattr(box.xyxy, 'tolist') else list(box.xyxy[0])
                x1, y1, x2, y2 = map(int, xyxy[:4])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                if conf < CONFIDENCE_THRESHOLD:
                    continue
                
                label = model.names[cls_id] if hasattr(model, "names") else str(cls_id)
                
                # Fix phone/laptop confusion
                if label == "laptop":
                    area = (x2 - x1) * (y2 - y1)
                    if area < 5000:
                        label = "cell phone"
                
                if label not in ALLOWED_CLASSES:
                    continue
                
                detections.append({
                    "label": label,
                    "conf": conf,
                    "bbox": (x1, y1, x2, y2)
                })
            except:
                continue
    
    return frame, detections, calculate_threat(detections)

def export_csv_report(event_summary, total_alerts, screenshot_count):
    """Export CSV."""
    rows = []
    for event, data in event_summary.items():
        risk = "HIGH" if event in THREAT_CLASSES else "LOW"
        rows.append({
            "Event": event,
            "Count": data["count"],
            "Max Threat": data["max_threat"],
            "Risk": risk
        })
    
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode()

def export_txt_report(event_summary, total_alerts, screenshot_count):
    """Export TXT."""
    text = "=" * 60 + "\n"
    text += "AI SURVEILLANCE REPORT\n"
    text += "=" * 60 + "\n\n"
    text += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    text += f"Total Alerts: {total_alerts}\n"
    text += f"Screenshots: {screenshot_count}\n\n"
    text += "THREATS DETECTED:\n"
    text += "-" * 60 + "\n"
    
    for event, data in event_summary.items():
        risk = "HIGH" if event in THREAT_CLASSES else "LOW"
        text += f"\n{event.upper()}\n"
        text += f"  Detections: {data['count']}\n"
        text += f"  Max Threat: {data['max_threat']}%\n"
        text += f"  Risk: {risk}\n"
    
    return text.encode()

def summarize_events(events):
    """Summarize detections."""
    summary = {}
    for event in events:
        label = event.get("label", "unknown")
        if label not in summary:
            summary[label] = {"count": 1, "max_threat": event.get("threat", 0)}
        else:
            summary[label]["count"] += 1
            summary[label]["max_threat"] = max(summary[label]["max_threat"], event.get("threat", 0))
    return summary

# ============================================================
# UI HEADER
# ============================================================
st.title("🎥 AI Surveillance System")
st.markdown("Real-time YOLO detection + threat analysis")
st.markdown("---")

# ============================================================
# SIDEBAR (UNIQUE KEYS)
# ============================================================
with st.sidebar:
    st.title("⚙️ Controls")
    mode = st.radio("📋 Select Mode:", ["📷 Live Camera", "📤 Upload Video"], key="main_mode_radio")
    st.markdown("---")
    
    st.markdown("**System Status:**")
    model = load_model()
    if model:
        st.success("✓ Model Loaded (YOLOv8s)")
    else:
        st.error("✗ Model Failed")
    
    st.markdown("---")
    st.markdown("**Settings:**")
    alarm_enabled = st.checkbox("🔊 Enable Alarm", value=True, key="alarm_checkbox_main")

# ============================================================
# LIVE CAMERA MODE
# ============================================================
if mode == "📷 Live Camera":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📹 Live Feed")
    
    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("▶ Start Camera", key="btn_start", use_container_width=True)
    with col2:
        stop_btn = st.button("⏹ Stop Camera", key="btn_stop", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if start_btn:
        st.session_state.running = True
        st.session_state.alert_count = 0
        st.session_state.suspicious_events = []
        st.session_state.threat_history.clear()
        st.session_state.screenshots = []
    
    if stop_btn:
        st.session_state.running = False
        if st.session_state.video_writer is not None:
            st.session_state.video_writer.release()
            st.session_state.video_writer = None
        if st.session_state.cap is not None:
            st.session_state.cap.release()
            st.session_state.cap = None
    
    frame_placeholder = st.empty()
    status_placeholder = st.empty()
    metrics_placeholder = st.empty()
    
    if st.session_state.running:
        try:
            if st.session_state.cap is None:
                st.session_state.cap = cv2.VideoCapture(0)
                st.session_state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                st.session_state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            
            cap = st.session_state.cap
            
            if not cap.isOpened():
                st.error("❌ Camera unavailable")
                st.session_state.running = False
            else:
                status_placeholder.info("🔄 Recording...")
                
                # Video writer
                if st.session_state.video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    st.session_state.video_writer = cv2.VideoWriter(
                        "output.avi", 
                        fourcc, 
                        20.0, 
                        (FRAME_WIDTH, FRAME_HEIGHT)
                    )
                
                frame_limit = 30 * 10  # 10 seconds
                frame_num = 0
                
                while st.session_state.running and frame_num < frame_limit:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_num += 1
                    current_time = time.time()
                    
                    # Process
                    processed_frame, detections, threat = process_frame(frame.copy(), model)
                    processed_frame = draw_boxes(processed_frame, detections)
                    
                    # Write video
                    if st.session_state.video_writer is not None:
                        st.session_state.video_writer.write(processed_frame)
                    
                    # Threat history
                    st.session_state.threat_history.append(threat)
                    
                    # Alert spike detection
                    alert_spike = (threat > ALERT_THREAT_THRESHOLD and 
                                 st.session_state.last_threat_score <= ALERT_THREAT_THRESHOLD)
                    
                    if alert_spike:
                        st.session_state.alert_count += 1
                        
                        # Add events
                        for det in detections:
                            if det["label"] in THREAT_CLASSES:
                                st.session_state.suspicious_events.append({
                                    "label": det["label"],
                                    "threat": int(det["conf"] * 100),
                                    "time": current_time
                                })
                        
                        # Alarm
                        if alarm_enabled and not st.session_state.alarm_active:
                            st.session_state.alarm_active = True
                            threading.Thread(target=play_alarm, daemon=True).start()
                        
                        # Screenshot
                        if len(st.session_state.screenshots) < 10:
                            Path("screenshots").mkdir(exist_ok=True)
                            path = f"screenshots/screenshot_{int(current_time * 1000)}.jpg"
                            try:
                                cv2.imwrite(path, processed_frame)
                                st.session_state.screenshots.append(path)
                            except:
                                pass
                    else:
                        st.session_state.alarm_active = False
                    
                    st.session_state.last_threat_score = threat
                    
                    # Display
                    frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame_rgb, use_container_width=True)
                    
                    # Metrics
                    with metrics_placeholder.container():
                        m1, m2, m3 = st.columns(3)
                        with m1:
                            st.metric("🚨 Alerts", st.session_state.alert_count)
                        with m2:
                            st.metric("⚠️ Events", len(st.session_state.suspicious_events))
                        with m3:
                            st.metric("📸 Screenshots", len(st.session_state.screenshots))
                    
                    time.sleep(0.03)
                
                status_placeholder.success("✅ Recording stopped")
        
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.session_state.running = False
    
    # Results
    if not st.session_state.running and st.session_state.alert_count > 0:
        st.markdown("---")
        
        # Download video
        if os.path.exists("output.avi") and os.path.getsize("output.avi") > 0:
            st.markdown("## 📹 Recorded Video")
            with open("output.avi", "rb") as f:
                st.download_button(
                    "⬇️ Download Video",
                    f.read(),
                    "surveillance.avi",
                    "video/x-msvideo",
                    use_container_width=True,
                    key="dl_video_live"
                )
        
        st.markdown("---")
        st.markdown("## 📊 Analytics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🚨 Alerts", st.session_state.alert_count)
        with col2:
            st.metric("⚠️ Events", len(set(e["label"] for e in st.session_state.suspicious_events)))
        with col3:
            st.metric("📸 Screenshots", len(st.session_state.screenshots))
        
        # Timeline
        if st.session_state.threat_history:
            st.markdown("### 📈 Threat Timeline")
            df = pd.DataFrame({
                "Frame": range(len(st.session_state.threat_history)),
                "Threat %": list(st.session_state.threat_history)
            })
            st.line_chart(df, x="Frame", y="Threat %", use_container_width=True)
        
        # Events
        if st.session_state.suspicious_events:
            st.markdown("### 📋 Events")
            summary = summarize_events(st.session_state.suspicious_events)
            
            cols = st.columns(2)
            for i, (event, data) in enumerate(summary.items()):
                with cols[i % 2]:
                    risk = "🔴 HIGH" if event in THREAT_CLASSES else "🟡 LOW"
                    st.markdown(f"**{event}** | {data['count']}x | {data['max_threat']}% | {risk}")
        
        # Export
        st.markdown("---")
        st.markdown("## 📄 Export Report")
        
        col1, col2 = st.columns(2)
        if st.session_state.suspicious_events:
            summary = summarize_events(st.session_state.suspicious_events)
            
            with col1:
                csv = export_csv_report(summary, st.session_state.alert_count, len(st.session_state.screenshots))
                st.download_button(
                    "⬇ Download CSV Report",
                    csv,
                    "report.csv",
                    "text/csv",
                    use_container_width=True,
                    key="dl_csv_live"
                )
            
            with col2:
                txt = export_txt_report(summary, st.session_state.alert_count, len(st.session_state.screenshots))
                st.download_button(
                    "⬇ Download Text Report",
                    txt,
                    "report.txt",
                    "text/plain",
                    use_container_width=True,
                    key="dl_txt_live"
                )
        
        # Screenshots
        if st.session_state.screenshots:
            st.markdown("---")
            st.markdown("## 📸 Screenshots")
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
    st.subheader("📤 Analyze Video")
    st.markdown('</div>', unsafe_allow_html=True)
    
    file = st.file_uploader("Choose video", type=["mp4", "avi", "mov", "mkv"], key="upload_main")
    
    if file is not None:
        if st.button("🔍 Analyze", key="btn_analyze", use_container_width=True):
            temp = Path(tempfile.gettempdir()) / file.name
            with open(temp, "wb") as f:
                f.write(file.getbuffer())
            
            status = st.empty()
            prog = st.progress(0)
            status.info("🔄 Processing...")
            
            try:
                cap = cv2.VideoCapture(str(temp))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                frame_num = 0
                alerts = 0
                events = []
                threats = []
                shots = []
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_num += 1
                    
                    _, detections, threat = process_frame(frame.copy(), model)
                    threats.append(threat)
                    
                    if threat > ALERT_THREAT_THRESHOLD:
                        alerts += 1
                        for det in detections:
                            if det["label"] in THREAT_CLASSES:
                                events.append({
                                    "label": det["label"],
                                    "threat": int(det["conf"] * 100)
                                })
                        
                        if len(shots) < 10:
                            Path("screenshots").mkdir(exist_ok=True)
                            path = f"screenshots/upload_{int(time.time() * 1000)}.jpg"
                            cv2.imwrite(path, frame)
                            shots.append(path)
                    
                    prog.progress(min(frame_num / total, 1.0))
                
                cap.release()
                
                st.session_state.alert_count = alerts
                st.session_state.suspicious_events = events
                st.session_state.threat_history = deque(threats[-100:], maxlen=100)
                st.session_state.screenshots = shots
                
                status.success("✅ Done!")
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
            finally:
                if temp.exists():
                    temp.unlink()
        
        # Results
        if st.session_state.alert_count > 0:
            st.markdown("---")
            st.markdown("## 📊 Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🚨 Alerts", st.session_state.alert_count)
            with col2:
                st.metric("⚠️ Events", len(set(e["label"] for e in st.session_state.suspicious_events)))
            with col3:
                st.metric("📸 Screenshots", len(st.session_state.screenshots))
            
            # Timeline
            if st.session_state.threat_history:
                st.markdown("### 📈 Timeline")
                df = pd.DataFrame({
                    "Frame": range(len(st.session_state.threat_history)),
                    "Threat %": list(st.session_state.threat_history)
                })
                st.line_chart(df, x="Frame", y="Threat %", use_container_width=True)
            
            # Events
            if st.session_state.suspicious_events:
                st.markdown("### 📋 Events")
                summary = summarize_events(st.session_state.suspicious_events)
                
                cols = st.columns(2)
                for i, (event, data) in enumerate(summary.items()):
                    with cols[i % 2]:
                        risk = "🔴 HIGH" if event in THREAT_CLASSES else "🟡 LOW"
                        st.markdown(f"**{event}** | {data['count']}x | {data['max_threat']}% | {risk}")
            
            # Export
            st.markdown("---")
            st.markdown("## 📄 Export")
            
            col1, col2 = st.columns(2)
            if st.session_state.suspicious_events:
                summary = summarize_events(st.session_state.suspicious_events)
                
                with col1:
                    csv = export_csv_report(summary, st.session_state.alert_count, len(st.session_state.screenshots))
                    st.download_button(
                        "📊 CSV",
                        csv,
                        "report.csv",
                        "text/csv",
                        use_container_width=True,
                        key="dl_csv_upload"
                    )
                
                with col2:
                    txt = export_txt_report(summary, st.session_state.alert_count, len(st.session_state.screenshots))
                    st.download_button(
                        "📝 TXT",
                        txt,
                        "report.txt",
                        "text/plain",
                        use_container_width=True,
                        key="dl_txt_upload"
                    )
            
            # Screenshots
            if st.session_state.screenshots:
                st.markdown("---")
                st.markdown("## 📸 Screenshots")
                cols = st.columns(3)
                for i, path in enumerate(st.session_state.screenshots):
                    try:
                        img = cv2.imread(path)
                        if img is not None:
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            cols[i % 3].image(img_rgb, caption=f"Alert {i+1}", use_container_width=True)
                    except:
                        pass

st.markdown("---")
st.markdown("<div style='text-align:center;color:gray;font-size:11px;'><b>🎥 AI Surveillance v4.0 FIXED</b><br>Production-Ready | YOLO | Real-time Analysis</div>", unsafe_allow_html=True)
