"""
SurveillanceProcessor — streamlit-webrtc VideoProcessorBase subclass.

This runs in a BACKGROUND THREAD (not the Streamlit main thread).
All shared state is protected by a threading.Lock.

Usage:
    from services.webrtc_processor import SurveillanceProcessor, SharedState
"""

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import av
import cv2
import numpy as np

from config.settings import (
    CONFIDENCE_THRESHOLD,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MAX_SCREENSHOTS,
    SNAPSHOT_OUTPUT_DIR,
    THREAT_THRESHOLD,
    THREAT_CLASS_SCORES,
    ALLOWED_CLASSES,
)
from services.detection import process_frame
from services.alerts import capture_snapshot, should_alert

try:
    from streamlit_webrtc import VideoProcessorBase
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    # Stub so the module can be imported safely even without streamlit-webrtc
    class VideoProcessorBase:
        def recv(self, frame):
            return frame


@dataclass
class SharedState:
    """Thread-safe container for live metrics shared between WebRTC BG thread and Streamlit main thread."""
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # Metrics (written by BG thread, read by main thread)
    threat_history: List[int] = field(default_factory=list)
    suspicious_events: List[dict] = field(default_factory=list)
    alert_count: int = 0
    screenshots: List[str] = field(default_factory=list)
    last_threat_score: int = 0
    last_frame_ts: float = 0.0
    new_alert: bool = False  # flag: main thread reads this to trigger visual alarm

    def snapshot(self):
        """Return a consistent copy of the current state (main thread safe)."""
        with self._lock:
            return {
                "threat_history": list(self.threat_history),
                "suspicious_events": list(self.suspicious_events),
                "alert_count": self.alert_count,
                "screenshots": list(self.screenshots),
                "last_threat_score": self.last_threat_score,
                "new_alert": self.new_alert,
            }

    def clear_new_alert(self):
        with self._lock:
            self.new_alert = False

    def reset(self):
        with self._lock:
            self.threat_history.clear()
            self.suspicious_events.clear()
            self.alert_count = 0
            self.screenshots.clear()
            self.last_threat_score = 0
            self.last_frame_ts = 0.0
            self.new_alert = False


class SurveillanceProcessor(VideoProcessorBase):
    """
    WebRTC video processor.

    - Receives raw video frames from the browser via WebRTC.
    - Runs YOLO inference via process_frame().
    - Returns annotated frame (with bounding boxes) back to browser.
    - Updates SharedState for the Streamlit main thread to read.

    IMPORTANT: recv() runs in a background thread — never call st.* here.
    """

    def __init__(self, model, shared_state: SharedState):
        self.model = model
        self.state = shared_state
        self._last_screenshot_ts = 0.0
        self._alarm_cooldown_ts = 0.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Process one incoming frame, return annotated frame."""
        try:
            # Convert PyAV frame → numpy BGR (OpenCV format)
            img = frame.to_ndarray(format="bgr24")

            # Resize to our standard dimensions for consistent inference speed
            img_resized = cv2.resize(img, (FRAME_WIDTH, FRAME_HEIGHT))

            # Run YOLO inference
            annotated, detections, threat_score = process_frame(img_resized.copy(), self.model)

            # Update shared state (thread-safe)
            with self.state._lock:
                self.state.threat_history.append(threat_score)
                # Keep history bounded to prevent memory growth
                if len(self.state.threat_history) > 500:
                    self.state.threat_history = self.state.threat_history[-500:]

                prev_threat = self.state.last_threat_score
                self.state.last_threat_score = threat_score
                self.state.last_frame_ts = time.time()

                # Alert logic
                if should_alert(prev_threat, threat_score, THREAT_THRESHOLD):
                    self.state.alert_count += 1
                    self.state.new_alert = True
                    for det in detections:
                        self.state.suspicious_events.append({
                            "label": det["label"],
                            "confidence": int(det["confidence"] * 100),
                            "threat": int(det["confidence"] * 100),
                        })

                # Screenshot: max every 3 seconds, up to MAX_SCREENSHOTS
                now = time.time()
                if (
                    threat_score > THREAT_THRESHOLD
                    and len(self.state.screenshots) < MAX_SCREENSHOTS
                    and now - self._last_screenshot_ts > 3.0
                ):
                    path = capture_snapshot(annotated, SNAPSHOT_OUTPUT_DIR, "webrtc_alert")
                    if path:
                        self.state.screenshots.append(str(path))
                        self._last_screenshot_ts = now

            # Convert annotated BGR frame back to RGB for WebRTC
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            return av.VideoFrame.from_ndarray(annotated_rgb, format="rgb24")

        except Exception:
            # Never crash the WebRTC stream — return the original frame unchanged
            return frame
