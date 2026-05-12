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
from typing import List

import av
import cv2

from config.settings import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MAX_SCREENSHOTS,
    SNAPSHOT_OUTPUT_DIR,
    THREAT_THRESHOLD,
)
from services.detection import process_frame
from services.alerts import capture_snapshot, should_alert

try:
    from streamlit_webrtc import VideoProcessorBase
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

    class VideoProcessorBase:  # type: ignore[misc]
        """Stub — allows module to import safely without streamlit-webrtc."""
        def recv(self, frame):
            return frame


@dataclass
class SharedState:
    """
    Thread-safe container for live metrics.

    Written by the WebRTC background thread (SurveillanceProcessor.recv).
    Read by the Streamlit main thread on every rerun.
    All access is guarded by _lock.
    """
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    threat_history: List[int] = field(default_factory=list)
    suspicious_events: List[dict] = field(default_factory=list)
    alert_count: int = 0
    screenshots: List[str] = field(default_factory=list)
    last_threat_score: int = 0
    last_frame_ts: float = 0.0
    new_alert: bool = False

    def snapshot(self) -> dict:
        """Return a consistent shallow copy — safe to call from main thread."""
        with self._lock:
            return {
                "threat_history": list(self.threat_history),
                "suspicious_events": list(self.suspicious_events),
                "alert_count": self.alert_count,
                "screenshots": list(self.screenshots),
                "last_threat_score": self.last_threat_score,
                "new_alert": self.new_alert,
            }

    def clear_new_alert(self) -> None:
        with self._lock:
            self.new_alert = False

    def reset(self) -> None:
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
    WebRTC video processor (runs in background thread).

    Receives raw BGR frames from the browser via WebRTC, runs YOLO inference,
    draws bounding boxes, returns annotated RGB frame back to the browser,
    and writes threat metrics to SharedState for the main thread to read.

    IMPORTANT: Never call st.* functions here — this is a background thread.
    """

    def __init__(self, model, shared_state: SharedState) -> None:
        self.model = model
        self.state = shared_state
        self._last_screenshot_ts: float = 0.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Process one incoming video frame and return the annotated version."""
        try:
            # PyAV VideoFrame → numpy BGR
            img = frame.to_ndarray(format="bgr24")

            # Resize for consistent, efficient inference
            img_resized = cv2.resize(img, (FRAME_WIDTH, FRAME_HEIGHT))

            # YOLO inference — returns (annotated_bgr, detections_list, threat_score_int)
            annotated, detections, threat_score = process_frame(img_resized.copy(), self.model)

            now = time.time()

            with self.state._lock:
                # Threat history (bounded to 500 entries to prevent memory growth)
                self.state.threat_history.append(threat_score)
                if len(self.state.threat_history) > 500:
                    self.state.threat_history = self.state.threat_history[-500:]

                prev_threat = self.state.last_threat_score
                self.state.last_threat_score = threat_score
                self.state.last_frame_ts = now

                # Rising-edge alert
                if should_alert(prev_threat, threat_score, THREAT_THRESHOLD):
                    self.state.alert_count += 1
                    self.state.new_alert = True
                    for det in detections:
                        self.state.suspicious_events.append({
                            "label": det["label"],
                            "confidence": int(det["confidence"] * 100),
                            "threat": int(det["confidence"] * 100),
                        })

                # Auto-screenshot: max every 3 s, cap at MAX_SCREENSHOTS
                if (
                    threat_score > THREAT_THRESHOLD
                    and len(self.state.screenshots) < MAX_SCREENSHOTS
                    and now - self._last_screenshot_ts > 3.0
                ):
                    path = capture_snapshot(annotated, SNAPSHOT_OUTPUT_DIR, "webrtc_alert")
                    if path:
                        self.state.screenshots.append(str(path))
                        self._last_screenshot_ts = now

            # BGR → RGB for WebRTC output
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            return av.VideoFrame.from_ndarray(annotated_rgb, format="rgb24")

        except Exception:
            # Never crash the WebRTC stream — return the original frame
            return frame
