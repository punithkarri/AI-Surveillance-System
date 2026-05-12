import os
import threading
from pathlib import Path

# winsound is Windows-only — safe to import conditionally
try:
    import winsound
except ImportError:
    winsound = None

# playsound is INTENTIONALLY NOT imported here.
# It is incompatible with Python 3.13 on Linux (Hugging Face Spaces).
# On Windows, winsound.Beep() is used instead.
# On Linux/cloud, audio is silently disabled — the app never crashes.

from config.settings import ALARM_SOUND_FILE, ALARM_COOLDOWN_SECONDS
from utils.helpers import ensure_folder, get_timestamp_string


def _play_windows_alarm():
    """Play a short beep sequence using winsound (Windows only)."""
    try:
        if winsound:
            winsound.Beep(1000, 300)
            winsound.Beep(1200, 300)
    except Exception:
        pass


def _play_linux_alarm():
    """Audio on Linux/cloud is silently disabled — no crash, no error."""
    pass


def play_alarm():
    """
    Play an alarm sound in a background thread.

    - Windows:     uses winsound.Beep (no extra dependency)
    - Linux/Cloud: silently skipped (HF Spaces / deployment mode)
    """
    if os.name == "nt":
        threading.Thread(target=_play_windows_alarm, daemon=True).start()
    else:
        # Cloud/Linux — audio not available, do nothing
        pass


def ensure_alarm_sound() -> Path:
    if ALARM_SOUND_FILE.exists():
        return ALARM_SOUND_FILE
    raise FileNotFoundError(f"Alarm sound missing: {ALARM_SOUND_FILE}")


def capture_snapshot(frame, output_dir, name_prefix="alert"):
    output_dir = ensure_folder(output_dir)
    filename = f"{name_prefix}_{get_timestamp_string()}.jpg"
    path = output_dir / filename
    try:
        import cv2
        cv2.imwrite(str(path), frame)
    except Exception:
        return None
    return path


def should_alert(previous_threat: int, current_threat: int, threshold: int) -> bool:
    return current_threat > threshold and previous_threat <= threshold
