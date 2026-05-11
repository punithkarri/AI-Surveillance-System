import os
import threading
from pathlib import Path

try:
    import winsound
except ImportError:
    winsound = None

try:
    from playsound import playsound
except ImportError:
    playsound = None

from config.settings import ALARM_SOUND_FILE, ALARM_COOLDOWN_SECONDS
from utils.helpers import ensure_folder, get_timestamp_string


def _play_windows_alarm():
    try:
        if winsound:
            winsound.Beep(1000, 300)
            winsound.Beep(1200, 300)
    except Exception:
        pass


def _play_file_alarm(sound_path: Path):
    if playsound and sound_path.exists():
        try:
            playsound(str(sound_path), block=False)
        except Exception:
            pass


def play_alarm():
    if os.name == "nt":
        _play_windows_alarm()
    else:
        _play_file_alarm(ALARM_SOUND_FILE)


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
