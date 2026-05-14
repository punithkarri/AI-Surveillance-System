import os
import threading
import time
from pathlib import Path

# winsound is Windows-only — safe to import conditionally
try:
    import winsound
except ImportError:
    winsound = None

# pygame for cross-platform audio playback
try:
    import pygame
    PYGAME_AVAILABLE = True
except Exception as e:
    print(f"pygame import failed: {e}")
    pygame = None
    PYGAME_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except Exception as e:
    print(f"numpy import failed: {e}")
    np = None
    NUMPY_AVAILABLE = False

from config.settings import ALARM_SOUND_FILE, ALARM_COOLDOWN_SECONDS
from utils.helpers import ensure_folder, get_timestamp_string

# Global alarm state
_alarm_playing = False
_alarm_thread = None
_mixer_initialized = False


def _initialize_mixer_once():
    global _mixer_initialized, PYGAME_AVAILABLE
    if _mixer_initialized or not PYGAME_AVAILABLE or pygame is None:
        return
    try:
        pygame.mixer.init()
        _mixer_initialized = True
        print("pygame mixer initialized")
    except Exception as e:
        print(f"pygame mixer initialization failed: {e}")
        PYGAME_AVAILABLE = False
        _mixer_initialized = False


def _make_beep_sound(duration=0.35, freq=880, volume=0.5, sample_rate=44100):
    if not NUMPY_AVAILABLE or not PYGAME_AVAILABLE:
        return None
    samples = np.arange(int(sample_rate * duration), dtype=np.float32)
    wave = np.sin(2.0 * np.pi * freq * samples / sample_rate)
    audio = np.int16(wave * volume * 32767)
    return pygame.sndarray.make_sound(audio)


def _play_pygame_alarm():
    """Play alarm sound using pygame.mixer."""
    global _alarm_playing
    _initialize_mixer_once()
    try:
        sound = None
        if PYGAME_AVAILABLE and _mixer_initialized:
            print("Alarm path:", ALARM_SOUND_FILE)
            if ALARM_SOUND_FILE.exists() and ALARM_SOUND_FILE.stat().st_size > 0:
                try:
                    sound = pygame.mixer.Sound(str(ALARM_SOUND_FILE))
                    print("Loaded alarm sound file")
                except Exception as e:
                    print(f"Failed to load alarm sound file: {e}")
                    sound = None
            else:
                print("Alarm sound file missing or empty", ALARM_SOUND_FILE)

        if sound is None and PYGAME_AVAILABLE and NUMPY_AVAILABLE:
            sound = _make_beep_sound()
            print("Using generated beep sound fallback")

        if sound is not None:
            sound.play(loops=-1)
            print("Playing alarm")
            while _alarm_playing:
                time.sleep(0.1)
            try:
                pygame.mixer.stop()
            except Exception:
                pass
        elif os.name == "nt":
            print("Falling back to winsound alarm")
            _play_windows_alarm()
        else:
            print("No alarm backend available")
            while _alarm_playing:
                time.sleep(0.25)
    except Exception as e:
        print(f"Alarm playback error: {e}")
        _play_windows_alarm()


def _play_windows_alarm():
    """Play a short beep sequence using winsound (Windows only)."""
    global _alarm_playing
    try:
        if winsound:
            while _alarm_playing:
                winsound.Beep(1000, 300)
                if not _alarm_playing:
                    break
                winsound.Beep(1200, 300)
                time.sleep(0.1)
    except Exception:
        pass


def start_alarm():
    """Start the alarm in a background thread."""
    global _alarm_playing, _alarm_thread

    if _alarm_playing:
        print("Alarm already playing, skip start")
        return
    _alarm_playing = True

    print("Starting alarm")
    if PYGAME_AVAILABLE:
        _alarm_thread = threading.Thread(target=_play_pygame_alarm, daemon=True)
        _alarm_thread.start()
    elif os.name == "nt":
        _alarm_thread = threading.Thread(target=_play_windows_alarm, daemon=True)
        _alarm_thread.start()
    else:
        print("No audio backend available to start alarm")


def stop_alarm():
    """Stop the currently playing alarm."""
    global _alarm_playing, _alarm_thread
    print("Stopping alarm")
    _alarm_playing = False

    if pygame and pygame.mixer.get_init():
        try:
            pygame.mixer.stop()
        except Exception:
            pass

    if _alarm_thread and _alarm_thread.is_alive():
        _alarm_thread.join(timeout=1.0)


def play_alarm():
    """Backward-compatible wrapper for starting the alarm."""
    start_alarm()


def ensure_alarm_sound() -> Path:
    if ALARM_SOUND_FILE.exists() and ALARM_SOUND_FILE.stat().st_size > 0:
        return ALARM_SOUND_FILE
    raise FileNotFoundError(f"Alarm sound missing or empty: {ALARM_SOUND_FILE}")


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
    return current_threat >= threshold and previous_threat < threshold
