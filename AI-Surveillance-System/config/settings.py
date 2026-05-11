import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:  # pragma: no cover
    pass

ROOT_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT_DIR / "assets"
MODELS_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"
VIDEO_OUTPUT_DIR = OUTPUT_DIR / "videos"
REPORT_OUTPUT_DIR = OUTPUT_DIR / "reports"
SNAPSHOT_OUTPUT_DIR = OUTPUT_DIR / "snapshots"
SCREENSHOTS_DIR = ROOT_DIR / "screenshots"

DEFAULT_MODEL_FILENAME = "yolov8n.pt"
DEFAULT_MODEL_PATH = MODELS_DIR / DEFAULT_MODEL_FILENAME
MODEL_PATH = DEFAULT_MODEL_PATH
FALLBACK_MODEL_NAME = "yolov8n"
ALARM_SOUND_FILE = ASSETS_DIR / "alarm.wav"
LOGO_FILE = ASSETS_DIR / "logo.png"

def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default

CONFIDENCE_THRESHOLD = _env_float("CONFIDENCE_THRESHOLD", 0.5)
THREAT_THRESHOLD = _env_int("THREAT_THRESHOLD", 70)
RECORDING_FPS = _env_int("RECORDING_FPS", 20)
FRAME_WIDTH = _env_int("FRAME_WIDTH", 640)
FRAME_HEIGHT = _env_int("FRAME_HEIGHT", 480)
MAX_SCREENSHOTS = _env_int("MAX_SCREENSHOTS", 10)
ALARM_COOLDOWN_SECONDS = _env_float("ALARM_COOLDOWN_SECONDS", 1.5)
DEPLOYMENT_MODE = _env_bool("DEPLOYMENT_MODE", False)
ENABLE_CAMERA = not DEPLOYMENT_MODE
ENABLE_RECORDING = not DEPLOYMENT_MODE

ALLOWED_CLASSES = {
    "person",
    "cell phone",
    "laptop",
    "book",
    "tablet",
    "bottle",
}
THREAT_CLASS_SCORES = {
    "cell phone": 85,
    "laptop": 70,
    "book": 55,
    "tablet": 65,
}

EMAIL_SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER", "")
EMAIL_PORT = _env_int("EMAIL_PORT", 587)
EMAIL_USER = os.getenv("EMAIL_USER", "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
ALERT_RECIPIENT = os.getenv("ALERT_RECIPIENT", "")

VIDEO_CODEC = "XVID"

# Auto-create folders when the module is imported.
for folder in [ASSETS_DIR, MODELS_DIR, VIDEO_OUTPUT_DIR, REPORT_OUTPUT_DIR, SNAPSHOT_OUTPUT_DIR, SCREENSHOTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)
