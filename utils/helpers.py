from pathlib import Path
import time
import re


def ensure_folder(path):
    """Create a folder if it does not exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_timestamp_string():
    """Return a timestamp string for filenames."""
    return time.strftime("%Y%m%d_%H%M%S")


def normalized_filename(value: str) -> str:
    """Turn an arbitrary string into a safe filename."""
    name = re.sub(r"[^a-zA-Z0-9_.-]", "_", value)
    return name.strip("_-") or "output"


def read_text_file(path):
    path = Path(path)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")
