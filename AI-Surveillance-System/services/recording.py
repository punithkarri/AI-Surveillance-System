import cv2
from pathlib import Path

from config.settings import VIDEO_CODEC, RECORDING_FPS
from utils.helpers import ensure_folder


class VideoRecorder:
    def __init__(self, output_path, frame_size, fps=RECORDING_FPS):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.frame_size = frame_size
        self.fps = fps
        fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
        self.writer = cv2.VideoWriter(str(self.output_path), fourcc, float(self.fps), self.frame_size)

    def write(self, frame):
        if self.writer is None:
            return
        try:
            self.writer.write(frame)
        except Exception:
            pass

    def stop(self):
        if self.writer is not None:
            try:
                self.writer.release()
            except Exception:
                pass
            self.writer = None

    @property
    def active(self):
        return self.writer is not None
