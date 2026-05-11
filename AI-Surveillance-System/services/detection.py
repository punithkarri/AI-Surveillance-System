import cv2
from pathlib import Path
from ultralytics import YOLO

from config.settings import (
    ALLOWED_CLASSES,
    CONFIDENCE_THRESHOLD,
    DEFAULT_MODEL_PATH,
    FALLBACK_MODEL_NAME,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    THREAT_CLASS_SCORES,
)


def load_model(model_path: Path = None):
    model_path = model_path or DEFAULT_MODEL_PATH
    try:
        return YOLO(str(model_path))
    except FileNotFoundError:
        return YOLO(FALLBACK_MODEL_NAME)
    except Exception:
        return YOLO(FALLBACK_MODEL_NAME)


def _normalize_label(label: str, bbox: tuple) -> str:
    if label == "laptop":
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        if area < 5000:
            return "cell phone"
    return label


def calculate_threat_score(detections):
    threat = 0
    for det in detections:
        label = det.get("label")
        if label in THREAT_CLASS_SCORES:
            threat += THREAT_CLASS_SCORES[label]
        elif label == "person":
            threat += 20
        else:
            threat += 10
    return min(threat, 100)


def draw_boxes(frame, detections):
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        label = detection["label"]
        conf = detection["confidence"]

        if label in THREAT_CLASS_SCORES:
            color = (0, 0, 255)
        elif label == "person":
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label_text = f"{label} {conf:.2f}"
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x1, y1 - 26), (x1 + text_size[0] + 10, y1), color, -1)
        cv2.putText(frame, label_text, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


def process_frame(frame, model):
    if frame is None or model is None:
        return frame, [], 0

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    try:
        results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
    except Exception:
        return frame, [], 0

    detections = []
    for result in results:
        if not hasattr(result, "boxes"):
            continue
        for box in result.boxes:
            try:
                xyxy = box.xyxy
                if xyxy is None or len(xyxy) == 0:
                    continue
                bbox = tuple(map(int, xyxy[0].tolist()))
                x1, y1, x2, y2 = bbox
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if conf < CONFIDENCE_THRESHOLD:
                    continue
                label = model.names[cls_id] if hasattr(model, "names") else str(cls_id)
                label = _normalize_label(label, bbox)
                if label not in ALLOWED_CLASSES:
                    continue
                detections.append({
                    "label": label,
                    "confidence": conf,
                    "bbox": bbox,
                })
            except Exception:
                continue

    threat_score = calculate_threat_score(detections)
    annotated_frame = draw_boxes(frame.copy(), detections)
    return annotated_frame, detections, threat_score
