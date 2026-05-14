import cv2
import math
from pathlib import Path
from ultralytics import YOLO

from config.settings import (
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


def _bbox_center(bbox: tuple) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _normalize_label(label: str, bbox: tuple) -> str:
    if label == "laptop":
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        if area < 5000:
            return "cell phone"
    return label


def _base_threat(label: str, confidence: float) -> int:
    if label == "person":
        base = THREAT_CLASS_SCORES.get("person", 20)
    else:
        base = THREAT_CLASS_SCORES.get(label, 10)
    return min(int(base * confidence), 100)


def calculate_threat_score(detections):
    threat = 0
    persons = [det for det in detections if det["label"] == "person"]
    objects = [det for det in detections if det["label"] != "person"]

    for det in detections:
        threat += _base_threat(det["label"], det["confidence"])

    # Suspicious multi-person gathering
    if len(persons) >= 2:
        crowd_bonus = min(len(persons) * 5, 20)
        threat += crowd_bonus
        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                dist = _distance(
                    _bbox_center(persons[i]["bbox"]),
                    _bbox_center(persons[j]["bbox"]),
                )
                if dist < FRAME_WIDTH * 0.25:
                    threat += 18
                elif dist < FRAME_WIDTH * 0.45:
                    threat += 8

    # Person + suspicious object proximity
    for person in persons:
        pc = _bbox_center(person["bbox"])
        for obj in objects:
            oc = _bbox_center(obj["bbox"])
            dist = _distance(pc, oc)
            if dist < FRAME_WIDTH * 0.30:
                if obj["label"] == "cell phone":
                    threat += 30
                elif obj["label"] == "laptop":
                    threat += 22
                elif obj["label"] in {"backpack", "handbag", "suitcase"}:
                    threat += 18
                else:
                    threat += 6

    # Person + object combination boosts
    if persons and any(obj["label"] == "cell phone" for obj in objects):
        threat += 26
    if persons and any(obj["label"] == "laptop" for obj in objects):
        threat += 18
    if persons and any(obj["label"] in {"backpack", "handbag"} for obj in objects):
        threat += 12

    # Crowd detection dynamic boost
    if len(persons) >= 4:
        threat += 18
    if len(persons) >= 6:
        threat += 12

    return min(int(threat), 100)


def draw_boxes(frame, detections):
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        label = detection["label"]
        conf = detection["confidence"]
        threat = detection.get("threat", 0)

        if threat >= 70:
            color = (0, 0, 255)
        elif threat >= 35:
            color = (0, 215, 255)
        else:
            color = (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        label_text = f"{label} {conf:.2f} | T {threat}%"
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x1, y1 - 30), (x1 + text_size[0] + 12, y1), color, -1)
        cv2.putText(frame, label_text, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


def process_frame(frame, model):
    if frame is None or model is None:
        return frame, [], 0

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    try:
        results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD, imgsz=640, half=False)
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
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if conf < CONFIDENCE_THRESHOLD:
                    continue
                label = model.names[cls_id] if hasattr(model, "names") else str(cls_id)
                label = _normalize_label(label, bbox)
                detections.append({
                    "label": label,
                    "confidence": conf,
                    "bbox": bbox,
                    "threat": _base_threat(label, conf),
                })
            except Exception:
                continue

    threat_score = calculate_threat_score(detections)
    annotated_frame = draw_boxes(frame.copy(), detections)
    return annotated_frame, detections, threat_score
