# utils/helpers.py

import numpy as np
import cv2
import pickle 
import os


# -----------------------------
# 🧠 General Utility Functions
# -----------------------------

def get_center(bbox):
    """
    Returns the center point of a bounding box.
    bbox: list or tuple of (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = map(int, bbox)
    return (x1 + x2) // 2, (y1 + y2) // 2


def is_point_in_roi(point, roi=None):
    """
    Checks if a point (x, y) lies inside a ROI rectangle.
    roi: [x1, y1, x2, y2]
    """
    if roi is None:
        from detection_service.config import PROTEIN_ROI
        roi = PROTEIN_ROI

    px, py = point
    x1, y1, x2, y2 = roi
    return x1 < px < x2 and y1 < py < y2


def calculate_iou(boxA, boxB):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


# -----------------------------
# 🖼️ Image Drawing Utilities
# -----------------------------

def draw_bounding_boxes(image, boxes, labels, track_ids=None, color=(0, 255, 0)):
    """
    Draws bounding boxes with labels and optional tracking IDs.
    """
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        label = labels[i] if labels else ""
        text = f"{label}"
        if track_ids:
            text += f" ID:{track_ids[i]}"

        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Draw label
        cv2.putText(
            image,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )
    return image


def draw_roi(frame, roi, label="ROI", color=(0, 255, 255)):
    """
    Draws a rectangular ROI on the frame with a label.
    """
    x1, y1, x2, y2 = roi
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def draw_violation_alert(frame, count=0, color=(0, 0, 255)):
    """
    Draws a red alert box showing number of violations.
    """
    label = f"Violations: {count}"
    cv2.rectangle(frame, (10, 10), (250, 40), color, -1)
    cv2.putText(frame, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame


# -----------------------------
# 📦 Serialization Helpers
# -----------------------------

def serialize_frame(frame_id, frame, violation_count=0):
    """
    Serializes frame data for sending via RabbitMQ/Kafka.
    """
    return pickle.dumps((frame_id, frame, violation_count))


def deserialize_frame(data):
    """
    Deserializes frame data received from message broker.
    """
    return pickle.loads(data)


# -----------------------------
# 🧪 File & Path Utilities
# -----------------------------

def ensure_dir_exists(path):
    """
    Ensures a directory exists; creates it if not.
    """
    os.makedirs(path, exist_ok=True)


def save_frame(frame, output_dir="results/violations"):
    """
    Saves a single frame to disk under timestamped filename.
    """
    import time
    ensure_dir_exists(output_dir)
    timestamp = int(time.time())
    path = os.path.join(output_dir, f"violation_{timestamp}.jpg")
    cv2.imwrite(path, frame)
    print(f"[INFO] Violation frame saved at {path}")