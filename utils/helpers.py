# helpers.py
import os
import time
import cv2
import numpy as np

def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) // 2), int((y1 + y2) // 2)

def is_point_in_roi(point, roi):
    px, py = point
    x1, y1, x2, y2 = roi
    return x1 < px < x2 and y1 < py < y2

def draw_roi(frame, roi, label="", color=(0, 255, 255)):
    x1, y1, x2, y2 = roi
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

def draw_rois(frame, roi_dict):
    for roi_id, roi in roi_dict.items():
        frame = draw_roi(frame, roi, label=roi_id)  # No "ROI" prefix
    return frame

def save_violation_frame(frame, output_dir="results/violations"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = int(time.time())
    path = os.path.join(output_dir, f"violation_{timestamp}.jpg")
    cv2.imwrite(path, frame)
    print(f"[INFO] Violation frame saved at {path}")
