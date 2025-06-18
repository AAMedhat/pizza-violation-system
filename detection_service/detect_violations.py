import cv2
import pika
import pickle
import logging
import json
import os
import time
import tempfile
from multiprocessing import Queue
from yolov12.ultralytics import YOLO
from detection_service.config import CLASS_NAMES, ROI_ZONES
from utils.helpers import get_center, draw_rois, save_violation_frame
from utils.virtual_id_tracker import VirtualIDTracker

logging.getLogger("ultralytics").setLevel(logging.WARNING)
model = YOLO("models/best.pt")

output_video_path = "results/processed_video.mp4"
os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = None

violation_count = 0
roi_entry_log = {}
hand_roi_appearances = {}  # virtual_id -> list of frame_ids
last_violation_frame = {}  # (virtual_id, roi_id) -> frame_id
tracker = VirtualIDTracker(distance_threshold=80)

CLEANING_TIMEOUT_FRAMES = 330
ENTRY_CONFIRMATION_FRAMES = 30
VIOLATION_COOLDOWN_FRAMES = 120
PIZZA_TOUCH_DIST = 70
SCOOPER_TOUCH_DIST = 80

violations_queue = Queue()

def bboxes_intersect(b1, b2):
    x1, y1, x2, y2 = b1
    a1, b1_, a2, b2_ = b2
    return max(0, min(x2, a2) - max(x1, a1)) > 0 and max(0, min(y2, b2_) - max(y1, b1_)) > 0

def is_point_in_roi_bbox(hand_box, roi_box):
    return bboxes_intersect(hand_box, roi_box)

def process_frame(frame, frame_id):
    global roi_entry_log, violation_count, video_writer

    results = model.track(frame, persist=True, conf=0.2, verbose=False)
    if not results:
        return frame, violation_count

    r = results[0]
    boxes = r.boxes
    class_ids = boxes.cls.cpu().numpy().astype(int)
    bboxes = boxes.xyxy.cpu().numpy()
    track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else []

    detections = {
        tid: {"label": CLASS_NAMES.get(cls, "Unknown"), "bbox": bbox}
        for cls, bbox, tid in zip(class_ids, bboxes, track_ids)
    }

    virtual_map = tracker.update(detections)

    # Track hand appearances in ROI
    for real_id, obj in detections.items():
        if obj["label"] != "Hand":
            continue
        virtual_id = virtual_map.get(real_id)
        if virtual_id is None:
            continue

        in_roi = None
        for roi_id, roi_box in ROI_ZONES.items():
            if is_point_in_roi_bbox(obj["bbox"], roi_box):
                in_roi = roi_id
                break

        if in_roi:
            # Track frame appearances within sliding window
            hand_roi_appearances.setdefault(virtual_id, []).append(frame_id)
            hand_roi_appearances[virtual_id] = [
                f for f in hand_roi_appearances[virtual_id]
                if f >= frame_id - ENTRY_CONFIRMATION_FRAMES
            ]

            if virtual_id not in roi_entry_log and len(hand_roi_appearances[virtual_id]) >= 1:
                roi_entry_log[virtual_id] = {
                    "roi_id": in_roi,
                    "entry_frame": frame_id,
                    "last_seen": frame_id,
                    "touched_pizza": False,
                    "used_scooper": False,
                    "scooper_id": None
                }
                print(f"[DEBUG] Hand {virtual_id} confirmed in ROI {in_roi} at frame {frame_id}")
            elif virtual_id in roi_entry_log:
                roi_entry_log[virtual_id]["last_seen"] = frame_id

    # Scooper detection
    for real_id, obj in detections.items():
        if obj["label"] != "Scooper":
            continue
        for hand_id, hobj in detections.items():
            if hobj["label"] != "Hand":
                continue
            vid = virtual_map.get(hand_id)
            if vid in roi_entry_log and bboxes_intersect(obj["bbox"], hobj["bbox"]):
                roi_entry_log[vid]["used_scooper"] = True
                roi_entry_log[vid]["scooper_id"] = real_id
                print(f"[DEBUG] Hand {vid} used scooper {real_id} at frame {frame_id}")

    # Pizza interaction
    for real_id, obj in detections.items():
        if obj["label"] != "Pizza":
            continue
        for hand_id, hobj in detections.items():
            if hobj["label"] != "Hand":
                continue
            vid = virtual_map.get(hand_id)
            if vid in roi_entry_log and bboxes_intersect(obj["bbox"], hobj["bbox"]):
                roi_entry_log[vid]["touched_pizza"] = True
                print(f"[DEBUG] Hand {vid} touched pizza at frame {frame_id}")

    # Evaluation logic
    to_delete = []
    for vid, entry in roi_entry_log.items():
        duration = frame_id - entry["entry_frame"]
        roi_id = entry["roi_id"]
        print(f"[TRACE] Hand {vid} | ROI: {roi_id} | Scooper: {entry['used_scooper']} | Pizza: {entry['touched_pizza']} | Duration: {duration}")

        last_frame = last_violation_frame.get(vid, -VIOLATION_COOLDOWN_FRAMES - 1)

        if entry["touched_pizza"] and not entry["used_scooper"] and duration < CLEANING_TIMEOUT_FRAMES:
            if frame_id - last_frame >= VIOLATION_COOLDOWN_FRAMES:
                violation_count += 1
                last_violation_frame[vid] = frame_id
                print(f"[🚨 VIOLATION] Hand {vid} touched pizza too early without scooper in ROI {roi_id}")
                save_violation_frame(frame, "results/violations")
                log_violation_info(frame_id, vid, roi_id, entry["scooper_id"])
            to_delete.append(vid)

        elif not entry["touched_pizza"] and duration >= CLEANING_TIMEOUT_FRAMES:
            print(f"[✅ CLEANING] Hand {vid} cleaned for 11s in ROI {roi_id}")
            to_delete.append(vid)

        elif entry["touched_pizza"] and entry["used_scooper"]:
            print(f"[INFO] Hand {vid} used scooper in ROI {roi_id}")
            to_delete.append(vid)

        elif entry["touched_pizza"] and not entry["used_scooper"] and duration >= CLEANING_TIMEOUT_FRAMES:
            print(f"[INFO] Hand {vid} touched pizza after timeout (no violation) in ROI {roi_id}")
            to_delete.append(vid)

    for vid in to_delete:
        roi_entry_log.pop(vid, None)

    # Draw results
    annotated_frame = frame.copy()
    for real_id, obj in detections.items():
        virtual_id = virtual_map.get(real_id)
        if virtual_id is None:
            continue
        x1, y1, x2, y2 = map(int, obj["bbox"])
        label = f"ID:{virtual_id} {obj['label']}"
        color = (255, 255, 0) if obj["label"] == "Hand" else (0, 255, 0)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    annotated_frame = draw_rois(annotated_frame, ROI_ZONES)
    cv2.putText(annotated_frame, f"Violations: {violation_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if video_writer is None:
        h, w, _ = annotated_frame.shape
        video_writer = cv2.VideoWriter(output_video_path, fourcc, 30, (w, h))
    video_writer.write(annotated_frame)

    return annotated_frame, violation_count

def log_violation_info(frame_id, hand_id, roi_id, scooper_id=None):
    log_file = "results/violations/violations.json"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    entry = {
        "frame_id": frame_id,
        "hand_id": hand_id,
        "roi_id": roi_id,
        "scooper_id": scooper_id,
        "timestamp": int(time.time())
    }
    try:
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                data = json.load(f)
        else:
            data = []
        data.append(entry)
        with tempfile.NamedTemporaryFile("w", dir=os.path.dirname(log_file), delete=False) as tf:
            json.dump(data, tf, indent=2)
            temp_name = tf.name
        os.replace(temp_name, log_file)
        print(f"[INFO] Violation logged for frame {frame_id} in ROI {roi_id}")
    except Exception as e:
        print("[ERROR] Failed to log violation:", str(e))
    violations_queue.put(entry)

def callback(ch, method, properties, body):
    try:
        frame_id, frame = pickle.loads(body)
        result_frame, v_count = process_frame(frame, frame_id)
        ch.basic_publish(
            exchange='results',
            routing_key='detections',
            body=pickle.dumps((frame_id, result_frame, v_count)),
            properties=pika.BasicProperties(delivery_mode=2)
        )
    except Exception as e:
        print("[ERROR] Failed to process frame:", str(e))

def main():
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()
    channel.exchange_declare(exchange='results', exchange_type='fanout', durable=True)
    channel.queue_declare(queue='detection')
    channel.basic_consume(queue='detection', on_message_callback=callback, auto_ack=True)
    channel.queue_bind(exchange='frames', queue='detection')
    print("[Detection Service] Waiting for frames...")
    try:
        channel.start_consuming()
    finally:
        if video_writer is not None:
            video_writer.release()
        connection.close()

if __name__ == "__main__":
    main()
