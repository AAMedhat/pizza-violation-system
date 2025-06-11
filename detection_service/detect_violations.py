import cv2
import numpy as np
import pika
import pickle
from ultralytics import YOLO
from detection_service.config import CLASS_NAMES, PROTEIN_ROI
from utils.helpers import get_center, is_point_in_roi, calculate_iou, draw_bounding_boxes, draw_roi, save_frame
model = YOLO("models/yolo12m-v2.pt")

# Tracking states
roi_entry_log = {}  # {hand_track_id: frame_id}
scooper_usage_log = {}  # {hand_track_id: True/False}
interaction_with_pizza = {}  # {hand_track_id: True/False}
violation_count = 0

latest_frame = None
latest_violation_count = 0

def process_frame(frame, frame_id):
    global roi_entry_log, scooper_usage_log, interaction_with_pizza, violation_count

    results = model.track(frame, persist=True)
    if not results:
        return frame, violation_count

    r = results[0]
    boxes = r.boxes.cpu().numpy()
    class_ids = boxes.cls.astype(int)
    bboxes = boxes.xyxy
    track_ids = boxes.id.astype(int) if boxes.id is not None else []

    detections = {}
    for cls_id, bbox, track_id in zip(class_ids, bboxes, track_ids):
        label = CLASS_NAMES.get(cls_id, "Unknown")
        detections[track_id] = {"label": label, "bbox": bbox}

    # Step 1: Hand enters ROI
    for track_id, obj in detections.items():
        label = obj["label"]
        bbox = obj["bbox"]
        center = get_center(bbox)

        if label == "Hand" and is_point_in_roi(center, PROTEIN_ROI):
            if track_id not in roi_entry_log:
                roi_entry_log[track_id] = {
                    'frame_id': frame_id,
                    'undetected_frames': 0  # Initialize undetected frame count
                }
                print(f"[Frame {frame_id}] 🖐️ Hand entered ROI (Track ID: {track_id})")

    # Step 2: Scooper used by hand
    scooper_boxes = [bbox for tid, obj in detections.items() if obj["label"] == "Scooper"]

    for tid, obj in detections.items():
        if obj["label"] == "Hand":
            hand_bbox = obj["bbox"]
            hand_center = get_center(hand_bbox)
            for sbox in scooper_boxes:
                scooper_center = get_center(sbox)
                dist = np.linalg.norm(np.array(hand_center) - np.array(scooper_center))
                if dist < 50:
                    scooper_usage_log[tid] = True
                    print(f"[Frame {frame_id}] 🥄 Scooper used by Hand (Track ID: {tid})")

    # Step 3: Hand interacts with Pizza
    for tid, obj in detections.items():
        if obj["label"] == "Pizza":
            pizza_center = get_center(obj["bbox"])
            for h_tid, h_obj in detections.items():
                if h_obj["label"] == "Hand":
                    hand_center = get_center(h_obj["bbox"])
                    dist = np.linalg.norm(np.array(pizza_center) - np.array(hand_center))
                    if dist < 60:
                        interaction_with_pizza[h_tid] = True
                        print(f"[Frame {frame_id}] 🍕 Hand interacted with Pizza (Track ID: {h_tid})")

    # Step 4: Detect Violations
    for hand_id in list(roi_entry_log.keys()):
        # Check if hand is still in current detections
        if hand_id not in detections:
            # Increment undetected frame count
            roi_entry_log[hand_id]['undetected_frames'] += 1
            
            # Remove hand if it hasn't been detected for 8 frames
            if roi_entry_log[hand_id]['undetected_frames'] >= 8:
                print(f"[Frame {frame_id}] 🖐️ Hand {hand_id} removed from ROI tracking")
                roi_entry_log.pop(hand_id, None)
                interaction_with_pizza.pop(hand_id, None)
                scooper_usage_log.pop(hand_id, None)
            continue
        
        # Reset undetected frame count if hand is detected
        roi_entry_log[hand_id]['undetected_frames'] = 0

        # Check for violations
        if hand_id in interaction_with_pizza and not scooper_usage_log.get(hand_id, False):
            violation_count += 1
            print(f"[🚨 Violation {violation_count}] Detected at frame {frame_id} (Hand Track ID: {hand_id})")
            roi_entry_log.pop(hand_id, None)
            interaction_with_pizza.pop(hand_id, None)

    # Annotate final frame
    annotated_frame = r.plot()
    annotated_frame = draw_roi(annotated_frame, PROTEIN_ROI, "Protein Container")
    cv2.putText(annotated_frame, f"Violations: {violation_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return annotated_frame, violation_count

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
    except KeyboardInterrupt:
        print("[Detection Service] Shutting down gracefully.")
        connection.close()

if __name__ == "__main__":
    main()