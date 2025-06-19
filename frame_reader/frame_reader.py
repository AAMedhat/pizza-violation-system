# frame_reader.py
import cv2
import pika
import pickle
from utils.helpers import draw_rois
from detection_service.config import ROI_ZONES
import os

VIDEO_PATH = os.environ.get("VIDEO_PATH", "samples/input.mp4")

def publish_frame(channel, frame, frame_id):
    try:
        data = pickle.dumps((frame_id, frame))
        channel.basic_publish(
            exchange='frames',
            routing_key='video',
            body=data,
            properties=pika.BasicProperties(delivery_mode=2)
        )
        print(f"[Frame Reader] Frame {frame_id} published.")
    except Exception as e:
        print("Error publishing:", str(e))

def main():
    connection = pika.BlockingConnection(pika.ConnectionParameters('rabbitmq'))
    channel = connection.channel()
    channel.exchange_declare(exchange='frames', exchange_type='fanout', durable=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ Error: Could not open video file")
        return

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("🖚 End of video stream.")
            break

        frame_with_roi = draw_rois(frame.copy(), ROI_ZONES)
        publish_frame(channel, frame_with_roi, frame_id)
        frame_id += 1

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    connection.close()

if __name__ == "__main__":
    main()
