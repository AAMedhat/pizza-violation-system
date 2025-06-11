# frame_reader/frame_reader.py

import cv2
import pika
import pickle

VIDEO_PATH = r"samples\Sah w b3dha ghalt (2).mp4"  # Make sure this path is correct


def publish_frame(channel, frame, frame_id):
    try:
        data = pickle.dumps((frame_id, frame))
        channel.basic_publish(
            exchange='frames',
            routing_key='video',
            body=data,
            properties=pika.BasicProperties(delivery_mode=2)  # Persistent message
        )
        print(f"[Frame Reader] Publishing frame {frame_id}")  # ✅ This is where you add the print
    except Exception as e:
        print("Error publishing frame:", e)


def main():
    # Connect to RabbitMQ
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()

    # Create exchange if it doesn't exist
    channel.exchange_declare(exchange='frames', exchange_type='fanout', durable=True)

    # Open video file
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ Error: Could not open video.")
        return

    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("🔚 End of video stream.")
            break

        publish_frame(channel, frame, frame_id)
        frame_id += 1

        # Simulate real-time delay
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    connection.close()


if __name__ == "__main__":
    main()