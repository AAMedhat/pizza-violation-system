# # # test_rabbitmq.py
# # import pika

# # try:
# #     connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
# #     channel = connection.channel()
# #     print("✅ Successfully connected to RabbitMQ")
# #     connection.close()
# # except Exception as e:
# #     print("❌ Failed to connect to RabbitMQ:", str(e))

# # test_video.py
# import cv2

# cap = cv2.VideoCapture("samples/Sah w b3dha ghalt.mp4")
# if not cap.isOpened():
#     print("❌ Failed to open video file")
# else:
#     ret, frame = cap.read()
#     if ret:
#         print("✅ Successfully read first frame")
#     else:
#         print("❌ Couldn't read any frame from video")