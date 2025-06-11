# Pizza Violation Detection System

## Overview
This project implements a **microservices-based Computer Vision system** designed to monitor hygiene protocols in a pizza store. The system detects violations when workers interact with designated **Regions of Interest (ROIs)** without using a scooper. It uses **YOLOv8** for object detection and **RabbitMQ** as the message broker to facilitate communication between services.

## Architecture Overview

The system is built using a microservices architecture, consisting of the following components:

1. **Frame Reader Service**
   - Reads video frames from a video file or RTSP camera feed.
   - Publishes frames to RabbitMQ.

2. **Message Broker**
   - Uses **RabbitMQ** to facilitate communication between services.
   - Handles buffering and stream management.

3. **Detection Service**
   - Subscribes to RabbitMQ to receive frames.
   - Performs object detection using a pre-trained YOLOv8 model.
   - Detects violations based on defined logic:
     - If a hand enters an ROI and interacts with pizza without using a scooper → Counts as a violation.
     - If a scooper is used → No violation.
   - Sends results (bounding boxes, labels, violation status) to the Streaming Service.

4. **Streaming Service**
   - Serves detection results via:
     - REST API for metadata (e.g., number of violations).
     - WebSocket for real-time video streaming with detections drawn on it.
   - Uses **FastAPI** for backend and **HTML/CSS/JS** for frontend.

5. **Frontend UI**
   - Displays:
     - Live video stream with bounding boxes around detected objects.
     - Regions of Interest (ROIs).
     - Violation events (highlighted in red).

## Project Structure

```
pizza-violation-system/
├── detection_service/
│   ├── detect_violations.py
│   └── config.py
├── frame_reader/
│   └── frame_reader.py
├── models/
│   └── yolo12m-v2.pt
├── samples/
│   ├── Sah w b3dha ghalt.mp4
│   ├── Sah w b3dha ghalt (2).mp4
│   └── Sah w b3dha ghalt (3).mp4
├── streaming_service/
│   ├── app.py
│   ├── static/
│   └── templates/
│       └── index.html
├── utils/
│   └── helpers.py
├── requirements.txt
└── README.md
```

## Setup Instructions

### 1. Install Dependencies
Ensure you have Python 3.10+ installed. Install required packages:

```bash
pip install -r requirements.txt
```

### 2. Configure RabbitMQ
- Install and start RabbitMQ:
  ```bash
  sudo apt-get install rabbitmq-server
  sudo systemctl start rabbitmq-server
  ```
- Verify RabbitMQ is running:
  ```bash
  sudo systemctl status rabbitmq-server
  ```

### 3. Run Services
Start each service in separate terminals:

1. **Frame Reader Service**
   ```bash
   python frame_reader/frame_reader.py
   ```

2. **Detection Service**
   ```bash
   python detection_service/detect_violations.py
   ```

3. **Streaming Service**
   ```bash
   python streaming_service/app.py
   ```

### 4. Access Frontend
Open your browser and navigate to:
- **Live Video Stream**: http://localhost:8000

## Features

1. **Real-Time Detection**
   - Detects hands, scoopers, pizzas, and other objects in real time.
   - Tracks interactions with ROIs.

2. **Violation Detection Logic**
   - Flags violations when a hand enters an ROI and interacts with pizza without using a scooper.

3. **Visualization**
   - Bounding boxes around detected objects.
   - Highlighted ROIs.
   - Violation alerts in red.

4. **Scalability**
   - Microservices architecture for modularity and scalability.

## Dependencies

- **Python Libraries**
  - `ultralytics`: For YOLOv8 object detection.
  - `opencv-python`: For video processing.
  - `pika`: For RabbitMQ communication.
  - `fastapi`, `uvicorn`: For the Streaming Service.
  - `jinja2`: For templating in the Streaming Service.

- **Pre-trained Model**
  - `yolo12m-v2.pt`: Pre-trained YOLOv8 model for object detection.

## Sample Videos
The project includes sample videos in the `samples/` directory:
- `Sah w b3dha ghalt.mp4`: Contains 1 violation.
- `Sah w b3dha ghalt (2).mp4`: Contains 2 violations.
- `Sah w b3dha ghalt (3).mp4`: Contains 1 violation.

## Contributing
Feel free to contribute by:
- Improving detection accuracy.
- Adding new features like Docker Compose support.
- Enhancing the frontend UI.

## License
MIT License

## Contact
For questions or feedback, reach out at [your_email@example.com].

## Acknowledgments
- **YOLOv8**: For state-of-the-art object detection.
- **RabbitMQ**: For reliable message queuing.
- **FastAPI**: For building the Streaming Service.
```