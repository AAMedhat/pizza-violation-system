
# 🍕 Pizza Store Hygiene Violation Detection System

A real-time computer vision system designed to monitor hygiene compliance in a pizza store using microservices. It detects if an employee interacts with protein ingredients **without a scooper** and flags these interactions as violations.

---

## 🔧 Architecture Overview

This project is built as a **microservices-based architecture** comprising:

| Component         | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| 🖼️ Frame Reader   | Reads video stream and publishes frames to RabbitMQ                        |
| 🧠 Detection Service | Tracks hands, scoopers, pizza interactions using YOLOv12 + logic            |
| 🌐 Streaming API  | Displays real-time annotated video and serves REST/WebSocket endpoints     |
| 💻 Frontend UI    | Simple dashboard displaying video feed + violations in real time           |
| 📦 Message Broker | RabbitMQ manages communication between services                            |

---

## 📁 Folder Structure

```
PIZZA-VIOLATION-SYSTEM/
│
├── detection_service/        # Core logic: detection, violation logic, ROI handling
│   ├── config.py
│   └── detect_violations.py
│   
│── frame_reader/             # Video frame publisher
│   └── frame_reader.py
│
├── models/                   # Trained YOLOv12 model weights
│   └── best.pt
│
├── results/                  # Output results
│   ├── processed_video.mp4
│   └── violations/
│       ├── violation_*.jpg
│       └── violations.json
│
├── samples/                  # Test videos
│
├── streaming_service/        # FastAPI server
│   ├── app.py
│   └── templates/
│       └── index.html
│
├── utils/                    # Reusable utilities
│   ├── helpers.py
│   └── virtual_id_tracker.py
│
├── yolov12/                  # YOLOv12 source (cloned from GitHub)
├── requirements.txt
├── pizza-final.ipynb         # Notebook used to train YOLOv12
└── README.md
```

---

## 🎥 Detection Logic

1. **Define ROIs** (`protein_1`, `protein_2`, etc.) for scooper-required zones.
2. For each frame:
   - Detect `Hand`, `Pizza`, and `Scooper` objects via YOLOv12.
   - Track hands using `VirtualIDTracker` for consistent identification.
   - If a **hand touches pizza without using a scooper** AND:
     - Entered ROI but didn’t stay 11 seconds = 🚨 **Violation**
     - OR used hand directly without scooper = 🚨 **Violation**
   - Else if:
     - Hand stayed >11s without touching = ✅ **Cleaning**
     - Or touched pizza using scooper = ✅ **Safe**
3. Frame and violation details saved to `results/violations/`.

---

## 💻 Frontend UI Features

- Displays live video feed (updated per frame)
- Shows `VirtualID` per hand
- Highlights ROI zones
- Alerts violation with frame snapshot + log
- Displays total violation count

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure RabbitMQ

Install RabbitMQ:
```bash
sudo apt update
sudo apt install rabbitmq-server
sudo systemctl start rabbitmq-server
sudo systemctl enable rabbitmq-server
sudo systemctl status rabbitmq-server
```

Or use Docker:
```bash
docker run -d --hostname rabbit --name rabbitmq \
  -p 5672:5672 -p 15672:15672 rabbitmq:3-management
```

RabbitMQ UI: [http://localhost:15672](http://localhost:15672) (user: `guest`, pass: `guest`)

### 3. Run Services (in 3 terminals)

**Terminal 1: Frame Publisher**
```bash
python detection_service/frame_reader.py
```

**Terminal 2: Detection & Violation Logic**
```bash
python detection_service/detect_violations.py
```

**Terminal 3: Web Streaming Service**
```bash
python streaming_service/app.py
```

Then open: [http://localhost:8000](http://localhost:8000)

---

## 🧠 Virtual ID Tracking

YOLO’s `track_id` values can be unstable between frames. The `VirtualIDTracker` solves this by:
- Assigning stable virtual IDs based on object proximity
- Tracking object positions across frames
- Ensuring consistency for violation timing, scooper usage, and cleaning

This makes violation detection far more robust.

---

## 🧪 Model Training (YOLOv12)

This project uses a custom-trained **YOLOv12** model to detect:

- `Hand`
- `Scooper`
- `Pizza`
- `Person` 

### 📊 Dataset

- Annotated via Roboflow:  
  👉 [PizzaStore Dataset](https://app.roboflow.com/abdelrhman-medhat/pizzastore-qftv2/5)

### 🧠 Training Notebook

- Training performed using: `pizza-final.ipynb`
- Model base: YOLOv12 `yolov12m.pt`
- Training repo: [https://github.com/sunsmarterjie/yolov12](https://github.com/sunsmarterjie/yolov12)

### 🏋️‍♂️ How to Retrain

1. Clone YOLOv12:
```bash
git clone https://github.com/sunsmarterjie/yolov12
cd yolov12
```

2. Export dataset from Roboflow as YOLOv5 format.

3. Train the model:
```bash
python train.py --data dataset.yaml --weights yolov12m.pt --cfg yolov12m.yaml --epochs 100 --img 640
```

4. Copy weights to project:
```bash
mv runs/train/exp/weights/best.pt ../models/best.pt
```

---

## 📦 requirements.txt

```
ultralytics
opencv-python
pika
fastapi
uvicorn
numpy
```

---

## 📽️ Output Logs

- Annotated video: `results/processed_video.mp4`
- Violations images: `results/violations/*.jpg`
- Log file: `results/violations/violations.json`

---

## 📬 Contact & Submission

Developed by **Abdelrhman Medhat**  
Built for the **Eagle Vision** task submission: real-time microservices video analysis
