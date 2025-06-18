
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

1. **ROI Zones** (`protein_1`, `protein_2`, etc.) are defined in `config.py` and overlaid on each frame.
2. For each frame:
   - YOLOv12 detects: `Hand`, `Pizza`, `Scooper`, `Person`.
   - A **Virtual ID Tracker** ensures consistent tracking of each hand, even if YOLO's native IDs fluctuate.

3. **ROI Entry Confirmation**:
   - A hand is considered **entered into an ROI** only if it appears **at least twice** within a **30-frame sliding window**.
   - This avoids false entries from momentary detections.

4. **Violation Detection**:
   A violation is flagged if **ALL** the following are true:
   - The hand is confirmed to have entered the ROI ✅
   - The hand touches pizza ❌
   - The hand did **not use a scooper** ❌
   - The hand touched pizza **within 11 seconds (~330 frames)** of ROI entry ❌
   - The last violation from the same hand was **more than 120 frames ago** ✅  
     → Prevents duplicate violations being logged too frequently for the same hand.

5. **Safe Scenarios**:
   - ✅ **Cleaning:** Hand remains in ROI for more than 11 seconds and doesn’t touch pizza.
   - ✅ **Scooper Use:** Hand touches pizza using a scooper (within or after timeout).
   - ✅ **Touch after timeout:** Hand touches pizza after 11 seconds without scooper, considered Cleaning.

6. **Logging & Results**:
   - Annotated video saved to: `results/processed_video.mp4`
   - Violation snapshots saved in: `results/violations/*.jpg`
   - Violation metadata logged in: `results/violations/violations.json` including:
     - Frame ID
     - Virtual Hand ID
     - ROI ID
     - Timestamp


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
