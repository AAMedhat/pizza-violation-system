from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import pika
import pickle
import threading
import cv2
import numpy as np

app = FastAPI()
app.mount("/static", StaticFiles(directory="streaming_service/static"), name="static")
templates = Jinja2Templates(directory="streaming_service/templates")  # Correct path

latest_frame = None
latest_violation_count = 0

def consume_frames():
    def callback(ch, method, properties, body):
        global latest_frame, latest_violation_count
        frame_id, frame, count = pickle.loads(body)
        latest_frame = frame
        latest_violation_count = count

    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()
    channel.queue_declare(queue='streaming')
    channel.basic_consume(queue='streaming', on_message_callback=callback, auto_ack=True)
    channel.queue_bind(exchange='results', queue='streaming')
    channel.start_consuming()

@app.get("/")
async def index():
    return templates.TemplateResponse("index.html", {"request": {}, "violations": latest_violation_count})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        if latest_frame is not None:
            _, buffer = cv2.imencode('.jpg', latest_frame)
            await websocket.send_bytes(buffer.tobytes())

def generate():
    while True:
        if latest_frame is not None:
            _, buffer = cv2.imencode('.jpg', latest_frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    threading.Thread(target=consume_frames).start()
    uvicorn.run(app, host="0.0.0.0", port=8000)