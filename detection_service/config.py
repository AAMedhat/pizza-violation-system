MODEL_PATH = "../models/yolo12m-v2.pt"
CLASS_NAMES = {
    0: 'Hand',
    1: 'Person',
    2: 'Pizza',
    3: 'Scooper'
}

PROTEIN_ROI = [380, 250, 510, 680] # [x1, y1, x2, y2] — update this based on your setup