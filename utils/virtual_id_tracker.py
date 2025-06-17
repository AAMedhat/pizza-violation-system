import numpy as np
from collections import deque

class VirtualIDTracker:
    def __init__(self, distance_threshold=50, max_history=5):
        self.next_id = 1
        self.object_map = {}  # real_id -> virtual_id
        self.positions = {}   # virtual_id -> deque of centers
        self.distance_threshold = distance_threshold
        self.max_history = max_history

    def get_center(self, box):
        x1, y1, x2, y2 = box
        return (x1 + x2) / 2, (y1 + y2) / 2

    def update(self, detections):
        updated_map = {}
        new_positions = {}

        for real_id, det in detections.items():
            current_center = self.get_center(det["bbox"])

            best_vid = None
            min_dist = float("inf")
            for vid, centers in self.positions.items():
                if centers:
                    last_center = centers[-1]
                    dist = np.linalg.norm(np.array(current_center) - np.array(last_center))
                    if dist < self.distance_threshold and dist < min_dist:
                        best_vid = vid
                        min_dist = dist

            if best_vid is not None:
                updated_map[real_id] = best_vid
                new_positions.setdefault(best_vid, deque(maxlen=self.max_history)).append(current_center)
            else:
                updated_map[real_id] = self.next_id
                new_positions[self.next_id] = deque([current_center], maxlen=self.max_history)
                self.next_id += 1

        self.object_map = updated_map
        self.positions = new_positions
        return updated_map

    def get_velocity(self, virtual_id):
        if virtual_id not in self.positions or len(self.positions[virtual_id]) < 2:
            return 0.0
        positions = list(self.positions[virtual_id])
        displacement = np.array(positions[-1]) - np.array(positions[0])
        return np.linalg.norm(displacement)

    def get_path(self, virtual_id):
        return list(self.positions.get(virtual_id, []))
