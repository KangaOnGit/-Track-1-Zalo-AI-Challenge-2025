from ultralytics import YOLO
import numpy as np

def load_model(model_path):
    return YOLO(model_path)

def extract_yolo_boxes_and_confs(results) -> tuple[np.ndarray, np.ndarray]:
    """Extract xyxy and confs from an Ultralytics Results object."""
    xyxy = results.boxes.xyxy.detach().cpu().numpy()
    confs = results.boxes.conf.detach().cpu().numpy()
    return xyxy, confs