import cv2 as cv
import numpy as np

def apply_clahe_bgr(frame_bgr: np.ndarray) -> np.ndarray:
    hsv = cv.cvtColor(frame_bgr, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v2 = clahe.apply(v)
    hsv2 = cv.merge([h, s, v2])
    return cv.cvtColor(hsv2, cv.COLOR_HSV2BGR)