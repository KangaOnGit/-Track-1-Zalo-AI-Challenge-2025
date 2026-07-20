import numpy as np
import cv2 as cv
from src.inference.postprocess import safe_clip_box

def l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(vec)
    if n < eps:
        return vec
    return vec / n

def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    # expects 1D float32 vectors
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def image_to_vector(img_bgr: np.ndarray) -> np.ndarray:
    # Convert to float32 and scale to [0,1], flatten
    v = img_bgr.astype(np.float32) / 255.0
    return v.reshape(-1)

def compute_similarity_scores(
    frame,
    xyxy: np.ndarray,
    confs: np.ndarray,
    ref_images,
    ref_cache: dict,
    W: int,
    H: int,
) -> list:
    """Return list of similarity-weighted scores (same length as xyxy)."""
    scores = []
    for i in range(xyxy.shape[0]):
        x1, y1, x2, y2 = xyxy[i]
        clipped = safe_clip_box(x1, y1, x2, y2, W, H)
        if clipped is None:
            scores.append(-1e9)
            continue
        xi1, yi1, xi2, yi2 = clipped

        crop = frame[yi1:yi2, xi1:xi2]
        if crop.size == 0:
            scores.append(-1e9)
            continue

        base_conf = float(confs[i])
        best_sim = -10.0

        if ref_images:
            ch, cw = crop.shape[:2]
            key = (cw, ch)
            if key not in ref_cache:
                vecs = []
                for ref_path, ref_img in ref_images:
                    ref_resized = cv.resize(
                        ref_img, (cw, ch), interpolation=cv.INTER_LINEAR
                    )
                    ref_vec = l2_normalize(image_to_vector(ref_resized))
                    vecs.append((ref_path, ref_vec))
                ref_cache[key] = vecs

            crop_vec = l2_normalize(image_to_vector(crop))
            for ref_path, ref_vec in ref_cache[key]:
                sim = cosine_similarity(crop_vec, ref_vec)
                if float(sim) > float(best_sim):
                    best_sim = sim

        combined_score = best_sim * base_conf
        scores.append(combined_score)

    return scores, ref_cache