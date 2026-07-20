# Standard library
import json
import os

# Third-party
import cv2 as cv
import numpy as np

# Project modules
from src.models.detector import (
    load_model,
    extract_yolo_boxes_and_confs,
)
from src.models.ensemble import ensemble_predictions
from src.models.similarity import compute_similarity_scores

from src.preprocess.clahe import apply_clahe_bgr

def run_inference(
    *, # Forces keyword arguments -> a(b = b, c = c) not a(b, c)
    tta: bool = True,
    ref_img_dir: str,
    test_data_dir: str,
    model_1_weights: str,
    model_2_weights: str,
    output_file: str,
    clahe: bool = False,
    confidence_threshold: float = 0.25,
):

    try:
        model_2 = load_model(model_2_weights)
        model_1 = load_model(model_1_weights)
        
        print(f"Successfully loaded model")
    except Exception as e:
        print(f"Error: Could not load model")
        print(e)
        return

    # Gather videos
    try:
        video_folders = sorted([f for f in os.listdir(test_data_dir) if os.path.isdir(os.path.join(test_data_dir, f))])
    except FileNotFoundError:
        print(f"Error: Test data directory not found at: {test_data_dir}")
        return
    if not video_folders:
        print(f"Error: No video folders found in {test_data_dir}")
        return
    print(f"Found {len(video_folders)} videos to process...")

    # Preload and validate reference images
    ref_images = []
    ref_cache = {}
    for p in os.listdir(ref_img_dir):
        img = cv.imread(os.path.join(ref_img_dir, p))
        if img is None:
            print(f"Warning: cannot read reference image: {p}")
            continue
        ref_images.append((p, img))
    if not ref_images:
        print("Warning: no valid reference images loaded; cosine similarity will be skipped.")

    all_predictions = []

    for video_folder_name in video_folders:
        video_path = os.path.join(test_data_dir, video_folder_name, "drone_video.mp4")
        if not os.path.exists(video_path):
            print(f"Warning: 'drone_video.mp4' not found in {video_folder_name}, skipping.")
            continue

        video_bboxes = []

        try:
            cap = cv.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {video_path}...")
            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if clahe:
                    frame = apply_clahe_bgr(frame)
                    
                if (idx == 0):
                    print(f'Frame Type: {type(frame)}')
                    print(f"Frame Shape: {frame.shape}")
                    
                if frame.dtype != np.uint8: frame = (np.clip(frame, 0, 255)).astype(np.uint8)
                    
                # ================= model_1 =================
                reslist_model_1 = model_1.predict(
                    frame,
                    imgsz=640,
                    conf=confidence_threshold,
                    verbose=False,
                    augment=tta
                )
                
                if not reslist_model_1:
                    idx += 1
                    continue
                res_model_1 = reslist_model_1[0]

                if res_model_1.boxes is None or len(res_model_1.boxes) == 0:
                    idx += 1
                    continue

                H, W = frame.shape[:2]
                xyxy_model_1, confs_model_1 = extract_yolo_boxes_and_confs(res_model_1)

                weighted_scores_model_1, ref_cache = compute_similarity_scores(
                    frame=frame,
                    xyxy=xyxy_model_1,
                    confs=confs_model_1,
                    ref_images=ref_images,
                    ref_cache=ref_cache,
                    W=W,
                    H=H,
                )

                # ================= model-2 =================
                reslist_model_2 = model_2.predict(
                    frame,
                    imgsz=1024,
                    conf=confidence_threshold,
                    verbose=False,
                    augment=tta
                )
                
                if not reslist_model_2:
                    idx += 1
                    continue
                res_model_2 = reslist_model_2[0]

                if res_model_2.boxes is None or len(res_model_2.boxes) == 0:
                    idx += 1
                    continue
                    
                xyxy_model_2, confs_model_2 = extract_yolo_boxes_and_confs(res_model_2)
                
                weighted_scores_model_2, ref_cache = compute_similarity_scores(
                    frame=frame,
                    xyxy=xyxy_model_2,
                    confs=confs_model_2,
                    ref_images=ref_images,
                    ref_cache=ref_cache,
                    W=W,
                    H=H,)

                bbox = ensemble_predictions(
                    xyxy_model_1,
                    weighted_scores_model_1,
                    
                    xyxy_model_2,
                    weighted_scores_model_2,
                    
                    W,
                    H,
                    confidence_threshold,)
                
                if bbox is None:  
                    idx += 1
                    continue
                
                x1_fused, y1_fused, x2_fused, y2_fused = bbox
                bbox_data = {
                    "frame": int(idx),
                    "x1": int(x1_fused),
                    "y1": int(y1_fused),
                    "x2": int(x2_fused),
                    "y2": int(y2_fused),
                }
                
                video_bboxes.append(bbox_data)
                idx += 1
            cap.release()
            
        except Exception as e:
            print(f"Error while processing video {video_path}: {e}")
            continue

        detections_list = []
        if video_bboxes:
            detections_list.append({"bboxes": video_bboxes})
        final_video_obj = {
            "video_id": video_folder_name,
            "detections": detections_list
        }
        all_predictions.append(final_video_obj)

    try:
        print(f"\nSaving all {len(all_predictions)} video predictions to {output_file}...")
        with open(output_file, "w") as f:
            json.dump(all_predictions, f, indent=4)
        print("Inference complete.")
    except Exception as e:
        print(f"Error: Could not write output JSON file: {e}")