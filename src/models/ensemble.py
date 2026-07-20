import numpy as np
from ensemble_boxes import weighted_boxes_fusion

def ensemble_predictions(
    xyxy_1,
    weighted_scores_1,
    
    xyxy_2,
    weighted_scores_2,
    
    W,
    H,
    confidence_threshold,
):
    # WBF expects normalized [x1, y1, x2, y2] in [0,1] for each model
    boxes_1 = xyxy_1.astype(float).copy()
    boxes_1[:, 0] /= W  # x1
    boxes_1[:, 2] /= W  # x2
    boxes_1[:, 1] /= H  # y1
    boxes_1[:, 3] /= H  # y2

    boxes_2 = xyxy_2.astype(float).copy()
    boxes_2[:, 0] /= W
    boxes_2[:, 2] /= W
    boxes_2[:, 1] /= H
    boxes_2[:, 3] /= H

    scores_1 = np.array(weighted_scores_1, dtype=float)
    scores_2 = np.array(weighted_scores_2, dtype=float)

    # Single-class: all labels = 0
    labels_1 = np.zeros_like(scores_1, dtype=int)
    labels_2 = np.zeros_like(scores_2, dtype=int)

    boxes_list = [boxes_1.tolist(), boxes_2.tolist()]
    scores_list = [scores_1.tolist(), scores_2.tolist()]
    labels_list = [labels_1.tolist(), labels_2.tolist()]

    model_weights = [1.0, 1.0]

    fused_boxes_norm, fused_scores, fused_labels = weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        weights=model_weights,
        iou_thr=0.55,
        skip_box_thr=float(CONFIDENCE_THRESHOLD),
    )

    if len(fused_boxes_norm) == 0:
        return None
    
    # De-normalize fused boxes back to pixel coords
    fused_boxes = np.array(fused_boxes_norm, dtype=float)
    fused_boxes[:, 0] *= W
    fused_boxes[:, 2] *= W
    fused_boxes[:, 1] *= H
    fused_boxes[:, 3] *= H
    fused_boxes = fused_boxes.astype(int)

    fused_scores = np.array(fused_scores, dtype=float)

    # Take best fused box for this frame
    best_idx = int(np.argmax(fused_scores))
    x1_fused, y1_fused, x2_fused, y2_fused = fused_boxes[best_idx]
    
    return x1_fused, y1_fused, x2_fused, y2_fused