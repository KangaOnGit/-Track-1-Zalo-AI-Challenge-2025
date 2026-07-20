def convert_to_yolo(box, img_w, img_h):
    """
    Converts (x1, y1, x2, y2) absolute pixel coordinates to
    YOLO's (x_center_norm, y_center_norm, width_norm, height_norm) format.
    """
    try:
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        img_w, img_h = float(img_w), float(img_h)

        dw = 1.0 / img_w
        dh = 1.0 / img_h
        
        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1
        
        x_center_norm = x_center * dw
        y_center_norm = y_center * dh
        width_norm = width * dw
        height_norm = height * dh
        
        return (x_center_norm, y_center_norm, width_norm, height_norm)
    except Exception as e:
        print(f"Error in coordinate conversion: {e}")
        return None
    
def denormalize_yolo(yolo_coords, img_w, img_h):
    """
    Converts YOLO's (x_center_norm, y_center_norm, width_norm, height_norm)
    back to (x1, y1, x2, y2) absolute pixel coordinates.
    """
    x_c_norm, y_c_norm, w_norm, h_norm = yolo_coords
    
    # Calculate pixel values from normalized values
    box_w = w_norm * img_w
    box_h = h_norm * img_h
    x_c = x_c_norm * img_w
    y_c = y_c_norm * img_h
    
    # Calculate top-left corner (x1, y1)
    x1 = int(x_c - (box_w / 2))
    y1 = int(y_c - (box_h / 2))
    
    # Calculate bottom-right corner (x2, y2)
    x2 = int(x_c + (box_w / 2))
    y2 = int(y_c + (box_h / 2))
    
    return (x1, y1, x2, y2)