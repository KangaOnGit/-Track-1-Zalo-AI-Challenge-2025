import matplotlib.pyplot as plt
import cv2
import os
from .converter import *

def visualize_annotation(image_path, label_path):
    """
    Loads an image and its label, draws the bounding boxes, and displays it
    """
    
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return
        
    img_h, img_w, _ = image.shape
    
    if not os.path.exists(label_path):
        print(f"Warning: No label file found at {label_path}")
    else:
        # Read the label file
        with open(label_path, 'r') as f:
            for line in f:
                try:
                    # Parse the YOLO coordinates
                    parts = line.strip().split()
                    yolo_coords = [float(p) for p in parts[1:]]
                    
                    # Denormalize coordinates
                    x1, y1, x2, y2 = denormalize_yolo(yolo_coords, img_w, img_h)
                    
                    # --- Draw on the image ---
                    # Draw the bounding box (in green, BGR format)
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                                    
                except Exception as e:
                    print(f"Error processing line '{line}': {e}")

    # --- Display the image in the notebook ---
    print(f"Displaying: {os.path.basename(image_path)}")
    
    # 1. Convert from BGR (OpenCV) to RGB (Matplotlib)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 2. Set figure size to be reasonable
    plt.figure(figsize=(10, 10))
    
    # 3. Display the image
    plt.imshow(image_rgb)
    
    # 4. Hide the axes
    plt.axis('off')
    
    # 5. Show the plot
    plt.show()