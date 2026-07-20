from src.dataset.visualizer import visualize_annotation
from src.utils.config import load_config
import os

if __name__ == "__main__":
    cfg = load_config("configs/dataset.yaml")

    YOLO_DATASET_DIR = cfg["dataset"]["output_dir"]
    SPLIT_TO_CHECK = "val"

    images_dir = os.path.join(YOLO_DATASET_DIR, "images", SPLIT_TO_CHECK)
    labels_dir = os.path.join(YOLO_DATASET_DIR, "labels", SPLIT_TO_CHECK)

    random_image_name = "Backpack_0_frame_003483.jpg"

    image_path = os.path.join(images_dir, random_image_name)
    label_path = os.path.join(
        labels_dir,
        os.path.splitext(random_image_name)[0] + ".txt",
    )

    visualize_annotation(image_path, label_path)