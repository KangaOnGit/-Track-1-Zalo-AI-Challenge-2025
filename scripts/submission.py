import argparse

from src.utils.config import load_config
from src.inference.predictor import run_inference

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference."
    )

    parser.add_argument(
        "--config",
        default="configs/inference.yaml",
        help="Inference configuration file."
    )
    # action: what to do when the argument appears
    parser.add_argument(
        "--tta",
        action="store_true", # if --tta => tta = True else False
        help="Enable Test-Time Augmentation."
    )

    parser.add_argument(
        "--clahe",
        action="store_true",
        help="Enable CLAHE preprocessing."
    )


    parser.add_argument(
        "--imgsz1",
        type=int,
        default=None,
        help="Image size for model 1."
    )

    parser.add_argument(
        "--imgsz2",
        type=int,
        default=None,
        help="Image size for model 2."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    cfg = load_config(args.config)
    
    print(cfg)
    run_inference(
        model_1_weights=cfg["model"]["yolov8n"],
        model_2_weights=cfg["model"]["yolov11n"],

        test_data_dir=cfg["data"]["test_dir"],
        ref_img_dir=cfg["data"]["ref_img_dir"],

        output_file=cfg["output"]["file"],

        confidence_threshold=cfg["inference"]["confidence_threshold"],

        tta=args.tta,
        clahe=args.clahe,

        imgsz1 = args.imgsz1 if args.imgsz1 is not None else cfg["model"]["imgsz1"],
        imgsz2 = args.imgsz2 if args.imgsz2 is not None else cfg["model"]["imgsz2"],
    )


if __name__ == "__main__":
    main()
    
# python -m scripts.submission --args