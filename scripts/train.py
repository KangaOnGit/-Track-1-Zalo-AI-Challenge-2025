import argparse

from src.models.detector import load_model
from src.utils.config import load_config


def parse_args():
    
    # Create parser
    parser = argparse.ArgumentParser(
        description="Train a YOLO model."
    )

    # Config
    parser.add_argument(
        "--config",
        default="configs/train.yaml",
        help="Path to training config."
    )

    parser.add_argument(
        "--experiment",
        default="yolo11n",
        help="Experiment name inside train.yaml."
    )

    # Training overrides
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--patience", type=int)

    # Optimizer overrides
    parser.add_argument("--lr0", type=float)
    parser.add_argument("--optimizer", type=str)
    parser.add_argument("--cos-lr", action="store_true")

    # Misc
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--project", default=None)
    parser.add_argument("--name", default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    cfg = load_config(args.config)

    if args.experiment not in cfg["experiments"]:
        raise ValueError(f"Unknown experiment '{args.experiment}'")

    exp = cfg["experiments"][args.experiment]

    model = load_model(exp["model"])

    train_kwargs = {
        "data": exp["data"],

        "epochs": args.epochs or exp["training"]["epochs"],
        "batch": args.batch or exp["training"]["batch"],
        "imgsz": args.imgsz or exp["training"]["imgsz"],
        "patience": args.patience or exp["training"]["patience"],

        "optimizer": args.optimizer or exp["optimizer"]["name"],
        "lr0": args.lr0 or exp["optimizer"]["lr0"],
        "cos_lr": args.cos_lr or exp["optimizer"]["cos_lr"],

        "single_cls": exp["options"]["single_cls"],

        "name": args.name or exp["output"]["name"],

        **exp["augmentation"],
    }

    if args.device is not None:
        train_kwargs["device"] = args.device

    if args.workers is not None:
        train_kwargs["workers"] = args.workers

    if args.project is not None:
        train_kwargs["project"] = args.project

    print("Training configuration")
    for k, v in train_kwargs.items():
        print(f"{k:15}: {v}")
        
    print(args)
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
    
# python -m scripts.train --args