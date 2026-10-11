"""
train.py
--------
Trains a YOLOv8 object-detection model on the shrimp-counting dataset.

Usage (from the project root, with your virtual environment active):

    python train.py

Optional flags:

    python train.py --model yolov8n.pt --epochs 100 --imgsz 640 --batch 16

Once training finishes, the best weights are saved at:
    runs/detect/shrimp_count/weights/best.pt

Copy that file into the `models/` folder (or just point the counting
scripts at it directly) to use it for counting shrimp in images or
from your webcam.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on the shrimp dataset")
    parser.add_argument(
        "--data",
        type=str,
        default="dataset/data.yaml",
        help="Path to the dataset's data.yaml file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Base model to start from. yolov8n.pt is the fastest/smallest "
        "(good for a laptop, incl. CPU-only). Other options: yolov8s.pt, "
        "yolov8m.pt (more accurate, slower).",
    )
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (lower this if you run out of memory)")
    parser.add_argument("--device", type=str, default=None, help="'cpu', '0' (first GPU), etc. Auto-detected if omitted.")
    parser.add_argument("--name", type=str, default="shrimp_count", help="Run name (used for the output folder)")
    return parser.parse_args()


def main():
    args = parse_args()

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Could not find {data_yaml}.\n"
            "Make sure you extracted the Roboflow dataset zip into a folder "
            "called 'dataset' next to this script, so that 'dataset/data.yaml' "
            "exists (see README.md for the exact layout)."
        )

    model = YOLO(args.model)

    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        patience=20,       # stop early if validation stops improving
        plots=True,        # saves training curves, confusion matrix, etc.
    )

    print("\nTraining complete.")
    print(f"Best weights saved at: runs/detect/{args.name}/weights/best.pt")
    print("Copy that file into the 'models' folder to use it for counting, e.g.:")
    print(f"  cp runs/detect/{args.name}/weights/best.pt models/shrimp_best.pt")


if __name__ == "__main__":
    main()
