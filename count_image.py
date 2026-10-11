"""
count_image.py
---------------
Runs the trained shrimp model on an image (or a folder of images) and
reports the shrimp count, with an annotated image saved to disk.

Usage:

    python count_image.py --source path/to/image.jpg
    python count_image.py --source path/to/folder/
    python count_image.py --source path/to/image.jpg --model models/shrimp_best.pt --conf 0.25

Annotated results (bounding boxes drawn) are saved under:
    outputs/count_image/
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Count shrimp in image(s)")
    parser.add_argument("--source", type=str, required=True, help="Path to an image file or a folder of images")
    parser.add_argument("--model", type=str, default="models/shrimp_best.pt", help="Path to trained .pt weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (0-1)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--save-dir", type=str, default="outputs/count_image", help="Where to save annotated images")
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Could not find model weights at {model_path}.\n"
            "Train a model first with train.py, or point --model at your .pt file."
        )

    model = YOLO(str(model_path))
    class_names = model.names  # e.g. {0: 'Shrimp-baby', 1: 'shrimp'}

    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        save=True,
        project=str(Path(args.save_dir).parent),
        name=Path(args.save_dir).name,
        exist_ok=True,
    )

    grand_total = 0
    for r in results:
        counts = {}
        for cls_id in r.boxes.cls.tolist():
            name = class_names[int(cls_id)]
            counts[name] = counts.get(name, 0) + 1

        total = sum(counts.values())
        grand_total += total

        image_name = Path(r.path).name
        breakdown = ", ".join(f"{v} {k}" for k, v in counts.items()) if counts else "0 shrimp"
        print(f"{image_name}: {total} shrimp detected ({breakdown})")

    if len(results) > 1:
        print(f"\nTotal across {len(results)} images: {grand_total} shrimp")

    print(f"\nAnnotated image(s) saved in: {args.save_dir}")


if __name__ == "__main__":
    main()
