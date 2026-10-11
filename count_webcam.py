"""
count_webcam.py
-----------------
Runs the trained shrimp model live on your laptop's webcam (or any
connected camera / video file) and displays a running shrimp count
on screen in real time.

Usage:

    python count_webcam.py
    python count_webcam.py --model models/shrimp_best.pt --conf 0.3
    python count_webcam.py --camera 1          # use a different camera index
    python count_webcam.py --source path/to/video.mp4   # run on a video file instead

Controls while the window is open:
    q  -> quit
    s  -> save a snapshot of the current annotated frame to outputs/webcam_snapshots/
"""

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Live shrimp counting from a camera")
    parser.add_argument("--model", type=str, default="models/shrimp_best.pt", help="Path to trained .pt weights")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (0 is usually the built-in webcam)")
    parser.add_argument("--source", type=str, default=None, help="Optional: path to a video file instead of a live camera")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (0-1)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
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
    class_names = model.names

    cam_source = args.source if args.source else args.camera
    cap = cv2.VideoCapture(cam_source)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera/video source '{cam_source}'. "
            "If you have more than one camera, try --camera 1, --camera 2, etc."
        )

    snapshot_dir = Path("outputs/webcam_snapshots")
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    print("Press 'q' to quit, 's' to save a snapshot.")

    prev_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("No more frames / camera disconnected.")
            break

        results = model.predict(source=frame, conf=args.conf, imgsz=args.imgsz, verbose=False)
        r = results[0]

        counts = {}
        for cls_id in r.boxes.cls.tolist():
            name = class_names[int(cls_id)]
            counts[name] = counts.get(name, 0) + 1
        total = sum(counts.values())

        annotated = r.plot()  # draws boxes + labels on a copy of the frame

        # FPS (for info only)
        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        breakdown = " | ".join(f"{k}: {v}" for k, v in counts.items()) if counts else "shrimp: 0"
        header = f"Total shrimp: {total}  ({breakdown})  FPS: {fps:.1f}"
        cv2.putText(annotated, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Shrimp Counter - press q to quit, s to save", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            fname = snapshot_dir / f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(str(fname), annotated)
            print(f"Saved snapshot: {fname}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
