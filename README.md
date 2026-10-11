# Shrimp Counting Project (YOLOv8)

Counts shrimp in photos and in a live webcam feed using a YOLOv8 object
detection model trained on your Roboflow dataset
(`shrimp_counting_v7i_yolov8.zip` — 5,689 images, 2 classes:
`Shrimp-baby` and `shrimp`).

## 1. Project layout

Put everything in one folder like this:

```
shrimp_project/
├── dataset/              <- your extracted dataset goes here
│   ├── data.yaml
│   ├── train/images, train/labels
│   ├── valid/images, valid/labels
│   └── test/images,  test/labels
├── models/                <- trained weights go here after training
├── outputs/                <- results (annotated images/snapshots) are saved here
├── requirements.txt
├── train.py
├── count_image.py
└── count_webcam.py
```

You already have the dataset zip. Just extract it into a folder named
`dataset` next to these scripts:

- **Windows:** right-click the zip → "Extract All" → extract into a
  new folder called `dataset` inside `shrimp_project`.
- **Mac/Linux (terminal):**
  ```bash
  cd shrimp_project
  unzip shrimp_counting_v7i_yolov8.zip -d dataset
  ```

Check that `shrimp_project/dataset/data.yaml` exists — that's the file
the training script needs.

## 2. Set up your environment (one-time)

You need Python 3.9–3.11 installed. Then, from inside the
`shrimp_project` folder:

```bash
# create an isolated environment (recommended)
python -m venv venv

# activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

This installs `ultralytics` (YOLOv8), `opencv-python` (for the webcam
feed), and `numpy`. First run will also auto-download the small base
YOLOv8 weights (`yolov8n.pt`, ~6 MB) the first time you train.

If you have an NVIDIA GPU and want it used automatically, also install
the CUDA build of PyTorch first by following
https://pytorch.org/get-started/locally/ — otherwise training/inference
will simply run on CPU, which works fine for this dataset size, just
slower.

## 3. Train the model

From the `shrimp_project` folder, with the environment active:

```bash
python train.py
```

Defaults: `yolov8n.pt` base model, 80 epochs, image size 640, batch
size 16 — tuned to run reasonably on a normal laptop (CPU or GPU). You
can override any of these:

```bash
python train.py --epochs 50 --batch 8 --imgsz 512
```

- On CPU, expect roughly a few hours for the full run depending on
  your machine — for a first test, try `--epochs 10` to make sure
  everything works before committing to a long run.
- On a decent GPU, it should take well under an hour.

Training progress, sample predictions, and metrics (precision/recall,
mAP, confusion matrix) are saved under `runs/detect/shrimp_count/`.

When it finishes, copy the best weights into `models/`:

```bash
# Windows
copy runs\detect\shrimp_count\weights\best.pt models\shrimp_best.pt
# Mac/Linux
cp runs/detect/shrimp_count/weights/best.pt models/shrimp_best.pt
```

## 4. Count shrimp in an image

```bash
python count_image.py --source path\to\your\photo.jpg
```

Or run it on an entire folder of images at once:

```bash
python count_image.py --source path\to\folder_of_images
```

This prints the count per image (and a grand total for folders), and
saves annotated images (with boxes drawn around each detected shrimp)
into `outputs/count_image/`.

Useful flags:
- `--conf 0.3` — raise/lower the confidence threshold (fewer/more
  detections)
- `--model models/shrimp_best.pt` — point at a different weights file

## 5. Count shrimp live from your laptop camera

```bash
python count_webcam.py
```

This opens a window showing your webcam feed with bounding boxes
drawn around detected shrimp and a running total shown at the top.

- Press **q** to quit
- Press **s** to save a snapshot of the current frame (saved to
  `outputs/webcam_snapshots/`)

If you have more than one camera (e.g. a laptop cam and a USB cam),
try `--camera 1`, `--camera 2`, etc. to pick the right one. You can
also run it on a saved video file instead of a live camera:

```bash
python count_webcam.py --source path\to\video.mp4
```

## 6. Troubleshooting

- **"Could not find model weights"** — you need to train first
  (step 3) or point `--model` at wherever your `.pt` file actually is.
- **"Could not open camera"** — another app may be using the camera,
  or you need a different `--camera` index. On Mac, you may also need
  to grant Terminal/VS Code camera permission in System Settings →
  Privacy & Security → Camera.
- **Training is very slow** — lower `--imgsz` (e.g. 416) and
  `--epochs`, or use a smaller subset of the data for a quick test
  run first.
- **Out of memory during training** — lower `--batch` (e.g. 4 or 8).
- **Low accuracy / missed detections** — train for more epochs, try
  the slightly larger `yolov8s.pt` base model (`--model yolov8s.pt`),
  or lower `--conf` in the counting scripts so weaker detections are
  still shown.

## How it works (brief)

- `train.py` fine-tunes a pretrained YOLOv8 detection model on your
  labeled shrimp dataset so it learns to localize and classify
  `Shrimp-baby` vs `shrimp`.
- `count_image.py` and `count_webcam.py` load the trained weights,
  run detection on each image/frame, and simply count how many boxes
  were predicted — that count is your shrimp count.
