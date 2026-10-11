# 🦐 Shrimp Detection and Counting using YOLOv8

A computer vision project that detects shrimp in images using a custom-trained YOLOv8 object detection model. The project includes Python scripts for image-based detection and webcam-based detection.

## 📌 Project Overview

Shrimp detection can help automate the identification and counting of shrimp in aquaculture images. This project uses a YOLOv8 model trained on a labeled dataset to locate shrimp and draw bounding boxes around detected objects.

## ✨ Features

- Shrimp detection using YOLOv8
- Bounding-box visualization with confidence scores
- Image-based detection using Python
- Webcam-based detection script
- Support for two dataset classes: `Shrimp-baby` and `shrimp`

## 🛠️ Technologies Used

- Python
- Ultralytics YOLOv8
- OpenCV
- NumPy

## 📂 Project Structure

```text
shrimp-counting-yolov8/
├── count_image.py
├── count_webcam.py
├── train.py
├── requirements.txt
├── README.md
└── prediction images
```

## 🔍 Sample Results

The following sample images show the model's shrimp detections. Bounding boxes indicate detected objects, and the displayed values represent model confidence scores.

### Sample 1 — Shrimp Detection
![Shrimp detection sample 1](6451674985940_-pic_jpg.rf.c8da0cadb7ecaf1aaf6d52bc2f3367d3.jpg)

### Sample 2 — Multiple Shrimp Detection
![Shrimp detection sample 2](6581674986076_-pic_jpg.rf.e2e70490ef0abd45f7d54ef65fb24693.jpg)

### Sample 3 — Crowded Scene Detection
![Shrimp detection sample 3](8161675603780_-pic_jpg.rf.4b8485a8701d54dea65aa9927d2d6d21.jpg)

## ⚙️ Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/MuralaYamini/shrimp-counting-yolov8.git
   cd shrimp-counting-yolov8
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```

   Windows:

   ```bash
   .venv\Scripts\activate
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Detect shrimp in an image

```bash
python count_image.py --source path/to/image.jpg
```

### Run webcam detection

```bash
python count_webcam.py
```

Check the Python scripts for supported command-line options and model-weight paths.

## 🧠 Model and Dataset

The model was trained using a Roboflow shrimp dataset containing 5,689 annotated images and two classes: `Shrimp-baby` and `shrimp`.

The trained model weights are not included in this repository yet. You will need the compatible trained `.pt` file to run custom shrimp detection.

## 📈 Current Status

- Custom YOLOv8 model trained
- Prediction images generated
- Bounding-box detection demonstrated
- Image and webcam scripts included

Detection quality and counting accuracy should be evaluated further on representative images before making performance claims.

## 👩‍💻 Author

**Yamini Murala**

GitHub: [@MuralaYamini](https://github.com/MuralaYamini)
