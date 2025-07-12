````markdown
# Real-Time Object Detection System

This project implements a real-time object detection system using the YOLOv8 model and OpenCV.
It captures live video from a webcam, processes each frame to detect objects
and displays the results with class labels and confidence scores.

## Features

- Real-time video capture and object detection
- Utilizes YOLOv8 for high-speed and accurate recognition
- Displays bounding boxes, class names, and confidence levels
- Custom color assignment for each detected class

## Requirements

- Python 3.8+
- OpenCV
- Ultralytics (YOLOv8)

## Installation

1. **Clone the Repository**

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
````

2. **Create a Virtual Environment (Optional but Recommended)**

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

> Note: Ensure you have `torch` and the `ultralytics` package installed.
> If not, install them via:

```bash
pip install ultralytics opencv-python
```

## Usage

Run the application using:

```bash
python main.py
```

Press `q` to exit the video stream.

## Model

This application uses the YOLOv8s model (`yolov8s.pt`) by default. 
Make sure the model file is available or install it using the Ultralytics CLI:

```bash
yolo task=detect mode=train model=yolov8s.pt
```

## Author

**Sohan** • [afmtechz.anvil.app](https://afmtechz.anvil.app)

## Acknowledgements

* [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* [OpenCV](https://opencv.org/)
