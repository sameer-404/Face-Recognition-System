# Face Recognition System

Real-time face recognition using DeepFace and OpenCV.

## Features
- Live face detection and tracking
- Face verification against reference image
- Green/red borders for match status

## Requirements
```bash
pip install opencv-python deepface tf-keras
```

## Usage

1. Add your reference image as `refrence.jpg`
2. Run the program:
```bash
python3 main.py
```
3. Press 'q' to quit

## Note
First run downloads AI models (~1-2GB). Requires camera permissions on macOS.
