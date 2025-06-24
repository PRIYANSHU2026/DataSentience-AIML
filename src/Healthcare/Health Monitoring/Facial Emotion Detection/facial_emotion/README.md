# Facial Emotion Detection

This project detects facial emotions in real-time using a webcam feed, powered by PyTorch (ResNet18) and OpenCV.

## Features
- Trains on grayscale 48x48 images for 7 emotion classes.
- Real-time detection using webcam.
- Data augmentation during training for robustness.

## Dataset
FER-2013 dataset formatted into `train/` and `test/` folders.

## How to Run

### Training
```bash
python face_emotion_train.py
```

### Real-Time Detection
```bash
python face_emotion_realtime.py
```

## Dependencies

Install required packages using:
```bash
pip install -r requirements.txt
```