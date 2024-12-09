# Thermal Object Detection
A deep learning-based object detection system using YOLO architecture for thermal imagery analysis. This project implements real-time detection of objects like people, vehicles, and bicycles in thermal images using a fine-tuned YOLOv8n model with ONNX runtime optimization.

## Features
- Thermal image object detection
- Support for 4 classes: Person, Car, Bicycle, OtherVehicles
- YOLO model fine-tuning capabilities
- ONNX runtime optimization for inference

## Prerequisites
```bash
pip install -r tqdm requests zipfile onnxruntime torch torchvision numpy opencv-python matplotlib ultralytics imutils PIL
```