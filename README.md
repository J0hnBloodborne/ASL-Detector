A real-time American Sign Language (ASL) letter detection web application built with Python and FastAPI. It utilizes deep learning models to identify and label ASL hand signs (A-Z) from user-provided images or video streams.

Key Components
Web Interface: A FastAPI-based backend (app.py) serving an interactive frontend (index.html, scripts.js) for image upload and real-time detection.

Deep Learning Models: Support for two architectures:

Faster R-CNN: A ResNet50-FPN based object detector.

YOLOv8: A high-performance YOLOv8 nano model for comparison and faster inference.

Training Pipeline: Includes Jupyter notebooks for dataset preprocessing (Roboflow format), model initialization, training loops, and evaluation.

Deployment: A Dockerfile and requirements.txt are provided for containerized deployment using Uvicorn.

/train
This directory contains the machine learning core of the project.

train.ipynb: Training script for the Faster R-CNN model using the ASL dataset in Pascal VOC format. It covers data cleaning, custom DataLoader creation, and model export to .pth.

train_yolo.ipynb: A comparison notebook using Ultralytics YOLOv8. It trains a yolo26n model (26 classes for A-Z) and provides performance metrics like mAP@50.

model.pth & label_map.json: The serialized weights and class definitions used by the web app for inference.
