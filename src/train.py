from ultralytics import YOLO
import os

def train_model():
    # 1. Load a pretrained YOLO26 pose model
    model = YOLO('yolo26m-pose.pt')

    # 2. Train the model using your custom dataset
    # The 'device' argument '0' specifically targets your first NVIDIA GPU
    results = model.train(
        data='../config/coco8-pose.yaml',
        cache='disk',
        epochs=200,
        imgsz=640,
        device='0',
        workers=12, # Adjust based on your CPU cores for faster data loading
        batch=8,
        patience=20
    )

# The 'main' idiom is required for multiprocessing to work safely
if __name__ == '__main__':
    # freeze_support()  # Only needed if you're turning this into an .exe
    train_model()