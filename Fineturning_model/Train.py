"""
Model Training

Fine-tunes YOLOv8s on the construction safety dataset.
"""

from ultralytics import YOLO

PWD_dataset_folder = "../Dataset/Phat_project-3"
epochs = 100
images_size = 640
batch_size = 16

model = YOLO("yolov8s.pt")
model.train(data=f"../data.yaml", epochs=epochs, imgsz=images_size, batch=batch_size)