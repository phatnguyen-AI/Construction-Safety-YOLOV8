"""
Baseline Metric Evaluation

Evaluates pretrained YOLOv8s on the validation set
to establish quantitative baseline metrics (mAP, precision, recall).
"""

from ultralytics import YOLO

PWD_dataset_folder = "../Dataset/Phat_project-3"
batch_size = 16
img_size = 640

model = YOLO("yolov8s.pt")
metrics = model.val(data=f"{PWD_dataset_folder}/data.yaml", batch=batch_size, imgsz=img_size)
print(metrics)