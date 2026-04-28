"""
Post Fine-Tuning Metric Evaluation

Evaluates the fine-tuned model on the validation set
to measure mAP, precision, and recall improvements.
"""

from ultralytics import YOLO

PWD_dataset_folder = "../Dataset/Phat_project-3"
batch_size = 16
img_size = 640

model = YOLO(f"{PWD_dataset_folder}/best.pt")
metrics = model.val(data=f"{PWD_dataset_folder}/data.yaml", batch=batch_size, imgsz=img_size)
print(metrics)