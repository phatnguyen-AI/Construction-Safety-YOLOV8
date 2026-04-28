from roboflow import Roboflow
import os

# Download dataset from Roboflow (version 3, YOLOv8 format)
rf = Roboflow(api_key=os.getenv("ROBoflow_API_KEY"))
project = rf.workspace("phat-tlik5").project("phat_project-mlezd")
version = project.version(3)
dataset = version.download("yolov8")