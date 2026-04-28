"""
Label Quality Check

Randomly samples training images and renders bounding boxes
to visually verify annotation correctness before training.
"""

import glob
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import random

PWD_dataset_folder = "../Dataset/Phat_project-3"
n_sample = 20

# Class ID to (name, color) mapping
class_map = {
    0: ("head", "red"),
    1: ("helmet", "blue")
}

images = glob.glob(f"{PWD_dataset_folder}/train/images/*.jpg")

for img_path in random.sample(images, n_sample):
    filename_wo_ext = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(f"{PWD_dataset_folder}/train/labels", filename_wo_ext + ".txt")

    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            content = f.read().strip()

        img_ = Image.open(img_path)
        draw = ImageDraw.Draw(img_)
        W, H = img_.size

        if content:
            for line in content.splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue

                class_id, x_center, y_center, w, h = map(float, parts)

                # Convert normalized YOLO coords to pixel coords
                x_center *= W
                y_center *= H
                w *= W
                h *= H

                x_min = x_center - w / 2
                y_min = y_center - h / 2
                x_max = x_center + w / 2
                y_max = y_center + h / 2

                class_name, color = class_map.get(int(class_id), ("unknown", "white"))
                draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=2)
                draw.text((x_min, max(y_min - 12, 0)), class_name, fill=color)

        plt.figure(figsize=(8, 8))
        plt.imshow(img_)
        plt.axis('off')
        plt.show()