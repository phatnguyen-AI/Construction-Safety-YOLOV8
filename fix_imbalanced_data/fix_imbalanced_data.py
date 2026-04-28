"""
Class Imbalance Fix

Oversamples the minority class ('head') via Albumentations-based
augmentation to balance the training set. Updates data.yaml
to include augmented data paths, then verifies the new distribution.
"""

import pandas as pd
import os
import yaml
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

PWD_dataset_folder = "../Dataset/Phat_project-3"

# ---------------------------------------------------------------------------
# 1. Determine how many augmented samples are needed
# ---------------------------------------------------------------------------
table = pd.read_csv("table.csv", index_col=0)

head_count = table.loc['0', 'train']
helmet_count = table.loc['1', 'train']
target_count = helmet_count
needed_head = target_count - head_count

print(f"Current 'head' count : {head_count}")
print(f"Current 'helmet' count: {helmet_count}")
print(f"Target count          : {target_count}")
print(f"Augmentations needed  : {needed_head}")

# ---------------------------------------------------------------------------
# 2. Index images by class
# ---------------------------------------------------------------------------
class_paths = {
    0: {"images": [], "labels": []},  # head
    1: {"images": [], "labels": []}   # helmet
}

label_dir = f"{PWD_dataset_folder}/train/labels"
image_dir = f"{PWD_dataset_folder}/train/images"

for label_filename in os.listdir(label_dir):
    if label_filename.endswith(".txt"):
        label_path = os.path.join(label_dir, label_filename)
        image_filename = label_filename.replace(".txt", ".jpg")
        image_path = os.path.join(image_dir, image_filename)

        classes_in_image = set()
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) > 0:
                        class_id = int(parts[0])
                        classes_in_image.add(class_id)

        for class_id in class_paths.keys():
            if class_id in classes_in_image:
                class_paths[class_id]["images"].append(image_path)
                class_paths[class_id]["labels"].append(label_path)

print(f"Images containing 'head'  : {len(class_paths[0]['images'])}")
print(f"Images containing 'helmet': {len(class_paths[1]['images'])}")


# ---------------------------------------------------------------------------
# 3. Augmentation function
# ---------------------------------------------------------------------------
def augment_image_and_labels(image_path, label_path, num_augmentations,
                              output_image_dir, output_label_dir, augmenter):
    """
    Apply random augmentations to an image and its YOLO-format labels.

    Bounding boxes are transformed alongside the image to maintain
    label consistency. Outputs are saved as new image/label pairs.

    Args:
        image_path: Path to the source image.
        label_path: Path to the YOLO label file.
        num_augmentations: Number of augmented copies to generate.
        output_image_dir: Directory to save augmented images.
        output_label_dir: Directory to save augmented labels.
        augmenter: An albumentations.Compose pipeline with bbox support.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to read image: {image_path}")
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) == 5:
                        class_id, x_center, y_center, width, height = map(float, parts)
                        labels.append([class_id, x_center, y_center, width, height])

        # Format: [x_c, y_c, w, h, class_id] — required by Albumentations YOLO format
        bboxes = [[l[1], l[2], l[3], l[4], int(l[0])] for l in labels]
        base_filename = os.path.splitext(os.path.basename(image_path))[0]

        for i in range(num_augmentations):
            augmented = augmenter(image=img, bboxes=bboxes)
            augmented_img = augmented['image']
            augmented_bboxes = augmented['bboxes']

            augmented_img_bgr = cv2.cvtColor(np.array(augmented_img), cv2.COLOR_RGB2BGR)

            aug_img_path = os.path.join(output_image_dir, f"{base_filename}_aug_{i}.jpg")
            aug_lbl_path = os.path.join(output_label_dir, f"{base_filename}_aug_{i}.txt")

            cv2.imwrite(aug_img_path, augmented_img_bgr)

            with open(aug_lbl_path, 'w') as f:
                for bbox in augmented_bboxes:
                    cls = int(bbox[4])
                    x_c, y_c, w, h = bbox[:4]
                    f.write(f"{cls} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

    except Exception as e:
        print(f"Error augmenting {image_path}: {e}")


# ---------------------------------------------------------------------------
# 4. Run augmentation on the minority class
# ---------------------------------------------------------------------------
augmented_train_image_dir = f"{PWD_dataset_folder}/train/augmented_images"
augmented_train_label_dir = f"{PWD_dataset_folder}/train/augmented_labels"

os.makedirs(augmented_train_image_dir, exist_ok=True)
os.makedirs(augmented_train_label_dir, exist_ok=True)

augment = A.Compose([
    A.VerticalFlip(p=0.3),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=20, p=0.5),
    A.HueSaturationValue(p=0.3),
], bbox_params=A.BboxParams(format='yolo', clip=True))

num_head_images = len(class_paths[0]['images'])

if num_head_images > 0:
    augmentations_per_head_image = needed_head // num_head_images
else:
    augmentations_per_head_image = 0

print(f"Augmenting 'head' class — needed: {needed_head}, "
      f"images: {num_head_images}, per image: {augmentations_per_head_image}")

augmented_head_count = 0

if num_head_images > 0:
    for img_path, label_path in zip(class_paths[0]['images'], class_paths[0]['labels']):
        augment_image_and_labels(
            img_path, label_path,
            augmentations_per_head_image,
            augmented_train_image_dir,
            augmented_train_label_dir,
            augment
        )
        augmented_head_count += augmentations_per_head_image

print(f"\nTotal augmented 'head' samples created: {augmented_head_count}")

# ---------------------------------------------------------------------------
# 5. Update data.yaml to include augmented paths
# ---------------------------------------------------------------------------
data_yaml_path = f"{PWD_dataset_folder}/data.yaml"
augmented_train_image_dir = f"{PWD_dataset_folder}/train/augmented_images"

with open(data_yaml_path, 'r') as f:
    data_yaml = yaml.safe_load(f)

original_train_image_dir = f"{PWD_dataset_folder}/train/images"
data_yaml['train'] = [original_train_image_dir, augmented_train_image_dir]

with open(data_yaml_path, 'w') as f:
    yaml.dump(data_yaml, f)

print(f"Updated {data_yaml_path} with augmented paths.")
print(f"New 'train' paths: {data_yaml['train']}")

# ---------------------------------------------------------------------------
# 6. Verify class distribution after augmentation
# ---------------------------------------------------------------------------
original_train_label_dir = f"{PWD_dataset_folder}/train/labels"
augmented_train_label_dir = f"{PWD_dataset_folder}/train/augmented_labels"
val_label_dir = f"{PWD_dataset_folder}/valid/labels"
test_label_dir = f"{PWD_dataset_folder}/test/labels"

dict_ = {
    "train": {"0": 0, "1": 0},
    "val":   {"0": 0, "1": 0},
    "test":  {"0": 0, "1": 0}
}


def count(folder_path: str, split: str):
    """Count class occurrences across all label files in a directory."""
    if not os.path.exists(folder_path):
        print(f"Warning: directory not found: {folder_path}")
        return

    for filename in os.listdir(folder_path):
        if not filename.endswith('.txt'):
            continue
        filepath = os.path.join(folder_path, filename)
        try:
            with open(filepath, "r") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) > 0:
                        class_id = int(parts[0])
                        key = str(class_id)
                        if key in dict_[split]:
                            dict_[split][key] += 1
        except Exception as e:
            print(f"Error reading {filepath}: {e}")


count(original_train_label_dir, "train")
count(augmented_train_label_dir, "train")
count(val_label_dir, "val")
count(test_label_dir, "test")

table = pd.DataFrame(dict_)

print("Class distribution after augmentation:")
display(table)

# Pie chart for training set distribution
percent_train = table['train'].div(table['train'].sum()) * 100
labels = ['head', 'helmet']
colors = sns.color_palette('pastel')[0:len(percent_train)]

plt.figure(figsize=(6, 6))
plt.pie(percent_train, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('Class Distribution After Augmentation (Train)')
plt.show()