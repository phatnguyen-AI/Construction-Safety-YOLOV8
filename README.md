<div align="center">

# Construction Site Safety Monitoring with YOLOv8

**AI-powered real-time hard-hat detection to eliminate manual safety checks, reduce workplace injuries, and enforce compliance at scale.**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)
![OpenCV](https://img.shields.io/badge/OpenCV-4.12-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)

`Real-time Detection` · `Edge-Deployable` · `End-to-End ML Pipeline` · `Data-Centric AI`

</div>

---

## Business Context & Problem Statement

Construction sites are among the most hazardous work environments globally. The **International Labour Organization (ILO)** estimates that over **60,000 fatal accidents** occur on construction sites each year, with head injuries being one of the leading causes of death and disability.

### The Manual Process — and Why It Fails

| Pain Point | Impact |
|---|---|
| **Labor-intensive gate checks** | Requires dedicated personnel at every entry point, 24/7 |
| **Human error & fatigue** | Compliance drops significantly during shift changes and peak hours |
| **No audit trail** | Manual checks leave no photographic evidence for incident investigation |
| **Non-scalable** | Adding gates or shifts means linearly adding headcount |

### The Opportunity

Replace manual helmet inspections with an **always-on, camera-based AI system** that:
- Detects helmet violations in **real-time** (<50ms per frame)
- Triggers **instant PA speaker alerts** when a violation is detected
- Captures violation snapshots and routes them to safety managers
- Scales to **multiple gates** with zero additional labor cost

---

## Solution Overview

```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  CCTV Camera │────▶│  YOLOv8 Inference │────▶│  Decision Engine    │
│  (Gate Feed) │     │  (Edge GPU)       │     │                     │
└──────────────┘     └──────────────────┘     │  ┌───────────────┐  │
                                               │  │ Helmet  → OK   │  │
                                               │  │ Head    → ALERT│  │
                                               │  └───────────────┘  │
                                               └────────┬────────────┘
                                                        │
                                          ┌─────────────┼─────────────┐
                                          ▼             ▼             ▼
                                    ┌──────────┐ ┌───────────┐ ┌──────────┐
                                    │ PA Speaker│ │ Dashboard │ │ Evidence │
                                    │ Alert     │ │ Logging   │ │ Storage  │
                                    └──────────┘ └───────────┘ └──────────┘
```

**Value Chain:** Camera captures entry → Model classifies `helmet` vs `head` → Violation triggers multi-channel response (audio alert + manager notification + evidence archiving).

---

## Key Results

<div align="center">

<img width="1107" alt="Model Performance Metrics" src="https://github.com/user-attachments/assets/78b6670a-0cbf-4d58-a2fe-7282e193e04d" />

*YOLOv8s performance after fine-tuning on the construction safety dataset*

</div>

### Before vs. After Fine-Tuning

| Aspect | Before (Pretrained YOLOv8s) | After (Fine-Tuned) |
|---|---|---|
| **Domain Fit** | Generic COCO classes — no `head`/`helmet` distinction | Custom 2-class detector optimized for construction PPE |
| **Detection Quality** | Missed most helmets; high false-positive rate | Accurate bounding boxes on both classes |
| **Business Readiness** | Not usable for safety enforcement | Deployment-ready for gate monitoring |

> The pretrained YOLOv8s was evaluated against the construction dataset as a baseline (`performance_before_finetuning/`). After fine-tuning with augmented data, the model achieved significant improvements across all metrics (`performance_after_finetuning/`).

---

## Technical Deep Dive — ML Pipeline

<details>
<summary><strong>1. Data Collection</strong> — Automated dataset acquisition</summary>

**What:** Pulled a labeled construction-safety dataset from Roboflow (version 3) via their Python SDK.

**Why:** Roboflow provides pre-labeled object detection datasets, enabling rapid prototyping without manual annotation from scratch. API-based download ensures reproducibility.

**Script:** [`Dataset/Get_dataset.py`](Dataset/Get_dataset.py)

</details>

<details>
<summary><strong>2. Data Quality Audit</strong> — Visual label verification</summary>

**What:** Built a custom visualization tool that randomly samples training images and overlays their YOLO-format bounding boxes to visually inspect annotation quality.

**Why:** The free-tier Roboflow dataset contained noisy and inconsistent labels. Blindly training on dirty data would degrade model performance. This step identified mislabeled, missing, and misaligned annotations that were corrected before training.

**Script:** [`Label_checking/label_checking.py`](Label_checking/label_checking.py)

</details>

<details>
<summary><strong>3. Exploratory Data Analysis</strong> — Understanding the data distribution</summary>

**What:** Statistical profiling of the dataset — class distribution, image dimensions, annotation density per image.

**Why:** Understanding data characteristics informs augmentation strategy and reveals class imbalance early, before it silently degrades model performance on minority classes.

**Notebook:** [`Image_stat/Image_stat.ipynb`](Image_stat/Image_stat.ipynb)

</details>

<details>
<summary><strong>4. Class Imbalance Handling</strong> — Augmentation pipeline for minority oversampling</summary>

**What:** Detected significant under-representation of the `head` class (no-helmet violations). Built a full augmentation pipeline using **Albumentations** to synthesize additional minority-class samples:
- Vertical flip (p=0.3)
- Random brightness/contrast (p=0.5)
- Rotation ±20° (p=0.5)
- Hue/saturation shift (p=0.3)

Bounding boxes are transformed alongside images to maintain label consistency. The script automatically updates `data.yaml` to include augmented data paths.

**Why:** In a safety-critical application, **missing a violation (false negative) is far more costly than a false alarm**. Class imbalance would bias the model toward the majority class (`helmet`), causing it to under-detect the exact cases we care about most — exposed heads.

**Script:** [`fix_imbalanced_data/fix_imbalanced_data.py`](fix_imbalanced_data/fix_imbalanced_data.py)

</details>

<details>
<summary><strong>5. Model Training</strong> — Fine-tuning YOLOv8s</summary>

**What:** Fine-tuned the `yolov8s` (small) variant for 100 epochs at 640×640 resolution with batch size 16.

**Why:** YOLOv8s offers the optimal trade-off between inference speed and accuracy for edge deployment. The small variant runs efficiently on edge GPUs (e.g., NVIDIA Jetson) while maintaining sufficient detection quality for the 2-class problem.

| Hyperparameter | Value | Rationale |
|---|---|---|
| Model | YOLOv8s | Speed-accuracy balance for edge inference |
| Epochs | 100 | Sufficient convergence for a 2-class fine-tune |
| Image Size | 640×640 | Standard YOLO input; balances detail vs. speed |
| Batch Size | 16 | Fits comfortably in GPU memory |

**Script:** [`Fineturning_model/Train.py`](Fineturning_model/Train.py)

</details>

<details>
<summary><strong>6. Performance Evaluation</strong> — Quantitative + qualitative assessment</summary>

**What:** Two-pronged evaluation approach:
1. **Quantitative** — `model.val()` on the validation set to compute mAP, precision, recall.
2. **Qualitative** — Visual inference on 20 random test images to manually inspect detection quality, edge cases, and failure modes.

**Why:** Metrics alone can be misleading. Visual inspection catches failure patterns (e.g., occluded heads, unusual helmet colors) that aggregate metrics obscure. Both before-and-after comparisons provide clear evidence of fine-tuning impact.

**Scripts:**
- Before fine-tuning: [`performance_before_finetuning/`](performance_before_finetuning/)
- After fine-tuning: [`performance_after_finetuning/`](performance_after_finetuning/)

</details>

---

## Project Structure

```
Construction-Safety-YOLOV8/
│
├── Dataset/
│   └── Get_dataset.py              # Automated dataset download from Roboflow API
│
├── Label_checking/
│   └── label_checking.py            # Visual annotation quality inspector
│
├── Image_stat/
│   └── Image_stat.ipynb             # EDA — class distribution & dataset profiling
│
├── fix_imbalanced_data/
│   └── fix_imbalanced_data.py       # Albumentations-based minority oversampling
│
├── Fineturning_model/
│   └── Train.py                     # YOLOv8s fine-tuning script
│
├── Model/
│   └── fine_turned.pt               # Trained model weights (~45MB)
│
├── performance_before_finetuning/
│   ├── manual.py                    # Visual inference (baseline)
│   └── test_by_metric.py            # Metric evaluation (baseline)
│
├── performance_after_finetuning/
│   ├── manual.py                    # Visual inference (fine-tuned)
│   └── test_by_metric.py            # Metric evaluation (fine-tuned)
│
├── Images_Video/                    # Demo assets (video, screenshots)
├── requirement.txt                  # Python dependencies
└── README.md
```

---

## Demo

<div align="center">

https://github.com/user-attachments/assets/cc7f7ecf-a182-4bd9-82b2-74a64979732a

*Real-time hard-hat detection on construction site footage*

</div>

---

## Tech Stack

| Category | Tools |
|---|---|
| **Object Detection** | YOLOv8s (Ultralytics) |
| **Data Augmentation** | Albumentations |
| **Image Processing** | OpenCV, Pillow |
| **Data Analysis** | Pandas, NumPy, Seaborn, Matplotlib |
| **Dataset Source** | Roboflow |
| **Model Format** | PyTorch (.pt) |

---

## Getting Started

### Prerequisites
- Python 3.10+
- GPU recommended (NVIDIA CUDA-compatible)

### Installation

```bash
# Clone the repository
git clone https://github.com/phatnguyen-DS/Construction-safety-yolov8.git
cd Construction-safety-yolov8

# Install dependencies
pip install -r requirement.txt
```

### Quick Run

```bash
# 1. Download the dataset
python Dataset/Get_dataset.py

# 2. (Optional) Inspect label quality
python Label_checking/label_checking.py

# 3. (Optional) Fix class imbalance via augmentation
python fix_imbalanced_data/fix_imbalanced_data.py

# 4. Train the model
python Fineturning_model/Train.py

# 5. Evaluate performance
python performance_after_finetuning/test_by_metric.py
```

> **Note:** A Roboflow API key is required for dataset download. Set it as an environment variable: `ROBoflow_API_KEY=your_key`

---

## Future Roadmap

| Phase | Initiative | Business Impact |
|---|---|---|
| **Edge Deployment** | Export to TensorRT / ONNX for NVIDIA Jetson | Sub-20ms inference; production-ready hardware |
| **Multi-Camera Support** | Parallel stream processing with async inference | Scale to 10+ gates with a single server |
| **Violation Dashboard** | Web-based logging UI with search, filters, and export | Audit trail for regulatory compliance |
| **Model Monitoring** | Drift detection + automated retraining triggers | Sustained accuracy as site conditions change |
| **Additional PPE Classes** | Extend to vests, gloves, safety glasses | Comprehensive safety enforcement |

---

## Author

**Phat Nguyen** — Data Scientist & ML Engineer

[![GitHub](https://img.shields.io/badge/GitHub-phatnguyen--DS-181717?style=for-the-badge&logo=github)](https://github.com/phatnguyen-DS)

---


