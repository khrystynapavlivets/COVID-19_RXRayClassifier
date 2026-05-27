# COVID-19 Radiography Classifier

Deep learning pipeline for **multi-class classification** of chest X-ray images into four categories: **COVID-19**, **Lung Opacity**, **Normal**, and **Viral Pneumonia**. Built with PyTorch and transfer learning (ResNet-18).

---

## Results


| Metric                     | Value                                                    |
| -------------------------- | -------------------------------------------------------- |
| **Hold-out test accuracy** | **90.74%**                                               |
| Test split                 | 20% (`train_test_split`, `test_size=0.2`)                |
| Metric                     | `sklearn.metrics.accuracy_score` on hold-out predictions |
| Evaluation output          | `Final Test Accuracy` printed at end of `train.py`       |


---

## Dataset

Uses the [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) (21,000+ chest X-ray images).

### Setup

1. Download the dataset from Kaggle.
2. Extract the archive into the project root.
3. Ensure the folder is named: `COVID-19_Radiography_Dataset/`.

Expected layout:

```
COVID-19_Radiography_Dataset/
├── COVID/images/
├── Lung_Opacity/images/
├── Normal/images/
└── Viral_Pneumonia/images/
```

---

## Model & Training

### Architecture

- **Backbone:** [ResNet-18](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) pretrained on ImageNet (`ResNet18_Weights.DEFAULT`)
- **Head:** `Dropout(0.4)` + `Linear` → 4 output classes (`CovidResNet` in `train.py`)
- **Input:** 224×224 RGB (grayscale X-rays converted via `PIL.Image.convert("RGB")`)

### Training configuration


| Parameter           | Value                                                |
| ------------------- | ---------------------------------------------------- |
| Optimizer           | Adam (`lr=0.001`)                                    |
| Loss                | Weighted `CrossEntropyLoss` (balanced class weights) |
| Scheduler           | `ReduceLROnPlateau` (patience=2, factor=0.5)         |
| Epochs              | 7                                                    |
| Batch size          | 64                                                   |
| Train augmentations | Random horizontal flip, random rotation (±10°)       |
| Normalization       | ImageNet mean/std                                    |
| Device              | MPS (Apple Silicon) if available, else CPU           |


### Data split

- **80%** training / **20%** hold-out test
- Labels encoded with `LabelEncoder`

---

## Tech Stack

- **Python 3.13+**
- **PyTorch** & **Torchvision** (model, transforms, pretrained weights)
- **scikit-learn** (split, metrics, class weights, confusion matrix)
- **Pandas** & **Pillow** (dataset indexing and image I/O)
- **Matplotlib** (visualization)
- **uv** (dependency and environment management)

---

## Project Structure


| File / folder                   | Description                                                     |
| ------------------------------- | --------------------------------------------------------------- |
| `train.py`                      | Training loop, evaluation, confusion matrix, sample plots       |
| `DataPred.py`                   | Dataset loading (`CovidDatasetLoader`) and sample visualization |
| `COVID-19_Radiography_Dataset/` | Local dataset (not included in repo; download manually)         |
| `pyproject.toml` / `uv.lock`    | Dependencies                                                    |


---

## Quick Start

### 1. Install dependencies

Requires [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/khrystynapavlivets/COVID-19_RXRayClassifier.git
cd COVID-19_RXRayClassifier
uv sync
```

### 2. Train

```bash
uv run train.py
```

---

## Author

**Khrystyna Pavlivets** — [GitHub](https://github.com/khrystynapavlivets)