# COVID-19 Radiography Classifier 🩺 ⚡️

This project leverages Deep Learning to classify chest X-ray images into four categories: COVID-19, Lung Opacity, Normal, and Viral Pneumonia.

---

## 📊 About the Dataset

The project utilizes the widely recognized **COVID-19 Radiography Database**, which contains over 21,000 digital chest X-ray images.

### How to obtain the data:
1. Download the dataset from Kaggle: [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database).
2. Extract the archive into the project root.
3. Ensure the dataset folder is named: `COVID-19_Radiography_Dataset/`.

---

## 🛠 Tech Stack

- **Python 3.13+**
- **PyTorch** (Core Deep Learning framework)
- **Keras/TensorFlow** (Data loading utilities)
- **Torchvision** (Image transformations)
- **Scikit-learn** (Preprocessing & metrics)
- **Pandas** (Data handling)
- **Matplotlib** (Visualization)
- **uv** (Environment & dependency management)

---

## 📂 Project Structure

- `train.py`: Main script for model training and evaluation.
- `DataPred.py`: Data loading and preprocessing utilities.
- `main.py`: Entry point placeholder.
- `COVID-19_Radiography_Dataset/`: Dataset directory (to be added manually).

---

## 🏗 Model Architecture

The classification is performed by a custom Convolutional Neural Network (CNN):
- **4 Convolutional Layers** with BatchNorm and MaxPool.
- **Adaptive Average Pooling** for flexible output sizing.
- **Fully Connected Layers** with Dropout (0.5) to prevent overfitting.
- **Input size**: `224x224` (Grayscale).

---

## 🚀 Quick Start

### 1. Installation
This project uses the `uv` package manager for high-performance dependency management. If you don't have it, install it via [astral.sh](https://astral.sh/uv).

```bash
# Clone the repository
git clone https://github.com/your-username/COVID-19_RXRayClassifier.git
cd COVID-19_RXRayClassifier

# Sync dependencies and create a virtual environment
uv sync
```

### 2. Training the Model
To start the training process, run:
```bash
uv run train.py
```

---

## 📈 Results
Upon completion of the training, the script generates a **Confusion Matrix** and reports the **Accuracy** on the test set, providing a detailed breakdown of the model's performance across all classes.

---

## 👨‍💻 Author
**Khrystyna Pavlivets** - [GitHub Profile](https://github.com/khrystynapavlivets)
