# 🧠 Brain Tumor Segmentation using U-Net

## 📌 Project Overview

This project uses Deep Learning and Computer Vision techniques to perform Brain Tumor Segmentation on MRI scans using the U-Net architecture.

The model is trained to identify and segment tumor regions from MRI images, helping improve medical image analysis and assisting in early diagnosis workflows.

---

## 🚀 Features

- MRI Brain Tumor Segmentation
- U-Net Architecture with ResNet34 Encoder
- Dice Loss + BCE Loss
- Data Augmentation using Albumentations
- PyTorch-based Training Pipeline
- Tumor Mask Prediction
- Overlay Visualization
- Dice Score Evaluation

---

## 🛠️ Technologies Used

- Python
- PyTorch
- OpenCV
- NumPy
- Albumentations
- Segmentation Models PyTorch
- Matplotlib
- Google Colab

---

## 🧠 Model Architecture

- U-Net
- ResNet34 Encoder
- Binary Segmentation
- Attention-based Feature Learning

---

## 📊 Model Performance

| Metric | Score |
|--------|--------|
| Dice Score | 87% |
| Loss Function | Dice + BCE Loss |

---

## 📂 Dataset Used

- LGG MRI Segmentation Dataset
- BraTS Dataset (for experimentation)

Dataset contains MRI brain scans and corresponding tumor masks for supervised segmentation training.

---

## 🔄 Project Workflow

```text
MRI Image
    ↓
Preprocessing
    ↓
Data Augmentation
    ↓
U-Net Segmentation
    ↓
Tumor Mask Prediction
    ↓
Overlay Visualization
```

---

## 🖼️ Sample Results

(Add screenshots here later)

- Original MRI
- Ground Truth Mask
- Predicted Mask
- Overlay Visualization

---

## ⚙️ How to Run

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/Brain_Tumour_Segmentation.git
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Notebook

Open:

```text
Brain_tumour_Segmentation.ipynb
```

---

## 📈 Future Improvements

- Attention U-Net
- 3D Medical Segmentation
- Better Encoder Architectures
- Deployment Interface
- Real-time MRI Analysis

---

## 👨‍💻 Author

Dev Shrivastava
