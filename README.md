#  Brain Tumor Detection: Classical ML vs. Deep Learning

## Executive Summary
This repository contains a comprehensive, end-to-end Data Science project aimed at detecting Brain Tumors from raw MRI scans. 

To demonstrate a full mastery of Computer Vision and Artificial Intelligence, this project was deliberately split into two distinct phases to compare methodologies:
1. **Phase 1 (Classical Machine Learning):** Advanced mathematical feature extraction (HOG, GLCM, Gabor, Wavelets) paired with an automated algorithm bake-off (Random Forest, KNN, SVC, Naive Bayes, Decision Trees).
2. **Phase 2 (Deep Learning):** A completely custom-built Convolutional Neural Network (CNN) engineered from scratch in PyTorch, utilizing strict K-Fold Cross Validation and Early Stopping.

**Goal:** To establish a Classical ML baseline and attempt to surpass it using modern Deep Learning architectures.

## The Dataset
- **Volume:** 6,999 high-resolution Brain MRI scans.
- **Classes:** Binary Classification (Normal vs. Tumor).
  - Normal Class: 3,799 scans
  - Tumor Class: 3,200 scans

## Project Architecture
```text
Brain-Tumor-Detection/
│
├── data/
│   ├── Normal/             # 3,799 Normal MRI scans
│   └── Tumor/              # 3,200 Tumor MRI scans
│
├── DL.ipynb                # Phase 2: PyTorch Deep Learning CNN Pipeline
├── PR_ML.ipynb             # Phase 1: Scikit-Image & Classical ML Pipeline
│
├── requirements.txt        # Conda/Pip environment dependencies
└── README.md
```

---

## Phase 1: Classical Machine Learning (`PR_ML.ipynb`)
Before jumping straight to Deep Learning, this repository constructs a strict Classical ML baseline. Because traditional algorithms cannot automatically learn features from pixels, we engineered a massively complex mathematical extraction pipeline.

### Highlights:
- **Image Preprocessing:** MRI scans are subjected to a 7-step filtration regime using `scikit-image`. Noise is eradicated via **Gaussian & Median Filtering**, while dense boundaries are highlighted using **CLAHE** (Contrast Limited Adaptive Histogram Equalization) and **Unsharp Masking**.
- **Feature Extraction (~3,928 Metrics per Image):**
  - **HOG:** Spatial geometries and edge detection.
  - **LBP:** Micro-texture extraction at multi-radius scales.
  - **Gabor Filters & Wavelets:** Frequency-domain compression and directional analysis.
  - **GLCM:** Gray Level Co-occurrence Matrix texture property charting.
- **Dimensionality Reduction (PCA):** Mathematical compression of the dataset to prevent the "Curse of Dimensionality" and over-fitting, strictly retaining `95%` true variance.
- **The ML Bake-Off:** Five separate algorithms (Random Forest, K-Nearest Neighbors, SVC, Decision Tree, Naive Bayes) are rigorously pushed through 5-Fold Cross-Validation, strictly scoring against physical compute time, accuracy, and overfitting gaps to crown a champion.


---

## Phase 2: Deep Learning (`DL.ipynb`)
To push accuracy and classification robustness to the next level, Phase 2 implements a completely custom PyTorch Convolutional Neural Network (CNN) inspired by VGG-style architectures. 

Unlike Phase 1, where features were manually extracted, this architecture dynamically learns hierarchical shapes, textures, and borders directly from raw physical pixel tensors using Deep Learning.

### Highlights:
- **Custom Architecture:** A robust `MyModel` PyTorch Neural Network constructed from scratch with Max Pooling, ReLU non-linearities, and highly structured Linear Classification layers.
- **Dataset Standardization:** Automated dynamic resizing and 1-channel Grayscale tensor casting using `torchvision.transforms`.
- **Absolute Data Integrity:** Implemented a strict 80/20 physical split separating blind **Testing Data** from the Training/Validation loop, entirely preventing data leakage.
- **Rigorous K-Fold Validation:** The training block utilizes dynamic 5-Fold Cross-Validation, resetting the model weights on each split to mathematically prove stability across multiple data distributions.
- **Early Stopping & Patience:** Implemented automated halt mechanisms to freeze training epochs if validation loss plateaus, completely stopping the model from overfitting.

---

## How to Run Locally
If you would like to test or run these notebooks on your own machine, simply clone this repository and install the dependencies.

```bash
# 1. Clone the Repository
git clone https://github.com/YourUsername/Brain-Tumor-Detection.git
cd Brain-Tumor-Detection

# 2. Create and Activate a Python Environment (Recommended)
conda create -n DL_env python=3.12
conda activate DL_env

# 3. Install Dependencies
pip install -r requirements.txt
```