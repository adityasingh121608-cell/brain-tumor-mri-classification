# Brain Tumor MRI Image Classification

## Project Overview
Deep learning-based medical imaging system to classify brain MRI scans into tumor categories (Glioma, Meningioma, Pituitary, No Tumor) using custom CNNs and Transfer Learning architectures.

## Features
- **Data Preprocessing & Augmentation:** Image normalization, resizing, rotation, and flips.
- **Deep Learning Architectures:** Custom CNN baseline + Transfer Learning models (ResNet50 / MobileNetV2).
- **Streamlit App:** Interactive web interface for uploading MRI images and obtaining real-time tumor predictions with confidence metrics.

## Quick Start
```bash
pip install -r requirements.txt
streamlit run app.py
