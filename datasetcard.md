# Dataset Card
---
# Dataset Card for Lung Cancer CT Scan Dataset

A comprehensive medical imaging dataset for lung cancer detection and classification using CT scan images.

## Dataset Details
### Dataset Description
This dataset contains CT scan images specifically curated for lung cancer detection and classification. It provides a carefully compiled collection of medical images representing different lung tissue conditions, enabling advanced machine learning research in medical image analysis.

- **Curated by:** Medical Imaging Research Team (Exact attribution needed)
- **License:** Research and Educational Use (Specific license details to be confirmed)

### Dataset Sources
- **Repository:** [HugginFace](https://huggingface.co/datasets/dorsar/lung-cancer)

## Uses
### Direct Use
This dataset is specifically designed for:
- Medical image classification research
- Development of AI-powered diagnostic support systems
- Machine learning model training for lung cancer detection
- Multi-class classification tasks

## Dataset Structure
The dataset is organized into three primary splits: train, test, and validation, with images categorized into four distinct classes:

1. **Adenocarcinoma**: 223 images (100 train, 100 test, 23 validation)
2. **Large Cell Carcinoma**: 172 images (100 train, 51 test, 21 validation)
3. **Normal Lung Tissue**: 167 images (100 train, 54 test, 13 validation)
4. **Squamous Cell Carcinoma**: 205 images (100 train, 90 test, 15 validation)

Total Images: 662

## Dataset Creation
### Source Data
#### Data Collection and Processing
- **Data Source:**[HugginFace](https://huggingface.co/datasets/dorsar/lung-cancer)
- **Image Modality:** CT Scans
- **Image Format:** PNG

#### Features and the Target
- **Input Feature:** CT Scan Images
- **Target Variable:** Lung Cancer Type Classification
  - Binary Classification: Cancerous vs. Non-Cancerous
  - Multi-Class Classification: Specific Cancer Type

## Bias, Risks, and Limitations
- Limited dataset size (662 total images)
- Potential geographical and demographic sampling bias
- Variation in imaging equipment and techniques
- Requires validation by medical professionals
- Potential underrepresentation of rare cancer subtypes


## Potential Machine Learning Approaches
- Convolutional Neural Networks (CNN)
- Transfer Learning
- Ensemble Methods
- Data Augmentation Techniques
