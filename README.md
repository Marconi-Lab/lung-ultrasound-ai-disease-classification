This repository contains code for preprocessing and sorting lung ultrasound (LUS) image datasets and building image processing pipelines for automated classification of lung diseases using AI techniques.

### 🧠 Project Overview

Lung ultrasound (LUS) is a portable, cost-effective, and radiation-free tool for diagnosing lung conditions, ideal for low-resource and emergency settings. However, its interpretation is difficult due to variability and image quality. This project develops AI-based tools to automate and standardize LUS image analysis. It focuses on two key components:

- `sort_lus_data.py`: Organizes and structures raw ultrasound datasets for streamlined training and evaluation.
- `process_image.py`: Provides image preprocessing and transformation functions to prepare input data for machine learning pipelines.

### 🗂️ Dataset Description

This repository supports a curated benchmark dataset of 1,062 labelled lung ultrasound images collected in Uganda. The data were acquired from patients at **Mulago National Referral Hospital** and **Kiruddu Referral Hospital** by senior radiologists. Each image is annotated and suitable for training and evaluating deep learning models—particularly convolutional neural networks (CNNs).

The dataset is intended to support the development of robust, automated deep learning systems for pulmonary disease diagnosis using LUS. It offers valuable real-world data for advancing medical AI research, especially in under-resourced clinical environments.

### 📁 Repository Structure

```bash
lung-ultrasound-ai-disease-classification/
├── sort_lus_data.py       # Script to sort and structure raw lung ultrasound image data
├── process_image.py       # Image preprocessing functions
├── README.md              # Project documentation
```

# Getting Started
1. Clone the Repository
      ```bash
      git clone https://github.com/yourusername/lung-ultrasound-ai-disease-classification.git
      cd lung-ultrasound-ai-disease-classification
      ```

2. Organize Your Dataset
    > Ensure your raw LUS images are placed in the correct input folder, then run:
  
      ```bash
      python sort_lus_data.py
      ```
      This will sort and structure the dataset into appropriate subfolders for training and evaluation.

3. Preprocess the Images
    > Use the functions in process_image.py to apply preprocessing such as:
  
      - Grayscale conversion
      - Normalization
      - Resizing
      - Noise reduction
      - Data augmentation
  
      Example Usege:
      
      ```python
      from process_image import preprocess_image
      processed_img = preprocess_image("path/to/image.png")
      ```

4. Dependencies
    > Install required Python packages using:
      ```bash
      pip install numpy opencv-python pillow scikit-learn matplotlib
      ```

5. Model Training (Example)
    > Here is a simple PyTorch CNN architecture you can build upon:
      ```python
      import torch.nn as nn
      
      class SimpleCNN(nn.Module):
          def __init__(self):
              super(SimpleCNN, self).__init__()
              self.features = nn.Sequential(
                  nn.Conv2d(1, 32, kernel_size=3, stride=1),
                  nn.ReLU(),
                  nn.MaxPool2d(2),
                  nn.Conv2d(32, 64, kernel_size=3, stride=1),
                  nn.ReLU(),
                  nn.MaxPool2d(2)
              )
              self.classifier = nn.Sequential(
                  nn.Linear(64 * 6 * 6, 128),
                  nn.ReLU(),
                  nn.Linear(128, 3)  # Adjust based on number of output classes
              )
      
          def forward(self, x):
              x = self.features(x)
              x = x.view(x.size(0), -1)
              x = self.classifier(x)
              return x
      ```
6. Evaluation
    > Evaluate your classification results using metrics like accuracy, F1-score, and confusion matrix:
      ```python
      from sklearn.metrics import classification_report
      print(classification_report(y_true, y_pred, target_names=class_names))
      ```




