# Leaders_Image_Classification_Model
## 🚀 Overview
Recognizing historical figures in images is often hampered by low-resolution archives and limited datasets. This project achieves 91.67% accuracy by combining:

Classical Computer Vision: Manual data cleaning and robust feature extraction.

Generative AI: Synthetic data generation using LoRA to balance classes for historical figures.

Supervised Learning: An SVM classifier optimized for high-dimensional feature sets.

### 👥 Targeted Leaders
Dr. Abiy Ahmed (Current Prime Minister)

Hailemariam Desalegn (Former Prime Minister)

Meles Zenawi (Former Prime Minister)

Mengistu Haile Mariam (Former President)

Emperor Haile Selassie (Former Emperor)

### 🛠️ Technical Workflow
#### 1. Data Preprocessing & Cleaning
Face & Eye Detection: Uses OpenCV Haar Cascades to filter images. Only images where two eyes are clearly visible are retained to ensure feature quality.

Cropping & Grayscaling: Faces are automatically cropped and converted to grayscale to reduce computational complexity.

#### 2. Feature Engineering
We use a combined feature vector consisting of:

Histogram of Oriented Gradients (HOG): Captures the structural shape and edges of facial features.

Wavelet Transform: Extracts frequency-domain features to identify textures and fine details around the eyes and nose.

#### 3. Classification
Model: Support Vector Machine (SVM) with an RBF kernel.

Optimization: Hyperparameter tuning conducted via GridSearchCV to find the optimal C and gamma values.

Accuracy: 91.67%.
