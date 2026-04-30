# Leaders_Image_Classification_Model
## 🚀 Overview
Recognizing historical figures in images is often hampered by low-resolution archives and limited datasets. This project achieves 91.67% accuracy by combining:

Classical Computer Vision: Manual data cleaning and robust feature extraction.

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

---

## 🌐 Streamlit Web Application

### Features
✅ **Reference Images**: Browse sample photos of all 5 leaders to understand what the model recognizes.

✅ **Image Upload**: Upload any JPG, PNG, or BMP image of a leader.

✅ **Face Detection**: Automatic face and eye detection to extract the relevant region.

✅ **Real-time Prediction**: Instant classification with confidence scores.

✅ **User-Friendly Interface**: Clean, modern UI built with Streamlit.

### Setup & Installation
1. **Installing required libraries**
    - [pip install : "Numpy", 'pandas', 'matplotlib'
        ,'streamlit', 'openCV', 'scikit-learn', 'seaborn', 'pywt']
2. **Verify Required Files**
   Ensure you have the following files in the project directory:
   - `saved_model.pkl` - Trained model
   - `class_dictionary.json` - Class mappings
   - `images/` - Directory with reference images
   - `streamlit_app.py` - Streamlit application

3. **Run the Application**
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Access the Web App**
   - The app will open in your default browser at `http://localhost:8501`
   - Or navigate manually to the URL shown in the terminal

### How to Use

1. **View Reference Images**: See sample photos of all 5 leaders at the top of the page.

2. **Upload an Image**: Click the upload button and select an image containing a leader's face.

3. **View Detected Face**: The app automatically detects and displays the cropped face region.

4. **Get Prediction**: Click the "🎯 Predict Leader" button to classify the image.

5. **View Results**: The predicted leader name and confidence score will be displayed.

### Supported Leaders

- Dr. Abiy Ahmed
- Emperor Haile Selassie
- Hailemariam Desalegn
- Meles Zenawi
- Mengistu Haile Mariam

### Requirements

- Python 3.7+
- See `requirements.txt` for all dependencies

### File Structure

```
📁 Leaders_Image_Classification_Model/
├── 📄 streamlit_app.py          # Main Streamlit application
├── 📄 saved_model.pkl            # Trained SVM model
├── 📄 class_dictionary.json       # Class label mappings
├── 📁 images/                     # Reference images of leaders
│   ├── abiy.png
│   ├── janhoy.png
│   ├── hailemariam.png
│   ├── meles.png
│   └── mengistu.png
|── 📄requirements.txt             # required libraries for deploy
├── 📄 Cleaning & Modeling.ipynb   # Model training notebook
├── 📄 README.md                   # This file
└── 📁 dataset/                    # Training dataset
```
