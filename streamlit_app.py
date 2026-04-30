import streamlit as st
import cv2
import numpy as np
import joblib
import pickle
import json
from PIL import Image
import os
import pywt

# Page configuration
st.set_page_config(
    page_title="Ethiopian Leaders Image Classification",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 30px;
    }
    .section-header {
        color: #ff7f0e;
        margin-top: 25px;
        margin-bottom: 15px;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 10px;
    }
    .leader-card {
        text-align: center;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .prediction-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 20px;
        border-radius: 5px;
        margin-top: 20px;
    }
    .prediction-text {
        font-size: 24px;
        font-weight: bold;
        color: #155724;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and class dictionary
def load_model_and_classes():
    """Load the trained model and class dictionary"""
    # Load model using joblib 
    model = joblib.load('saved_model.pkl')
    
    # Load class dictionary
    with open('class_dictionary.json', 'r', encoding='utf-8') as f:
        class_dict = json.load(f)
    
    # Create reverse mapping (index to class name)
    reverse_class_dict = {v: k for k, v in class_dict.items()}
    
    return model, class_dict, reverse_class_dict

# Initialize Haar Cascades for face and eye detection
@st.cache_resource
def load_cascades():
    """Load Haar Cascade classifiers for face and eye detection"""
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_eye.xml'
    )
    return face_cascade, eye_cascade

# Function to get cropped image if 2 eyes are detected
def get_cropped_image_if_2_eyes(image_path, face_cascade, eye_cascade):
    """Extract face region if at least 2 eyes are detected"""
    img = cv2.imread(image_path)
    if img is None:
        return None, "Could not read image"
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return None, "No face detected in the image"
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        if len(eyes) >= 2:
            return roi_color, "Success"
    
    return None, "Could not detect 2 eyes in any face"

# Wavelet transformation function (matches training preprocessing)
def w2d(img, mode='db1', level=5):
    """Apply 2D wavelet decomposition and return reconstructed Haar features"""
    if len(img.shape) == 3:
        imArray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        imArray = img
    imArray = np.float32(imArray)
    imArray /= 255.0
    coeffs = pywt.wavedec2(imArray, mode, level=level)
    # Zero out the approximation coefficients (matches training)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)
    return imArray_H

# Function to predict leader from image
def predict_leader(image, model, reverse_class_dict):
    """Predict the leader from the image - matches training preprocessing"""
    try:
        # Get raw RGB features: resize original to 32x32
        img_resized = cv2.resize(image, (32, 32))
        raw_features = img_resized.reshape(32*32*3).astype(np.float32)
        
        # Get wavelet Haar features: apply w2d to ORIGINAL image (like training), then resize
        img_har = w2d(image, 'db1', 5)
        img_har_resized = cv2.resize(img_har, (32, 32))
        haar_features = img_har_resized.reshape(32*32).astype(np.float32)
        
        # Combine features: vstack then flatten (3072 + 1024 = 4096 total)
        combined_features = np.vstack((raw_features.reshape(-1, 1), haar_features.reshape(-1, 1))).flatten()
        
        # Make prediction - reshape to 2D array for sklearn models
        prediction = model.predict(combined_features.reshape(1, -1))
        
        try:
            # Get probability if available
            confidence_proba = model.predict_proba(combined_features.reshape(1, -1))
            max_confidence = np.max(confidence_proba) * 100  # Convert to percentage
        except:
            max_confidence = 100.0
        
        predicted_class = int(prediction[0])
        predicted_name = reverse_class_dict.get(predicted_class, "Unknown")
        
        return predicted_name, max_confidence
    except Exception as e:
        return None, str(e)

# Load model and cascades
try:
    model, class_dict, reverse_class_dict = load_model_and_classes()
    st.toast("Model loaded successfully", icon="✅")
except Exception as e:
    st.error(f"Failed to load model or class dictionary: {e}")
    st.stop()

face_cascade, eye_cascade = load_cascades()

# Main header
st.markdown("<h1 class='main-header'>🎭 Ethiopian Leaders Image Classification Model</h1>", unsafe_allow_html=True)
st.markdown("Upload a photo of a leader to identify who they are!", unsafe_allow_html=True)

# Display reference images
st.markdown("<h2 class='section-header'>📷 Reference Leaders</h2>", unsafe_allow_html=True)
st.markdown("Here are sample images of the leaders the model can recognize:")

# Create columns for displaying reference images
col1, col2, col3, col4, col5 = st.columns(5)

reference_images = {
    "         Dr. Abiy Ahmed - Current Prime Minister \n [2 April 2018 - present] ": "images/abiy.png",
    " Emperor Haileslasie - Former Emperor \n [12 September 1974 - 2 November 1930]": "images/janhoy.png",
    " Hailemariam Desalegn - Former Prime Minister \n [20 August 2012 – 2 April 2018]": "images/hailemariam.png",
    "         Meles Zenawi - Former Prime Minister  \n [23 August 1995 – 20 August 2012]": "images/meles.png",
    " Mengistu Hailemariam - Former President \n [10 September 1987 – 21 May 1991]": "images/mengistu.png"
}

cols = [col1, col2, col3, col4, col5]
for idx, (leader_name, image_path) in enumerate(reference_images.items()):
    if os.path.exists(image_path):
        with cols[idx]:
            st.markdown(f"<div class='leader-card'>", unsafe_allow_html=True)
            try:
                img = Image.open(image_path)
                st.image(img, width=250)
                st.markdown(f"<p style='text-align: center; font-weight: bold;'>{leader_name}</p>", unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Could not load image for {leader_name}")
            st.markdown(f"</div>", unsafe_allow_html=True)
    else:
        with cols[idx]:
            st.markdown(f"<div class='leader-card'>", unsafe_allow_html=True)
            st.info(f"Image not found\n\n{leader_name}")
            st.markdown(f"</div>", unsafe_allow_html=True)

# Divider
st.markdown("---")

# Upload section
st.markdown("<h2 class='section-header'>📤 Upload Your Image</h2>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose an image file (JPG, PNG, etc.)",
    type=["jpg", "jpeg", "png", "bmp"],
    help="Upload a clear photo of a leader's face"
)

if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_image_path = "temp_uploaded_image.jpg"
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display uploaded image
    st.markdown("<h3 style='color: #ff7f0e;'>📸 Uploaded Image</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_img = Image.open(uploaded_file)
        st.image(uploaded_img, caption="Your Uploaded Image", width=300)
    
    # Process image and show detected face
    with col2:
        with st.spinner("Processing image..."):
            cropped_image = get_cropped_image_if_2_eyes(temp_image_path, face_cascade, eye_cascade)
            
            if cropped_image[0] is not None:
                cropped_img_pil = Image.fromarray(cv2.cvtColor(cropped_image[0], cv2.COLOR_BGR2RGB))
                st.image(cropped_img_pil, caption="Detected Face", width=300)
            else:
                st.warning(f"⚠️ {cropped_image[1]}")
                st.info("Please try uploading a clearer image with a visible face and both eyes.")
    
    # Prediction button
    st.markdown("<h3 style='color: #ff7f0e;'>🔮 Make Prediction</h3>", unsafe_allow_html=True)
    
    if st.button("🎯 Predict Leader", use_container_width=True, type="primary"):
        cropped_image_data, status = get_cropped_image_if_2_eyes(temp_image_path, face_cascade, eye_cascade)
        
        if cropped_image_data is not None:
            with st.spinner("Making prediction..."):
                predicted_leader, confidence = predict_leader(
                    cropped_image_data, 
                    model, 
                    reverse_class_dict
                )
                
                if predicted_leader is not None:
                    if confidence > 75:
                        st.markdown(f"<div class='prediction-box'>", unsafe_allow_html=True)
                        st.markdown(f"<div class='prediction-text'>✅ Predicted: {predicted_leader}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div style='font-size: 18px; color: #155724;'>Confidence: {confidence:.2f}%</div>", unsafe_allow_html=True)
                        st.markdown(f"</div>", unsafe_allow_html=True)
                    else:
                        st.warning(f"⚠️ Low Confidence Detection")
                        st.markdown(f"<div style='background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 20px; border-radius: 5px; margin-top: 20px;'>", unsafe_allow_html=True)
                        st.markdown(f"<p style='font-size: 18px; color: #856404;'>Model confidence: {confidence:.2f}% (Threshold: >75%)</p>", unsafe_allow_html=True)
                        st.markdown(f"<p style='color: #856404;'>The person in this image may not be one of the trained leaders.</p>", unsafe_allow_html=True)
                        st.markdown(f"</div>", unsafe_allow_html=True)
                else:
                    st.error(f"Prediction error: {confidence}")
        else:
            st.error(f"❌ {status}")
            st.info("The image needs to show a clear face with both eyes visible. Please try another image.")
    
    # Cleanup temporary file
    if os.path.exists(temp_image_path):
        os.remove(temp_image_path)

else:
    st.info("👆 Start by uploading an image of a leader to get a prediction!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; margin-top: 30px;'>
    <p><strong>Leaders Image Classification Model</strong></p>
    <p>Built with Streamlit | Uses OpenCV for Face Detection | Trained ML Model for Classification</p>
</div>
""", unsafe_allow_html=True)
