import streamlit as st
import cv2
import numpy as np
import joblib
import pickle
import json
from PIL import Image
import os

# Page configuration
st.set_page_config(
    page_title="Leaders Image Classification",
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
@st.cache_resource
def load_model_and_classes():
    """Load the trained model and class dictionary"""
    model = None
    
    # Try loading with joblib first (recommended for sklearn models)
    try:
        model = joblib.load('saved_model.pkl')
        st.toast("✓ Model loaded successfully with joblib", icon="✓")
    except Exception as e1:
        st.warning(f"Could not load with joblib: {e1}")
        
        # Fallback: Try loading with pickle
        try:
            with open('saved_model.pkl', 'rb') as f:
                model = pickle.load(f)
            st.toast("✓ Model loaded successfully with pickle", icon="✓")
        except Exception as e2:
            error_msg = f"Error loading model - joblib: {e1} | pickle: {e2}"
            st.error(error_msg)
            return None, None, None
    
    # Load class dictionary
    try:
        with open('class_dictionary.json', 'r') as f:
            class_dict = json.load(f)
        
        # Create reverse mapping (index to class name)
        reverse_class_dict = {v: k for k, v in class_dict.items()}
        
        return model, class_dict, reverse_class_dict
    except Exception as e:
        st.error(f"Error loading class dictionary: {e}")
        return model, None, None

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

# Function to predict leader from image
def predict_leader(image, model, reverse_class_dict):
    """Predict the leader from the image"""
    try:
        # Resize image to match model input
        image_resized = cv2.resize(image, (128, 128))  # Adjust size based on your model
        
        # Normalize the image
        image_normalized = image_resized / 255.0
        
        # Flatten the image
        image_flattened = image_normalized.flatten()
        
        # Make prediction - reshape to 2D array for sklearn models
        prediction = model.predict(image_flattened.reshape(1, -1))
        
        try:
            # Try to get probability if available
            confidence_proba = model.predict_proba(image_flattened.reshape(1, -1))
            max_confidence = np.max(confidence_proba)
        except:
            # If predict_proba not available, use confidence of 1.0
            max_confidence = 1.0
        
        predicted_class = prediction[0]
        predicted_name = reverse_class_dict.get(int(predicted_class), "Unknown")
        
        return predicted_name, max_confidence
    except Exception as e:
        return None, str(e)

# Load model and cascades
model, class_dict, reverse_class_dict = load_model_and_classes()
face_cascade, eye_cascade = load_cascades()

if model is None or class_dict is None:
    st.error("Failed to load model or class dictionary. Please check the files.")
    st.stop()

# Main header
st.markdown("<h1 class='main-header'>🎭 Leaders Image Classification Model</h1>", unsafe_allow_html=True)
st.markdown("Upload a photo of a leader to identify who they are!", unsafe_allow_html=True)

# Display reference images
st.markdown("<h2 class='section-header'>📷 Reference Leaders</h2>", unsafe_allow_html=True)
st.markdown("Here are sample images of the leaders the model can recognize:")

# Create columns for displaying reference images
col1, col2, col3, col4, col5 = st.columns(5)

reference_images = {
    "Dr. Abiy Ahmed": "images/abiy.png",
    "Emperor Haileslasie": "images/janhoy.png",
    "Hailemariam Desalegn": "images/hailemariam.png",
    "Meles Zenawi": "images/meles.png",
    "Mengistu Hailemariam": "images/mengistu.png"
}

cols = [col1, col2, col3, col4, col5]
for idx, (leader_name, image_path) in enumerate(reference_images.items()):
    if os.path.exists(image_path):
        with cols[idx]:
            st.markdown(f"<div class='leader-card'>", unsafe_allow_html=True)
            try:
                img = Image.open(image_path)
                st.image(img, use_column_width=True)
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
        st.image(uploaded_img, caption="Your Uploaded Image", use_column_width=True)
    
    # Process image and show detected face
    with col2:
        with st.spinner("Processing image..."):
            cropped_image = get_cropped_image_if_2_eyes(temp_image_path, face_cascade, eye_cascade)
            
            if cropped_image[0] is not None:
                cropped_img_pil = Image.fromarray(cv2.cvtColor(cropped_image[0], cv2.COLOR_BGR2RGB))
                st.image(cropped_img_pil, caption="Detected Face", use_column_width=True)
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
                    st.markdown(f"<div class='prediction-box'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='prediction-text'>✅ Predicted: {predicted_leader}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='font-size: 18px; color: #155724;'>Confidence: {confidence:.2%}</div>", unsafe_allow_html=True)
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
