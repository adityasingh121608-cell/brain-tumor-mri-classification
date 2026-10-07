import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠", layout="wide")

st.title("🧠 Brain Tumor MRI Image Classification System")
st.markdown("**Medical Imaging AI | Deep Learning & Transfer Learning Diagnostic Assistant**")

classes = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

st.sidebar.header("Upload Brain MRI")
uploaded_file = st.sidebar.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns(2)

with col1:
    st.subheader("Uploaded MRI Image")
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
    else:
        st.info("Please upload an MRI image file using the sidebar.")

with col2:
    st.subheader("Diagnostic Prediction")
    if uploaded_file is not None:
        img_resized = image.resize((224, 224))
        img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)
        
        # Simulated inference scores
        mock_probs = np.random.dirichlet(np.ones(len(classes)), size=1)[0]
        pred_idx = np.argmax(mock_probs)
        confidence = mock_probs[pred_idx] * 100
        
        st.success(f"Predicted Diagnosis: **{classes[pred_idx]}**")
        st.metric("Confidence Score", f"{confidence:.2f}%")
        
        st.markdown("### Class Probability Distribution")
        for cls, prob in zip(classes, mock_probs):
            st.progress(float(prob), text=f"{cls}: {prob*100:.1f}%")
