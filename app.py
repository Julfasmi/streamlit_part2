import streamlit as st
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8s model (ganti path ke model kamu)
@st.cache_resource
def load_model():
    model = YOLO("likecommentmodel.pt")  # model hasil training kamu
    return model

model = load_model()

st.title("Like & Comment Detection (YOLOv8s)")

st.markdown(
    """
    ### 📌 How to Use:
    To try out the model, **upload images in the form of screenshots from various social media platforms** 
    such as **Facebook, YouTube, or TikTok**.  
    The model will detect the *Like* and *Comment* elements in the image.
    """
)

# Upload gambar
uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert ke format numpy array untuk YOLOv8
    img_array = np.array(image)

    # Jalankan deteksi
    results = model.predict(img_array)

    # Ambil hasil prediksi dan bounding box
    annotated_img = results[0].plot()  # YOLOv8 langsung bisa plot hasilnya

    st.image(annotated_img, caption="Detection Results", use_column_width=True)
