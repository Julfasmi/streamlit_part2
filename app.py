import streamlit as st
from PIL import Image
# import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8s model (ganti path ke model kamu)
@st.cache_resource
def load_model():
    model = YOLO("likecommentmodel.pt")  # model hasil training kamu
    return model

model = load_model()

st.title("Deteksi Tombol Like & Komen (YOLOv8s)")

# Upload gambar
uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar asli", use_column_width=True)

    # Convert ke format numpy array untuk YOLOv8
    img_array = np.array(image)

    # Jalankan deteksi
    results = model.predict(img_array)

    # Ambil hasil prediksi dan bounding box
    annotated_img = results[0].plot()  # YOLOv8 langsung bisa plot hasilnya

    st.image(annotated_img, caption="Hasil Deteksi", use_column_width=True)
