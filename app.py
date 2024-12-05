import streamlit as st
import cv2
import numpy as np
from PIL import Image

def process_image(uploaded_file):
    # Membaca gambar
    image = np.array(Image.open(uploaded_file))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Threshold untuk mendeteksi objek (bounding box)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Cari kontur (boundary objek)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Gambar bounding box dan simpan koordinat
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Gambar bounding box pada gambar
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Simpan koordinat
        bounding_boxes.append((x, y, w, h))
    
    return image, bounding_boxes

# Antarmuka Streamlit
st.title(" Bounding Box detector")
uploaded_file = st.file_uploader("Upload", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Tampilkan gambar asli
    st.image(uploaded_file, caption="picture", use_column_width=True)

    # Proses gambar
    processed_image, bounding_boxes = process_image(uploaded_file)

    # Tampilkan hasil
    st.image(processed_image, caption="result", use_column_width=True)

    # Tampilkan koordinat bounding box
    if bounding_boxes:
        st.write("Bounding Bo Coordinate x:")
        for i, (x, y, w, h) in enumerate(bounding_boxes):
            st.write(f"Objek {i + 1}: x={x}, y={y}, width={w}, height={h}")
    else:
        st.write("No Onject Detected ")
