import streamlit as st
import cv2
import numpy as np
from PIL import Image
import imutils

def preprocess_image(image):
    """
    Praproses gambar untuk meningkatkan deteksi bounding box
    """
    # Konversi ke grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Reduksi noise
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Threshold adaptif
    gray = cv2.adaptiveThreshold(
        gray, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    return gray

def detect_advanced_bounding_boxes(image):
    """
    Deteksi bounding box dengan metode yang lebih canggih
    """
    # Praproses gambar
    processed = preprocess_image(image)
    
    # Deteksi tepi dengan Canny
    edges = cv2.Canny(processed, 30, 200)
    
    # Dilasi untuk menghubungkan tepi yang terputus
    dilated = cv2.dilate(edges, None, iterations=2)
    
    # Temukan kontur
    contours = cv2.findContours(
        dilated.copy(), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    contours = imutils.grab_contours(contours)
    
    # Filter dan simpan bounding box
    bounding_boxes = []
    for contour in contours:
        # Abaikan kontur yang terlalu kecil
        area = cv2.contourArea(contour)
        if area < 100:  # Sesuaikan threshold ukuran
            continue
        
        # Dapatkan bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter rasio aspek untuk menghindari noise
        aspect_ratio = w / float(h)
        if 0.2 <= aspect_ratio <= 5.0:
            bounding_boxes.append((x, y, w, h))
    
    return bounding_boxes

def analyze_bounding_boxes(image, bounding_boxes):
    """
    Analisis tambahan untuk bounding box
    """
    analysis = []
    for x, y, w, h in bounding_boxes:
        # Ekstrak region of interest (ROI)
        roi = image[y:y+h, x:x+w]
        
        # Hitung rata-rata warna
        avg_color = np.mean(roi, axis=(0,1))
        
        # Hitung kontras
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        contrast = gray_roi.std()
        
        analysis.append({
            'coordinates': (x, y, w, h),
            'avg_color': avg_color,
            'contrast': contrast
        })
    
    return analysis

def main():
    st.set_page_config(
        page_title="Bounding Box Analyzer", 
        page_icon=":mag:"
    )
    
    st.title('🔍 Bounding Box Koordinat & Analisis')
    
    # Sidebar untuk pengaturan
    st.sidebar.header('Setting')
    min_box_size = st.sidebar.slider(
        'Minimum Bounding Box Size', 
        10, 200, 50
    )
    
    # Upload gambar
    uploaded_file = st.file_uploader(
        "Upload Picture", 
        type=['jpg', 'png', 'jpeg']
    )
    
    if uploaded_file is not None:
        # Baca gambar
        image = cv2.imdecode(
            np.frombuffer(uploaded_file.read(), np.uint8), 
            cv2.IMREAD_COLOR
        )
        
        # Tampilkan gambar asli
        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Real Picture')
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Deteksi bounding box
        bounding_boxes = detect_advanced_bounding_boxes(image)
        
        # Filter berdasarkan ukuran minimum
        bounding_boxes = [
            box for box in bounding_boxes 
            if box[2] >= min_box_size and box[3] >= min_box_size
        ]
        
        # Analisis bounding box
        box_analysis = analyze_bounding_boxes(image, bounding_boxes)
        
        # Tampilkan hasil
        with col2:
            st.subheader('Bounding Box Detected')
            result_image = image.copy()
            for (x, y, w, h) in bounding_boxes:
                cv2.rectangle(
                    result_image, 
                    (x, y), 
                    (x+w, y+h), 
                    (0, 255, 0), 
                    2
                )
            st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        
        # Tampilkan detail bounding box
        st.subheader('📊 Detail Bounding Box')
        for i, analysis in enumerate(box_analysis, 1):
            with st.expander(f'Bounding Box {i}'):
                coords = analysis['coordinates']
                st.write(f"Coordinate: X={coords[0]}, Y={coords[1]}")
                st.write(f"Wide: {coords[2]}, Height: {coords[3]}")
                st.write(f"Avg_Color: {analysis['avg_color']}")
                st.write(f"Contrast: {analysis['contrast']:.2f}")

if __name__ == "__main__":
    main()
