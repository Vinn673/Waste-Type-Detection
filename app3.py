import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd

st.set_page_config(page_title="Waste Detection", layout="wide")

model = YOLO("best.pt") 

st.title("Waste Detection")

option = st.radio("Pilih Mode:", ["Webcam", "Upload Gambar"])


kategori_sampah = {
    "plastik": "Anorganik",
    "botol": "Anorganik",
    "kaca": "Anorganik",
    "glass": "Anorganik",       
    "logam": "Anorganik",
    "metal": "Anorganik", 
    "kertas": "Organik",
    "paper": "Organik",         
    "daun": "Organik"
}


st.markdown("""
<style>
.card {
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 12px;
    color: white;
    font-size: 18px;
}
.organik {
    background-color: #2ecc71;
}
.anorganik {
    background-color: #e74c3c;
}
.icon {
    font-size: 28px;
    margin-right: 10px;
}
</style>
""", unsafe_allow_html=True)


def tampilkan_kategori(results):
    detected_labels = set()

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = results[0].names[cls_id].lower()
        detected_labels.add(label)

    st.subheader("📌 Hasil Klasifikasi Sampah")

    if not detected_labels:
        st.write("Tidak ada objek terdeteksi.")
        return

  
    st.write("""
             Kartu Kategori Sampah
             🟩 Organik 
             🟥 Anorganik """)
    for label in detected_labels:
        kategori = kategori_sampah.get(label, "Tidak diketahui")

        css_class = "organik" if kategori == "Organik" else "anorganik"
        icon = "🟩" if kategori == "Organik" else "🟥"

        st.markdown(
            f"""
            <div class="card {css_class}">
                <span class="icon">{icon}</span>
                <b>{label.upper()}</b> → {kategori}
            </div>
            """,
            unsafe_allow_html=True
        )


    for label in detected_labels:
        kategori = kategori_sampah.get(label, "Tidak diketahui")
        icon = "🟩🗑 Organik" if kategori == "Organik" else "🟥🗑 Anorganik"
        st.write(f"- **{label}** → {icon}")




if option == "Webcam":
    camera_image = st.camera_input("Ambil foto menggunakan webcam")

    if camera_image:
        file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = model.predict(frame_rgb, conf=0.3)
        annotated = results[0].plot()

        st.image(annotated, caption="Hasil Deteksi")

        tampilkan_kategori(results)


else:
    uploaded = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        st.image(img_rgb, caption="Gambar Asli")

        results = model.predict(img_rgb, conf=0.5)
        annotated = results[0].plot()

        st.image(annotated, caption="Hasil Deteksi")

        tampilkan_kategori(results)
