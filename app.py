
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Pengaturan halaman
st.set_page_config(
    page_title="Intel Image Classification",
    page_icon="🌄",
    layout="centered"
)

# Judul website
st.title("🌄 Intel Image Classification")
st.write(
    "Klasifikasi gambar menggunakan "
    "Convolutional Neural Network (CNN)."
)

# Daftar kelas sesuai urutan training
nama_kelas = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street"
]

# Memuat model CNN
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "intel_cnn_model.keras"
    )

model = load_model()

# Upload gambar
st.subheader("Upload Gambar")

file_gambar = st.file_uploader(
    "Pilih gambar yang ingin diprediksi",
    type=["jpg", "jpeg", "png"]
)

if file_gambar is not None:

    # Membaca gambar
    gambar = Image.open(file_gambar).convert("RGB")

    # Menampilkan gambar
    st.image(
        gambar,
        caption="Gambar yang diunggah",
        use_container_width=True
    )

    # Mengubah ukuran gambar
    gambar = gambar.resize((150, 150))

    # Mengubah gambar menjadi array
    gambar_array = np.array(
        gambar,
        dtype=np.float32
    )

    # Menambahkan dimensi batch
    gambar_array = np.expand_dims(
        gambar_array,
        axis=0
    )

    # Melakukan prediksi
    hasil = model.predict(
        gambar_array,
        verbose=0
    )

    # Mengambil kelas dengan probabilitas tertinggi
    indeks = np.argmax(hasil[0])

    prediksi = nama_kelas[indeks]
    confidence = float(hasil[0][indeks]) * 100

    # Menampilkan hasil prediksi
    st.subheader("Hasil Prediksi")

    st.success(
        f"Kelas gambar: {prediksi.upper()}"
    )

    st.metric(
        label="Confidence",
        value=f"{confidence:.2f}%"
    )

    # Menampilkan probabilitas semua kelas
    st.subheader("Probabilitas Setiap Kelas")

    for i in range(len(nama_kelas)):

        nilai = float(hasil[0][i])

        st.write(
            f"{nama_kelas[i].capitalize()}: "
            f"{nilai * 100:.2f}%"
        )

        st.progress(nilai)
