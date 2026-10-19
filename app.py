"""Aplikasi Streamlit klasifikasi gambar CIFAR-10 dengan CNN."""
import io

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from inference import load_config, load_model, predict, resize_for_model

MAX_FILE_MB = 5
LOW_CONFIDENCE = 0.50

st.set_page_config(page_title="Klasifikasi Gambar CIFAR-10", page_icon="🖼️")

config = load_config()


@st.cache_resource
def get_model():
    return load_model(config)


# ---------- Sidebar ----------
with st.sidebar:
    st.header("Informasi Model")
    st.markdown(
        f"- Input: {config['input_size'][0]}×{config['input_size'][1]} RGB\n"
        f"- Normalisasi: piksel ÷ 255\n"
        f"- Jumlah kelas: {len(config['class_names'])}"
    )
    st.caption(config.get("model_note", ""))
    st.subheader("Kelas yang dikenali")
    st.write(", ".join(config["class_names"]))

# ---------- Header ----------
st.title("🖼️ Klasifikasi Gambar CIFAR-10")
st.write(
    "Unggah gambar (JPG/PNG) berisi salah satu dari 10 kelas CIFAR-10, "
    "lalu tekan **Prediksi**. Model CNN akan menebak kelas gambar beserta "
    "tingkat keyakinannya."
)

try:
    model = get_model()
except Exception as exc:  # file hilang, versi TF/Keras tidak cocok, dll.
    st.error(f"Model gagal dimuat: {exc}")
    st.stop()

uploaded = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Belum ada gambar diunggah.")
    st.stop()

# ---------- Validasi file ----------
data = uploaded.getvalue()
if len(data) == 0:
    st.error("File kosong. Silakan unggah gambar lain.")
    st.stop()
if len(data) > MAX_FILE_MB * 1024 * 1024:
    st.error(f"Ukuran file melebihi {MAX_FILE_MB} MB. Silakan kompres atau pilih gambar lain.")
    st.stop()
try:
    Image.open(io.BytesIO(data)).verify()  # deteksi file rusak
    image = Image.open(io.BytesIO(data))
    image.load()
except (UnidentifiedImageError, OSError, SyntaxError):
    st.error("File bukan gambar yang valid atau rusak. Gunakan JPG/PNG yang dapat dibuka.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    st.image(image, caption=f"Gambar asli ({image.width}×{image.height})", width="stretch")
with col2:
    w, h = config["input_size"]
    small = resize_for_model(image, config)
    st.image(
        small.resize((256, 256), Image.Resampling.NEAREST),
        caption=f"Yang dilihat model ({w}×{h})",
        width="stretch",
    )

# ---------- Prediksi ----------
if st.button("Prediksi", type="primary"):
    with st.spinner("Memprediksi..."):
        top3 = predict(model, image, config, top_k=3)

    label, conf = top3[0]
    st.subheader(f"Hasil: **{label}**")
    st.metric("Confidence", f"{conf:.1%}")

    if conf < LOW_CONFIDENCE:
        st.warning(
            f"Confidence di bawah {LOW_CONFIDENCE:.0%}. Model ragu; gambar mungkin "
            "bukan salah satu dari 10 kelas atau objek kurang jelas."
        )

    st.write("**Top-3 prediksi**")
    df = pd.DataFrame(top3, columns=["Kelas", "Probabilitas"]).set_index("Kelas")
    st.bar_chart(df, horizontal=True)
    st.dataframe(df.style.format({"Probabilitas": "{:.2%}"}))

    st.caption(
        "Catatan: probabilitas softmax yang tinggi tidak menjamin prediksi benar. "
        "Model hanya mengenal 10 kelas dan dilatih pada gambar beresolusi 32×32, "
        "sehingga gambar di luar kelas tetap akan dipaksa masuk ke salah satu kelas."
    )
