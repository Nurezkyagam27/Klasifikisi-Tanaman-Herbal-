import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pandas as pd

# ==================== KONFIGURASI ====================
st.set_page_config(
    page_title="Klasifikasi Daun Herbal",
    page_icon="🌿",
    layout="wide"
)

CONFIDENCE_THRESHOLD = 70.0
ENTROPY_THRESHOLD = 1.5
UNKNOWN_LABEL = "Tidak Diketahui"

# ==================== CLASS NAMES ====================
CLASS_NAMES = [
    "Belimbing Wuluh",
    "Jambu Biji",
    "Jeruk Nipis",
    "Kemangi",
    "Lidah Buaya",
    "Nangka",
    "Pandan",
    "Pepaya",
    "Seledri",
    "Sirih"
]

# ==================== LOAD MODEL ====================
@st.cache_resource
def load_trained_model():
    try:
        return load_model("daun-herbal.keras")
    except:
        return load_model("daun-herbal.h5")

# ==================== PREPROCESS ====================
def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image)

    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]

    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ==================== PREDICT ====================
def predict_image(model, image):
    processed = preprocess_image(image)
    preds = model.predict(processed, verbose=0)[0]

    idx = np.argmax(preds)
    confidence = preds[idx] * 100

    entropy = -np.sum(preds * np.log(preds + 1e-8))

    if confidence < CONFIDENCE_THRESHOLD or entropy > ENTROPY_THRESHOLD:
        return UNKNOWN_LABEL, confidence, preds

    return CLASS_NAMES[idx], confidence, preds

# ==================== SHOW RESULT ====================
def show_prediction_result(predicted_class, confidence, all_predictions, image):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼️ Gambar Input")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🎯 Hasil Prediksi")

        if predicted_class == UNKNOWN_LABEL:
            st.markdown("### ❓ Tidak Diketahui")
            st.error("⚠️ Objek tidak termasuk daun herbal dalam dataset")
            st.metric("Confidence Tertinggi", f"{confidence:.2f}%")
            st.info(
                "Sistem menerapkan mekanisme penolakan prediksi "
                "untuk citra di luar domain pelatihan (out-of-distribution)."
            )
            return

        st.markdown(f"### 🌿 {predicted_class}")
        st.metric("Tingkat Keyakinan", f"{confidence:.2f}%")
        st.progress(confidence / 100)

        if confidence >= 80:
            st.success("✅ Prediksi sangat yakin")
        elif confidence >= 50:
            st.warning("⚠️ Prediksi cukup yakin")
        else:
            st.error("❌ Prediksi kurang yakin")

    st.markdown("---")

    st.subheader("📊 Probabilitas Semua Kelas")
    prob_df = pd.DataFrame({
        "Tanaman": CLASS_NAMES,
        "Probabilitas (%)": [p * 100 for p in all_predictions]
    }).sort_values("Probabilitas (%)")

    st.bar_chart(prob_df.set_index("Tanaman"))

# ==================== SIDEBAR ====================
with st.sidebar:
    st.title("🌿 Herbal Classifier")
    menu = st.radio("Menu", ["Home", "Predict", "About"])

# ==================== HOME ====================
if menu == "Home":
    st.title("🌿 Klasifikasi Daun Herbal")
    st.markdown("""
    Aplikasi ini menggunakan **MobileNetV2** untuk mengklasifikasikan
    **10 jenis daun herbal Indonesia**.

    ✅ Mendeteksi objek di luar dataset  
    ✅ Tidak memaksa prediksi  
    ✅ Aman untuk penggunaan akademik
    """)

# ==================== PREDICT ====================
elif menu == "Predict":
    st.title("🔍 Prediksi Daun Herbal")

    with st.spinner("Memuat model..."):
        model = load_trained_model()

    tab1, tab2 = st.tabs(["📤 Upload Gambar", "📷 Kamera"])

    with tab1:
        uploaded_file = st.file_uploader(
            "Upload gambar",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            with st.spinner("Menganalisis gambar..."):
                pred, conf, preds = predict_image(model, image)
            show_prediction_result(pred, conf, preds, image)

    with tab2:
        cam = st.camera_input("Ambil foto")

        if cam:
            image = Image.open(cam).convert("RGB")
            with st.spinner("Menganalisis gambar..."):
                pred, conf, preds = predict_image(model, image)
            show_prediction_result(pred, conf, preds, image)

# ==================== ABOUT ====================
elif menu == "About":
    st.title("📚 Tentang Aplikasi")
    st.markdown("""
    **Model** : MobileNetV2  
    **Input** : 224×224 RGB  
    **Jumlah Kelas** : 10  

    🔍 Sistem menggunakan **confidence + entropy**
    untuk mendeteksi citra **di luar dataset**.
    """)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:gray;'>© 2025 - Herbal Leaf Classifier</p>",
    unsafe_allow_html=True
)
