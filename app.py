import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pandas as pd

# Konfigurasi halaman
st.set_page_config(
    page_title="Klasifikasi Daun Herbal",
    page_icon="🌿",
    layout="wide"
)

# Daftar nama kelas (sesuai urutan folder di dataset)
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

# Informasi lengkap tanaman herbal (manfaat dan cara penggunaan)
INFO_HERBAL = {
    "Belimbing Wuluh": {
        "emoji": "🍋",
        "nama_latin": "Averrhoa bilimbi",
        "manfaat": [
            "Mengobati batuk dan meredakan tenggorokan",
            "Menurunkan tekanan darah tinggi",
            "Mengatasi jerawat dan masalah kulit",
            "Meredakan gusi berdarah",
            "Mengatasi sariawan",
            "Menurunkan kadar kolesterol"
        ],
        "cara_penggunaan": [
            "**Untuk batuk**: Rebus 3-5 buah belimbing wuluh dengan segelas air, tambahkan gula batu, minum 2x sehari",
            "**Untuk jerawat**: Haluskan buah, oleskan pada jerawat, diamkan 15-30 menit lalu bilas",
            "**Untuk tekanan darah**: Rebus daun muda (10-15 lembar) dengan 3 gelas air hingga tersisa 1 gelas, minum rutin",
            "**Untuk gusi berdarah**: Makan langsung buah segar atau kumur dengan air rebusan daun"
        ]
    },
    "Jambu Biji": {
        "emoji": "🍈",
        "nama_latin": "Psidium guajava",
        "manfaat": [
            "Meningkatkan sistem imunitas tubuh",
            "Mengobati diare dan gangguan pencernaan",
            "Kaya vitamin C untuk kesehatan kulit",
            "Menjaga kesehatan jantung",
            "Menurunkan kadar gula darah",
            "Membantu meningkatkan trombosit"
        ],
        "cara_penggunaan": [
            "**Untuk diare**: Rebus 5-7 lembar daun muda dengan 2 gelas air, minum 2x sehari",
            "**Untuk meningkatkan trombosit**: Minum jus jambu biji merah 2-3 gelas per hari",
            "**Untuk imunitas**: Konsumsi buah segar secara rutin atau jus tanpa gula",
            "**Untuk luka**: Tumbuk daun muda, tempelkan pada luka sebagai antiseptik alami"
        ]
    },
    "Jeruk Nipis": {
        "emoji": "🍋",
        "nama_latin": "Citrus aurantifolia",
        "manfaat": [
            "Meredakan batuk dan flu",
            "Membantu menurunkan berat badan",
            "Mencerahkan dan menyehatkan kulit",
            "Melancarkan sistem pencernaan",
            "Menghilangkan bau badan",
            "Menjaga kesehatan mulut dan gigi"
        ],
        "cara_penggunaan": [
            "**Untuk batuk**: Campur perasan jeruk nipis dengan madu dan air hangat, minum 2-3x sehari",
            "**Untuk diet**: Minum air perasan jeruk nipis dengan air hangat setiap pagi sebelum makan",
            "**Untuk kulit**: Oleskan perasan pada wajah, diamkan 10-15 menit, bilas bersih",
            "**Untuk pencernaan**: Minum air jeruk nipis hangat setelah makan berat"
        ]
    },
    "Kemangi": {
        "emoji": "🌿",
        "nama_latin": "Ocimum basilicum",
        "manfaat": [
            "Mengatasi bau badan tidak sedap",
            "Meningkatkan nafsu makan",
            "Melancarkan produksi ASI",
            "Meredakan stres dan kecemasan",
            "Menjaga kesehatan jantung",
            "Memiliki sifat anti-inflamasi"
        ],
        "cara_penggunaan": [
            "**Untuk bau badan**: Konsumsi daun kemangi segar sebagai lalapan setiap hari",
            "**Untuk ASI**: Makan daun kemangi segar secara rutin atau seduh sebagai teh",
            "**Untuk stres**: Seduh daun kemangi kering dengan air panas, minum seperti teh",
            "**Untuk nafsu makan**: Tambahkan sebagai bumbu atau lalapan pada makanan"
        ]
    },
    "Lidah Buaya": {
        "emoji": "🌵",
        "nama_latin": "Aloe vera",
        "manfaat": [
            "Menyembuhkan luka bakar dan iritasi kulit",
            "Melembabkan kulit secara alami",
            "Menyehatkan dan menguatkan rambut",
            "Melancarkan sistem pencernaan",
            "Membantu menurunkan gula darah",
            "Mengandung antioksidan tinggi"
        ],
        "cara_penggunaan": [
            "**Untuk luka bakar**: Belah daun, oleskan gel langsung pada area luka, ulangi 2-3x sehari",
            "**Untuk rambut**: Oleskan gel pada rambut dan kulit kepala, diamkan 30 menit, bilas",
            "**Untuk pencernaan**: Blender gel dengan air atau jus buah, minum di pagi hari",
            "**Untuk kulit wajah**: Gunakan gel sebagai masker, diamkan 15-20 menit, bilas"
        ]
    },
    "Nangka": {
        "emoji": "🍈",
        "nama_latin": "Artocarpus heterophyllus",
        "manfaat": [
            "Meningkatkan sistem kekebalan tubuh",
            "Menjaga kesehatan mata",
            "Sumber energi yang baik",
            "Melancarkan sistem pencernaan",
            "Menjaga kesehatan tulang",
            "Mengandung antioksidan untuk kesehatan kulit"
        ],
        "cara_penggunaan": [
            "**Untuk energi**: Konsumsi buah nangka matang sebagai camilan sehat",
            "**Untuk pencernaan**: Makan buah segar yang mengandung serat tinggi",
            "**Untuk kesehatan mata**: Konsumsi rutin karena kaya vitamin A",
            "**Untuk imunitas**: Rebus daun nangka muda, minum air rebusannya"
        ]
    },
    "Pandan": {
        "emoji": "🌿",
        "nama_latin": "Pandanus amaryllifolius",
        "manfaat": [
            "Mengatasi rematik dan nyeri sendi",
            "Menurunkan tekanan darah tinggi",
            "Meredakan sakit kepala",
            "Sebagai pewangi alami makanan",
            "Mengatasi ketombe",
            "Meredakan kecemasan dan insomnia"
        ],
        "cara_penggunaan": [
            "**Untuk rematik**: Remas daun pandan dengan minyak kelapa, oleskan pada sendi yang sakit",
            "**Untuk tekanan darah**: Rebus 3-5 lembar daun dengan 3 gelas air, minum 2x sehari",
            "**Untuk ketombe**: Haluskan daun, oleskan pada kulit kepala, diamkan 30 menit, bilas",
            "**Untuk insomnia**: Seduh daun pandan dengan air panas, minum sebelum tidur"
        ]
    },
    "Pepaya": {
        "emoji": "🍈",
        "nama_latin": "Carica papaya",
        "manfaat": [
            "Melancarkan sistem pencernaan",
            "Meningkatkan sistem imunitas",
            "Menyehatkan dan mencerahkan kulit",
            "Mengandung antioksidan tinggi",
            "Membantu menurunkan kolesterol",
            "Meredakan nyeri haid"
        ],
        "cara_penggunaan": [
            "**Untuk pencernaan**: Makan buah pepaya matang secara rutin setiap pagi",
            "**Untuk kulit**: Haluskan buah, gunakan sebagai masker wajah 15-20 menit",
            "**Untuk nyeri haid**: Minum jus daun pepaya yang direbus",
            "**Untuk imunitas**: Konsumsi buah segar yang kaya vitamin C dan A"
        ]
    },
    "Seledri": {
        "emoji": "🥬",
        "nama_latin": "Apium graveolens",
        "manfaat": [
            "Menurunkan tekanan darah tinggi",
            "Mengurangi peradangan dalam tubuh",
            "Menjaga kesehatan ginjal",
            "Membantu kualitas tidur lebih baik",
            "Menurunkan kadar kolesterol",
            "Membantu menurunkan berat badan"
        ],
        "cara_penggunaan": [
            "**Untuk tekanan darah**: Minum jus seledri segar setiap pagi (1-2 batang)",
            "**Untuk ginjal**: Rebus batang dan daun, minum air rebusannya 2x sehari",
            "**Untuk tidur**: Makan seledri segar atau minum jusnya sebelum tidur",
            "**Untuk diet**: Tambahkan ke dalam salad atau jus sayuran"
        ]
    },
    "Sirih": {
        "emoji": "🍃",
        "nama_latin": "Piper betle",
        "manfaat": [
            "Menyembuhkan luka dan infeksi",
            "Mengatasi bau mulut",
            "Mengobati mimisan",
            "Memiliki sifat antiseptik kuat",
            "Menjaga kesehatan organ kewanitaan",
            "Mengatasi gatal-gatal pada kulit"
        ],
        "cara_penggunaan": [
            "**Untuk bau mulut**: Rebus 4-5 lembar daun, gunakan untuk berkumur setiap hari",
            "**Untuk luka**: Tumbuk daun, tempelkan pada luka sebagai antiseptik",
            "**Untuk mimisan**: Gulung daun sirih, masukkan ke lubang hidung yang berdarah",
            "**Untuk kesehatan kewanitaan**: Rebus daun sirih, gunakan untuk cebok setelah dingin"
        ]
    }
}

@st.cache_resource
def load_trained_model():
    """Load model yang sudah dilatih"""
    try:
        model = load_model("daun-herbal.keras")
    except:
        model = load_model("daun-herbal.h5")
    return model

def preprocess_image(image):
    """Preprocessing gambar sesuai dengan training"""
    # Resize ke 224x224
    image = image.resize((224, 224))
    # Konversi ke array
    img_array = np.array(image)
    # Pastikan 3 channel (RGB)
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    # Normalisasi (rescale 1/255)
    img_array = img_array / 255.0
    # Tambah dimensi batch
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(model, image):
    """Prediksi kelas gambar"""
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx] * 100
    return CLASS_NAMES[predicted_class_idx], confidence, predictions[0]

def show_prediction_result(predicted_class, confidence, all_predictions, image):
    """Menampilkan hasil prediksi"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖼️ Gambar Input")
        st.image(image, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Hasil Prediksi")
        info = INFO_HERBAL[predicted_class]
        st.markdown(f"### {info['emoji']} {predicted_class}")
        st.caption(f"*{info['nama_latin']}*")
        
        # Progress bar untuk confidence
        st.metric(label="Tingkat Keyakinan", value=f"{confidence:.2f}%")
        st.progress(float(confidence / 100))
        
        if confidence >= 80:
            st.success("✅ Prediksi dengan keyakinan tinggi")
        elif confidence >= 50:
            st.warning("⚠️ Prediksi dengan keyakinan sedang")
        else:
            st.error("❌ Prediksi dengan keyakinan rendah")
    
    st.markdown("---")
    
    # Manfaat dan cara penggunaan tanaman
    col_m, col_c = st.columns(2)
    
    with col_m:
        st.subheader("💊 Manfaat")
        for manfaat in info["manfaat"]:
            st.markdown(f"- {manfaat}")
    
    with col_c:
        st.subheader("📝 Cara Penggunaan")
        for cara in info["cara_penggunaan"]:
            st.markdown(cara)
    
    # Tampilkan probabilitas semua kelas
    st.markdown("---")
    st.subheader("📊 Probabilitas Semua Kelas")
    
    prob_df = pd.DataFrame({
        "Tanaman": CLASS_NAMES,
        "Probabilitas (%)": [p * 100 for p in all_predictions]
    })
    prob_df = prob_df.sort_values("Probabilitas (%)", ascending=True)
    
    st.bar_chart(prob_df.set_index("Tanaman"))

# ==================== SIDEBAR NAVIGATION ====================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/leaf.png", width=80)
    st.title("🌿 Herbal Classifier")
    st.markdown("---")
    
    # Menu navigasi
    menu = st.radio(
        "📌 Menu Navigasi",
        ["🏠 Home", "🔍 Predict", "📚 About"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### ℹ️ Info Aplikasi")
    st.markdown("""
    - **Model**: MobileNetV2
    - **Input Size**: 224x224 px
    - **Framework**: TensorFlow/Keras
    - **Jumlah Kelas**: 10 Tanaman
    """)
    
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray; font-size: 12px;'>
            Made with ❤️ using Streamlit
        </div>
        """,
        unsafe_allow_html=True
    )

# ==================== HOME PAGE ====================
if menu == "🏠 Home":
    st.title("🌿 Klasifikasi Daun Tanaman Herbal Indonesia")
    st.markdown("---")
    
    # Hero section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Selamat Datang! 👋
        
        Aplikasi ini menggunakan teknologi **Deep Learning** dengan arsitektur **MobileNetV2** 
        untuk mengklasifikasikan **10 jenis daun tanaman herbal** Indonesia secara otomatis.
        
        ### ✨ Fitur Utama:
        - 🖼️ **Upload Gambar** - Upload foto daun dari perangkat Anda
        - 📷 **Live Camera** - Ambil foto langsung menggunakan kamera
        - 🎯 **Prediksi Akurat** - Identifikasi jenis tanaman dengan tingkat akurasi tinggi
        - 💊 **Info Lengkap** - Dapatkan informasi manfaat dan cara penggunaan
        
        ### 🚀 Cara Menggunakan:
        1. Pilih menu **"🔍 Predict"** di sidebar
        2. Upload gambar atau gunakan kamera
        3. Lihat hasil prediksi dan informasi tanaman
        """)
        
        st.markdown("---")
        
        if st.button("🔍 Mulai Prediksi Sekarang", type="primary", use_container_width=True):
            st.session_state.nav = "predict"
            st.rerun()
    
    with col2:
        st.markdown("### 🌱 Tanaman yang Dikenali:")
        for i, name in enumerate(CLASS_NAMES, 1):
            emoji = INFO_HERBAL[name]["emoji"]
            st.markdown(f"{i}. {emoji} **{name}**")
    
    st.markdown("---")
    
    # Statistik
    st.subheader("📊 Statistik Model")
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    
    with col_s1:
        st.metric("Jumlah Kelas", "10")
    with col_s2:
        st.metric("Arsitektur", "MobileNetV2")
    with col_s3:
        st.metric("Input Size", "224x224")
    with col_s4:
        st.metric("Framework", "TensorFlow")

# ==================== PREDICT PAGE ====================
elif menu == "🔍 Predict":
    st.title("🔍 Prediksi Daun Tanaman Herbal")
    st.markdown("---")
    
    # Load model
    with st.spinner("Memuat model..."):
        model = load_trained_model()
    st.success("✅ Model berhasil dimuat!")
    
    st.markdown("---")
    
    # Tab untuk upload dan camera
    tab_upload, tab_camera = st.tabs(["📤 Upload Gambar", "📷 Live Camera"])
    
    # ===== TAB UPLOAD =====
    with tab_upload:
        st.subheader("📤 Upload Gambar Daun")
        st.markdown("Upload gambar daun tanaman herbal dari perangkat Anda.")
        
        uploaded_file = st.file_uploader(
            "Pilih gambar daun herbal...",
            type=["jpg", "jpeg", "png"],
            help="Upload gambar daun dengan format JPG, JPEG, atau PNG",
            key="upload_file"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            
            with st.spinner("🔄 Menganalisis gambar..."):
                predicted_class, confidence, all_predictions = predict_image(model, image)
            
            show_prediction_result(predicted_class, confidence, all_predictions, image)
        else:
            st.info("👆 Silakan upload gambar daun untuk memulai klasifikasi")
    
    # ===== TAB CAMERA =====
    with tab_camera:
        st.subheader("📷 Ambil Foto dengan Kamera")
        st.markdown("Gunakan kamera perangkat Anda untuk mengambil foto daun secara langsung.")
        
        camera_image = st.camera_input(
            "Arahkan kamera ke daun tanaman",
            key="camera_input"
        )
        
        if camera_image is not None:
            image = Image.open(camera_image).convert("RGB")
            
            with st.spinner("🔄 Menganalisis gambar..."):
                predicted_class, confidence, all_predictions = predict_image(model, image)
            
            show_prediction_result(predicted_class, confidence, all_predictions, image)
        else:
            st.info("📷 Klik tombol kamera untuk mengambil foto daun")

# ==================== ABOUT PAGE ====================
elif menu == "📚 About":
    st.title("📚 Informasi Tanaman Herbal")
    st.markdown("---")
    
    st.markdown("""
    Halaman ini berisi informasi lengkap tentang **10 jenis tanaman herbal** yang dapat dikenali oleh aplikasi.
    Setiap tanaman dilengkapi dengan **manfaat kesehatan** dan **cara penggunaan** yang benar.
    """)
    
    st.markdown("---")
    
    # Pilih tanaman untuk melihat detail
    selected_plant = st.selectbox(
        "🌿 Pilih Tanaman untuk Melihat Detail:",
        CLASS_NAMES,
        index=0
    )
    
    if selected_plant:
        info = INFO_HERBAL[selected_plant]
        
        st.markdown("---")
        
        # Header tanaman
        st.markdown(f"## {info['emoji']} {selected_plant}")
        st.markdown(f"*Nama Latin: **{info['nama_latin']}***")
        
        st.markdown("---")
        
        # Manfaat dan cara penggunaan dalam 2 kolom
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💊 Manfaat Kesehatan")
            for i, manfaat in enumerate(info["manfaat"], 1):
                st.markdown(f"{i}. {manfaat}")
        
        with col2:
            st.subheader("📝 Cara Penggunaan")
            for cara in info["cara_penggunaan"]:
                st.markdown(cara)
                st.markdown("")
    
    st.markdown("---")
    
    # Daftar semua tanaman dalam bentuk grid
    st.subheader("🌱 Daftar Semua Tanaman Herbal")
    
    cols = st.columns(5)
    for i, name in enumerate(CLASS_NAMES):
        with cols[i % 5]:
            info = INFO_HERBAL[name]
            with st.container():
                st.markdown(f"""
                <div style='text-align: center; padding: 10px; border-radius: 10px; background-color: #f0f2f6; margin: 5px;'>
                    <span style='font-size: 30px;'>{info['emoji']}</span>
                    <p style='margin: 5px 0; font-weight: bold;'>{name}</p>
                    <p style='font-size: 11px; color: gray; font-style: italic;'>{info['nama_latin']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Disclaimer
    st.warning("""
    ⚠️ **Disclaimer**: Informasi yang disajikan dalam aplikasi ini hanya untuk tujuan edukasi. 
    Selalu konsultasikan dengan ahli kesehatan atau dokter sebelum menggunakan tanaman herbal untuk pengobatan.
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>🌿 Klasifikasi Daun Tanaman Herbal dengan MobileNetV2</p>
        <p style='font-size: 12px;'>© 2025 - Herbal Leaf Classifier</p>
    </div>
    """,
    unsafe_allow_html=True
)
