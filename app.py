import streamlit as st
import joblib
import pandas as pd
import re

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# =====================================================
# KONFIGURASI HALAMAN
# =====================================================
st.set_page_config(
    page_title="Analisis Sentimen JogjaKita",
    page_icon="📊",
    layout="centered"
)

# =====================================================
# STYLE (CSS)
# =====================================================
st.markdown(
    """
    <style>
    /* Background utama */
    .stApp {
        background-color: #ffffff;
        color: #000000;
    }

    /* Semua teks */
    h1, h2, h3, h4, h5, h6, p, span, label {
        color: #000000 !important;
    }

    /* Subtitle / deskripsi metode */
.subtitle {
    font-size: 30px;        /* BESARIN FONT */
    font-weight: 500;
    color: #333333;
    margin-top: 4px;
}

    /* Text area & input */
    textarea, input {
        background-color: #f9f9f9 !important;
        color: #000000 !important;
        border: 1px solid #cccccc !important;
        border-radius: 8px !important;
    }

    /* KURSOR TEKS (INI YANG KAMU MINTA) */
    textarea {
        caret-color: #000000 !important;
    }

    /* Tombol */
    div.stButton > button {
        background-color: #e53935 !important;
        color: white !important;
        border-radius: 8px;
        padding: 0.5em 1.2em;
        border: none;
        font-weight: 600;
    }

    div.stButton > button:hover {
        background-color: #c62828 !important;
    }
    /* ===== HILANGKAN BAR HITAM STREAMLIT CLOUD ===== */

/* Toolbar atas */
div[data-testid="stToolbar"] {
    display: none !important;
}

/* Header utama */
header[data-testid="stHeader"] {
    display: none !important;
}

/* Decoration / Deploy bar */
div[data-testid="stDecoration"] {
    display: none !important;
}

/* Hilangkan padding atas bawaan */
.block-container {
    padding-top: 0rem !important;
}
    </style>
    """,
    unsafe_allow_html=True
)

# =====================================================
# LOAD MODEL & TF-IDF
# =====================================================
@st.cache_resource
def load_model():
    model = joblib.load("svm_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# =====================================================
# PREPROCESSING
# =====================================================
stemmer = StemmerFactory().create_stemmer()

stop_factory = StopWordRemoverFactory()
stopwords = set(stop_factory.get_stop_words())
stopwords.update({"nya","sih","kok","lah","dong","nih","deh","banget","ya","pun"})

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = [stemmer.stem(w) for w in text.split()
              if w not in stopwords and len(w) > 2]

    return " ".join(tokens)

# =====================================================
# SESSION STATE
# =====================================================
if "history" not in st.session_state:
    st.session_state.history = []

# =====================================================
# HEADER
# =====================================================
col1, col2 = st.columns([1, 5])

with col1:
    st.image("logo.png", width=110)

with col2:
    st.markdown(
        """
        <h2>Analisis Sentimen Ulasan JogjaKita</h2>
        <p class="subtitle">
            Menggunakan Algoritma <b>Support Vector Machine (SVM)</b>
        </p>
        """,
        unsafe_allow_html=True
    )

st.divider()

# =====================================================
# INPUT TEKS
# =====================================================
st.subheader("📝 Masukkan Ulasan Pengguna")

input_text = st.text_area(
    "Contoh: Aplikasi JogjaKita sangat membantu dan drivernya ramah",
    height=120,
    placeholder="Ketik ulasan pengguna di sini..."
)

# =====================================================
# PREDIKSI
# =====================================================
if st.button("🔍 Prediksi Sentimen"):
    if input_text.strip() == "":
        st.warning("Silakan masukkan teks ulasan terlebih dahulu.")
    else:
        clean_text = preprocess(input_text)
        vector = vectorizer.transform([clean_text])

        pred_label = model.predict(vector)[0]
        proba = model.predict_proba(vector)[0]

        prob_negatif = proba[0] * 100
        prob_positif = proba[1] * 100

        label_text = "Positif" if pred_label == 1 else "Negatif"

        # Simpan riwayat
        st.session_state.history.append({
            "Ulasan": input_text,
            "Sentimen": label_text,
            "Probabilitas Positif (%)": round(prob_positif, 2),
            "Probabilitas Negatif (%)": round(prob_negatif, 2)
        })

        st.divider()
        st.subheader("📌 Hasil Prediksi")

        if pred_label == 1:
            st.success("✅ Sentimen Positif")
        else:
            st.error("❌ Sentimen Negatif")

        st.markdown("### 📈 Probabilitas Prediksi")
        st.write(f"**Positif : {prob_positif:.2f}%**")
        st.progress(prob_positif / 100)

        st.write(f"**Negatif : {prob_negatif:.2f}%**")
        st.progress(prob_negatif / 100)

# =====================================================
# RIWAYAT PREDIKSI
# =====================================================
st.divider()
st.subheader("🗂️ Riwayat Prediksi")

if st.session_state.history:
    df_history = pd.DataFrame(st.session_state.history)
    st.dataframe(df_history, use_container_width=True)

    if st.button("🧹 Hapus Riwayat"):
        st.session_state.history = []
        st.rerun()
else:
    st.info("Belum ada riwayat prediksi.")

# =====================================================
# FOOTER
# =====================================================
st.divider()
st.caption("""
ℹ️ **Informasi Model**
- Algoritma : Support Vector Machine (Kernel RBF)
- Ekstraksi Fitur : TF-IDF (Unigram & Bigram)
- Dataset : Google Play Store – Aplikasi JogjaKita
""")



