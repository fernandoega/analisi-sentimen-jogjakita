"""
===============================================================================
ANALISIS SENTIMEN ULASAN PENGGUNA APLIKASI JOGJAKITA
MENGGUNAKAN ALGORITMA SUPPORT VECTOR MACHINE (SVM)
===============================================================================
Aplikasi ini dikembangkan untuk keperluan Skripsi
Framework : Streamlit
Bahasa    : Python
"""

# =====================================================
# IMPORT LIBRARY (WAJIB DI PALING ATAS)
# =====================================================
import streamlit as st
import joblib
import pandas as pd
import re
from datetime import datetime

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
# CUSTOM CSS (MINIMAL, AMAN, TANPA KONFLIK)
# =====================================================
st.markdown("""
<style>
.stApp {
    background-color: #F8F9FA;
}

h1, h2, h3 {
    color: #212121;
}

textarea {
    background-color: #FFFFFF !important;
    color: #000000 !important;
    border-radius: 8px !important;
}

div.stButton > button {
    background-color: #E53935;
    color: white;
    border-radius: 8px;
    padding: 10px 24px;
    font-weight: 600;
}

div.stButton > button:hover {
    background-color: #C62828;
}

.card {
    background-color: #FFFFFF;
    padding: 24px;
    border-radius: 14px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    margin-bottom: 24px;
}

.result-positive {
    background-color: #E8F5E9;
    border-left: 6px solid #2E7D32;
    padding: 16px;
    border-radius: 10px;
}

.result-negative {
    background-color: #FFEBEE;
    border-left: 6px solid #C62828;
    padding: 16px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

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
col1, col2 = st.columns([1, 6])

with col1:
    st.image("logo.png", width=90)

with col2:
    st.markdown("""
    <h2>Analisis Sentimen Ulasan JogjaKita</h2>
    <p>Menggunakan Algoritma <b>Support Vector Machine (SVM)</b></p>
    """, unsafe_allow_html=True)

st.divider()

# =====================================================
# DESKRIPSI SISTEM
# =====================================================
st.markdown("""
<div class="card">
<b>Deskripsi Sistem</b><br><br>
Sistem ini digunakan untuk menganalisis sentimen ulasan pengguna aplikasi
<b>JogjaKita</b> menjadi sentimen <b>positif</b> atau <b>negatif</b>
menggunakan algoritma <b>Support Vector Machine (SVM)</b> berbasis fitur
<b>TF-IDF</b>.
</div>
""", unsafe_allow_html=True)

# =====================================================
# INPUT ULASAN
# =====================================================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📝 Masukkan Ulasan Pengguna")

input_text = st.text_area(
    "Contoh: Aplikasi JogjaKita sangat membantu dan drivernya ramah",
    height=120,
    placeholder="Ketik ulasan pengguna di sini..."
)
st.markdown("</div>", unsafe_allow_html=True)

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
        sentiment = "Positif" if pred_label == 1 else "Negatif"

        st.session_state.history.append({
            "Ulasan": input_text,
            "Sentimen": sentiment,
            "Probabilitas Positif (%)": round(prob_positif, 2),
            "Probabilitas Negatif (%)": round(prob_negatif, 2),
            "Waktu": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📌 Hasil Prediksi")

        if sentiment == "Positif":
            st.markdown(f"""
            <div class="result-positive">
            ✅ <b>Sentimen Positif</b><br>
            Probabilitas Positif: {prob_positif:.2f}%
            </div>
            """, unsafe_allow_html=True)
            st.progress(prob_positif / 100)
        else:
            st.markdown(f"""
            <div class="result-negative">
            ❌ <b>Sentimen Negatif</b><br>
            Probabilitas Negatif: {prob_negatif:.2f}%
            </div>
            """, unsafe_allow_html=True)
            st.progress(prob_negatif / 100)

        st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# RIWAYAT
# =====================================================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📂 Riwayat Prediksi")

if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True)

    if st.button("🧹 Hapus Riwayat"):
        st.session_state.history = []
        st.rerun()
else:
    st.info("Belum ada riwayat prediksi.")

st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================
st.divider()
st.caption("""
ℹ️ **Informasi Model**
- Algoritma : Support Vector Machine (Kernel RBF)
- Ekstraksi Fitur : TF-IDF
- Dataset : Google Play Store – Aplikasi JogjaKita
""")
