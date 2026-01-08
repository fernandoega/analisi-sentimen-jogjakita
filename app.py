import streamlit as st
import joblib
import pandas as pd
import re

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Analisis Sentimen JogjaKita",
    page_icon="analisis.png",
    layout="centered"
)

# =====================================================
# GLOBAL STYLE (MINIMAL & AMAN)
# =====================================================
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f7f8fa;
    }

    h1, h2, h3 {
        color: #111111;
    }

    p {
        color: #444444;
    }

    /* Button */
    div.stButton > button {
        background-color: #e53935;
        color: white;
        border-radius: 8px;
        padding: 0.55em 1.4em;
        font-weight: 600;
        border: none;
    }

    div.stButton > button:hover {
        background-color: #c62828;
    }

    /* Hide Streamlit cloud bar */
    header, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =====================================================
# LOAD MODEL
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
stopwords = set(StopWordRemoverFactory().get_stop_words())
stopwords.update({"nya","sih","kok","lah","dong","nih","deh","banget","ya","pun"})

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(
        stemmer.stem(w) for w in text.split()
        if w not in stopwords and len(w) > 2
    )

# =====================================================
# SESSION STATE
# =====================================================
if "history" not in st.session_state:
    st.session_state.history = []

# =====================================================
# HEADER (DASHBOARD STYLE)
# =====================================================
col1, col2 = st.columns([1, 6])

with col1:
    st.image("logo.png", width=90)

with col2:
    st.title("Analisis Sentimen Ulasan JogjaKita")
    st.caption("Menggunakan Algoritma Support Vector Machine (SVM)")

st.divider()

# =====================================================
# DESKRIPSI SISTEM
# =====================================================
st.info(
    "Sistem ini menganalisis sentimen ulasan pengguna aplikasi **JogjaKita** "
    "menjadi **positif** atau **negatif** menggunakan algoritma "
    "**Support Vector Machine (SVM)** berbasis fitur **TF-IDF**."
)

# =====================================================
# INPUT
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
    if not input_text.strip():
        st.warning("Silakan masukkan teks ulasan terlebih dahulu.")
    else:
        clean = preprocess(input_text)
        vector = vectorizer.transform([clean])

        pred = model.predict(vector)[0]
        proba = model.predict_proba(vector)[0]

        pos = proba[1] * 100
        neg = proba[0] * 100

        st.divider()
        st.subheader("📌 Hasil Prediksi")

        if pred == 1:
            st.success(f"Sentimen Positif ({pos:.2f}%)")
            st.progress(pos / 100)
        else:
            st.error(f"Sentimen Negatif ({neg:.2f}%)")
            st.progress(neg / 100)

        st.session_state.history.append({
            "Ulasan": input_text,
            "Sentimen": "Positif" if pred == 1 else "Negatif",
            "Probabilitas Positif (%)": round(pos, 2),
            "Probabilitas Negatif (%)": round(neg, 2)
        })

# =====================================================
# RIWAYAT
# =====================================================
st.divider()
st.subheader("🗂️ Riwayat Prediksi")

if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True)

    if st.button("🧹 Hapus Riwayat"):
        st.session_state.history = []
        st.rerun()
else:
    st.info("Belum ada riwayat prediksi.")

# =====================================================
# FOOTER
# =====================================================
st.caption(
    "ℹ️ Model: SVM (Kernel RBF) | Fitur: TF-IDF | "
    "Dataset: Google Play Store – Aplikasi JogjaKita"
)
