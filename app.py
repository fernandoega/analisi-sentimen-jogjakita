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
    /* Global */
    .stApp {
        background-color: #f5f6f8;
        color: #000000;
    }

    h1, h2, h3, h4 {
        color: #000000;
    }

    /* Card utama */
    .main-card {
        background-color: #ffffff;
        padding: 32px 36px;
        border-radius: 14px;
        box-shadow: 0 8px 28px rgba(0,0,0,0.08);
        max-width: 900px;
        margin: 20px auto;
    }

    /* Card hasil */
    .result-card {
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 20px;
        background-color: #ffffff;
        margin-top: 10px;
    }

    .result-positive {
        color: #2e7d32;
        font-weight: 700;
        font-size: 18px;
    }

    .result-negative {
        color: #c62828;
        font-weight: 700;
        font-size: 18px;
    }

    /* Tombol */
    div.stButton > button {
        background-color: #e53935;
        color: white;
        border-radius: 10px;
        padding: 0.6em 1.4em;
        border: none;
        font-weight: 600;
        box-shadow: 0 6px 16px rgba(229,57,53,0.35);
    }

    div.stButton > button:hover {
        background-color: #c62828;
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

stop_factory = StopWordRemoverFactory()
stopwords = set(stop_factory.get_stop_words())
stopwords.update({"nya","sih","kok","lah","dong","nih","deh","banget","ya","pun"})

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [stemmer.stem(w) for w in text.split() if w not in stopwords]
    return " ".join(tokens)

# =====================================================
# SESSION STATE
# =====================================================
if "history" not in st.session_state:
    st.session_state.history = []

# =====================================================
# UI
# =====================================================
st.markdown("<div class='main-card'>", unsafe_allow_html=True)

# Header
col1, col2 = st.columns([1, 6])
with col1:
    st.image("logo.png", width=80)
with col2:
    st.markdown(
        """
        <h2>Analisis Sentimen Ulasan JogjaKita</h2>
        <p>Menggunakan Algoritma <b>Support Vector Machine (SVM)</b></p>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# Input
st.subheader("📝 Masukkan Ulasan Pengguna")
input_text = st.text_area(
    "Contoh: Aplikasi JogjaKita sangat membantu dan drivernya ramah",
    height=120
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

        # ===== HASIL =====
        st.subheader("🧪 Hasil Prediksi")

        with st.container():
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)

            if pred_label == 1:
                st.markdown("<div class='result-positive'>✅ Sentimen Positif</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='result-negative'>❌ Sentimen Negatif</div>", unsafe_allow_html=True)

            st.write(f"**Positif : {prob_positif:.2f}%**")
            st.progress(prob_positif / 100)

            st.write(f"**Negatif : {prob_negatif:.2f}%**")
            st.progress(prob_negatif / 100)

            st.markdown("</div>", unsafe_allow_html=True)

        # Simpan riwayat
        st.session_state.history.append({
            "Ulasan": input_text,
            "Sentimen": label_text,
            "Probabilitas Positif (%)": round(prob_positif, 2),
            "Probabilitas Negatif (%)": round(prob_negatif, 2)
        })

# =====================================================
# RIWAYAT
# =====================================================
st.markdown("---")
st.subheader("📂 Riwayat Prediksi")

if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True)
else:
    st.info("Belum ada riwayat prediksi.")

st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================
st.caption("""
ℹ️ **Informasi Model**
- Algoritma: Support Vector Machine (Kernel RBF)
- Ekstraksi Fitur: TF-IDF (Unigram & Bigram)
- Dataset: Google Play Store – Aplikasi JogjaKita
""")
