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
st.markdown(
    """
    <style>
    /* ===== GLOBAL ===== */
    .stApp {
        background-color: #f5f6f8;
        color: #000000;
    }

    h1, h2, h3, h4, h5, h6, p, span, label {
        color: #000000 !important;
    }

    /* ===== CARD UTAMA ===== */
    .main-card {
        background-color: #ffffff;
        padding: 32px 36px;
        border-radius: 14px;
        box-shadow: 0 8px 28px rgba(0,0,0,0.08);
        max-width: 920px;
        margin: 28px auto;
    }

    /* ===== SECTION TITLE ===== */
    .section-title {
        margin-top: 28px;
        margin-bottom: 10px;
        font-weight: 600;
    }

    /* ===== INPUT ===== */
    textarea, input {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #d0d5dd !important;
        border-radius: 10px !important;
    }

    /* ===== BUTTON (JogjaKita Red) ===== */
    div.stButton > button {
        background-color: #e53935 !important;
        color: #ffffff !important;
        border-radius: 10px;
        padding: 0.6em 1.4em;
        border: none;
        font-weight: 600;
        box-shadow: 0 6px 16px rgba(229,57,53,0.35);
    }

    div.stButton > button:hover {
        background-color: #c62828 !important;
        box-shadow: 0 8px 20px rgba(198,40,40,0.45);
        transform: translateY(-1px);
    }

    /* ===== INFO / ALERT ===== */
    .stAlert {
        border-radius: 10px;
    }

    /* ===== HR ===== */
    hr {
        margin: 26px 0;
        border: none;
        border-top: 1px solid #e6e6e6;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown("<div class='main-card'>", unsafe_allow_html=True)    
# =====================================================
# LOAD MODEL & TF-IDF (CACHED)
# =====================================================
@st.cache_resource
def load_model():
    model = joblib.load("svm_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# =====================================================
# PREPROCESSING (SAMA DENGAN TRAINING)
# =====================================================
stemmer = StemmerFactory().create_stemmer()

stop_factory = StopWordRemoverFactory()
stopwords = set(stop_factory.get_stop_words())
extra_stopwords = {
    "nya","sih","kok","lah","dong","nih","deh","banget","ya","pun"
}
stopwords.update(extra_stopwords)

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    tokens = [w for w in tokens if w not in stopwords and len(w) > 2]
    tokens = [stemmer.stem(w) for w in tokens]

    return " ".join(tokens)

# =====================================================
# SESSION STATE (RIWAYAT)
# =====================================================
if "history" not in st.session_state:
    st.session_state.history = []

# =====================================================
# HEADER
# =====================================================
st.markdown("<hr>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 6])

with col1:
    st.image("logo.png", width=80)

with col2:
    st.markdown(
        """
        <h2 style="margin-bottom:6px;">Analisis Sentimen Ulasan JogjaKita</h2>
        <p style="margin-top:0; color:#555;">
        Menggunakan Algoritma <b>Support Vector Machine (SVM)</b>
        </p>
        """,
        unsafe_allow_html=True
    )

st.markdown("<hr>", unsafe_allow_html=True)

# =====================================================
# INPUT TEKS
# =====================================================
st.markdown("<h4 class='section-title'>📝 Masukkan Ulasan Pengguna</h4>", unsafe_allow_html=True)
input_text = st.text_area(
    "Contoh: Aplikasi JogjaKita sangat membantu dan drivernya ramah",
    height=120
)

# =====================================================
# TOMBOL PREDIKSI
# =====================================================
if st.button("🔍 Prediksi Sentimen"):
    if input_text.strip() == "":
        st.warning("Silakan masukkan teks ulasan terlebih dahulu.")
    else:
        clean_text = preprocess(input_text)
        vector = vectorizer.transform([clean_text])

        pred_label = model.predict(vector)[0]

        # Probabilitas (probability=True)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(vector)[0]
            prob_negatif = proba[0] * 100
            prob_positif = proba[1] * 100
        else:
            prob_negatif = None
            prob_positif = None

        label_text = "Positif" if pred_label == 1 else "Negatif"

        # Simpan ke riwayat
        st.session_state.history.append({
            "Ulasan": input_text,
            "Sentimen": label_text,
            "Akurasi Positif (%)": round(prob_positif, 2) if prob_positif else "-",
            "Akurasi Negatif (%)": round(prob_negatif, 2) if prob_negatif else "-"
        })

        st.divider()
        st.subheader("📌 Hasil Prediksi")

        if pred_label == 1:
            st.success("✅ Sentimen Positif")
        else:
            st.error("❌ Sentimen Negatif")

        if prob_positif is not None:
            st.markdown("### 📈 Probabilitas Prediksi")
            st.write(f"**Positif : {prob_positif:.2f}%**")
            st.progress(prob_positif / 100)

            st.write(f"**Negatif : {prob_negatif:.2f}%**")
            st.progress(prob_negatif / 100)

# =====================================================
# RIWAYAT PREDIKSI
# =====================================================
st.divider()
st.markdown("<h4 class='section-title'>📂 Riwayat Prediksi</h4>", unsafe_allow_html=True)

if len(st.session_state.history) > 0:
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
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("""
ℹ️ **Informasi Model**
- Algoritma : Support Vector Machine (Kernel RBF)
- Ekstraksi Fitur : TF-IDF (Unigram & Bigram)
- Dataset : Google Play Store – Aplikasi JogjaKita
""")
st.markdown("</div>", unsafe_allow_html=True)
