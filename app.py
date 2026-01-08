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
    page_icon="analisis.png",
    layout="centered"
)

# =====================================================
# STYLE (CSS)
# =====================================================
st.markdown(
    """
    <style>
    /* ===== GLOBAL ===== */
    .stApp {
        background-color: #ffffff;
        color: #000000;
    }

    h1, h2, h3, h4, h5, h6, p, span, label {
        color: #000000 !important;
    }

    /* ===== SUBTITLE ===== */
    .subtitle {
        font-size: 22px;
        font-weight: 500;
        color: #333333;
        margin-top: 4px;
    }

    /* ===== INPUT ===== */
    textarea, input {
        background-color: #f9f9f9 !important;
        color: #000000 !important;
        border: 1px solid #cccccc !important;
        border-radius: 8px !important;
    }

    textarea {
        caret-color: #000000 !important;
    }

    /* ===== BUTTON ===== */
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

    /* ===== CARD ===== */
    .card {
        background-color: #ffffff;
        padding: 24px 28px;
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.08);
        margin-bottom: 28px;
    }

    .section-title {
        font-size: 20px;
        font-weight: 600;
        margin-bottom: 12px;
    }

    /* ===== RESULT ===== */
    .result-positive {
        background-color: #e8f5e9;
        border-left: 6px solid #2e7d32;
        padding: 16px;
        border-radius: 10px;
        margin-bottom: 12px;
    }

    .result-negative {
        background-color: #ffebee;
        border-left: 6px solid #c62828;
        padding: 16px;
        border-radius: 10px;
        margin-bottom: 12px;
    }

    /* ===== HILANGKAN BAR STREAMLIT CLOUD ===== */
    div[data-testid="stToolbar"],
    header[data-testid="stHeader"],
    div[data-testid="stDecoration"] {
        display: none !important;
    }

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
stopwords.update({"nya", "sih", "kok", "lah", "dong", "nih", "deh", "banget", "ya", "pun"})

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [stemmer.stem(w) for w in text.split() if w not in stopwords and len(w) > 2]
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
st.markdown(
    """
    <div class="card">
        <div class="section-title">📘 Deskripsi Sistem</div>
        <p style="font-size:16px; line-height:1.7; color:#444;">
        Sistem ini digunakan untuk menganalisis sentimen ulasan pengguna aplikasi
        <b>JogjaKita</b> menjadi sentimen <b>positif</b> atau <b>negatif</b>
        menggunakan algoritma <b>Support Vector Machine (SVM)</b>.
        <br><br>
        Pengguna dapat memasukkan satu ulasan pada kolom yang tersedia untuk
        memperoleh hasil prediksi sentimen secara otomatis.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# =====================================================
# INPUT
# =====================================================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>📝 Masukkan Ulasan Pengguna</div>", unsafe_allow_html=True)

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

        label_text = "Positif" if pred_label == 1 else "Negatif"

        st.session_state.history.append({
            "Ulasan": input_text,
            "Sentimen": label_text,
            "Probabilitas Positif (%)": round(prob_positif, 2),
            "Probabilitas Negatif (%)": round(prob_negatif, 2)
        })

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>📌 Hasil Prediksi</div>", unsafe_allow_html=True)

        if pred_label == 1:
            st.markdown(
                f"""
                <div class="result-positive">
                    ✅ <b>Sentimen Positif</b><br>
                    Probabilitas Positif: {prob_positif:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )
            st.progress(prob_positif / 100)
        else:
            st.markdown(
                f"""
                <div class="result-negative">
                    ❌ <b>Sentimen Negatif</b><br>
                    Probabilitas Negatif: {prob_negatif:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )
            st.progress(prob_negatif / 100)

        st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# RIWAYAT
# =====================================================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>🗂️ Riwayat Prediksi</div>", unsafe_allow_html=True)

if st.session_state.history:
    df_history = pd.DataFrame(st.session_state.history)
    st.dataframe(df_history, use_container_width=True)

    if st.button("🧹 Hapus Riwayat"):
        st.session_state.history = []
        st.rerun()
else:
    st.info("Belum ada riwayat prediksi.")

st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================
st.caption("""
ℹ️ **Informasi Model**  
- Algoritma : Support Vector Machine (Kernel RBF)  
- Ekstraksi Fitur : TF-IDF  
- Dataset : Google Play Store – Aplikasi JogjaKita  
""")

