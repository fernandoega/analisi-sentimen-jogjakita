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
# MODERN CSS + ANIMATION
# =====================================================
st.markdown(
    """
    <style>
    /* RESET */
    .stApp { background: #f4f6f8; }

    /* HIDE STREAMLIT UI */
    header, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {
        display: none !important;
    }

    /* HERO */
    .hero {
        background: linear-gradient(135deg, #e53935, #c62828);
        padding: 36px;
        border-radius: 20px;
        color: white;
        margin-bottom: 30px;
        animation: fadeDown 0.8s ease;
    }

    .hero h1 {
        font-size: 34px;
        margin-bottom: 8px;
    }

    .hero p {
        font-size: 18px;
        opacity: 0.95;
    }

    /* CARD */
    .card {
        background: white;
        padding: 28px;
        border-radius: 18px;
        box-shadow: 0 12px 30px rgba(0,0,0,0.08);
        margin-bottom: 26px;
        animation: fadeUp 0.6s ease;
    }

    /* BUTTON */
    div.stButton > button {
        background: linear-gradient(135deg, #e53935, #c62828);
        color: white;
        border-radius: 10px;
        padding: 0.6em 1.6em;
        font-weight: 600;
        border: none;
        transition: all 0.25s ease;
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(229,57,53,0.45);
    }

    /* RESULT */
    .positive {
        background: #e8f5e9;
        border-left: 6px solid #2e7d32;
        padding: 16px;
        border-radius: 12px;
        margin-top: 10px;
    }

    .negative {
        background: #ffebee;
        border-left: 6px solid #c62828;
        padding: 16px;
        border-radius: 12px;
        margin-top: 10px;
    }

    /* ANIMATION */
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes fadeDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
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
    return joblib.load("svm_model.pkl"), joblib.load("tfidf_vectorizer.pkl")

model, vectorizer = load_model()

# =====================================================
# PREPROCESS
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
# HERO SECTION
# =====================================================
st.markdown(
    """
    <div class="hero">
        <h1>📊 Analisis Sentimen JogjaKita</h1>
        <p>Menggunakan Support Vector Machine (SVM)</p>
    </div>
    """,
    unsafe_allow_html=True
)

# =====================================================
# INPUT CARD
# =====================================================
st.markdown(
    """
    <div class="card">
        <h3>📝 Masukkan Ulasan Pengguna</h3>
        <p style="color:#555;">
        Contoh: Aplikasi JogjaKita sangat membantu dan drivernya ramah
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

input_text = st.text_area(
    "",
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
        vec = vectorizer.transform([clean])
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]

        pos, neg = proba[1]*100, proba[0]*100
        label = "Positif" if pred == 1 else "Negatif"

        st.session_state.history.append({
            "Ulasan": input_text,
            "Sentimen": label,
            "Positif (%)": round(pos,2),
            "Negatif (%)": round(neg,2)
        })

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📌 Hasil Prediksi")

        if pred == 1:
            st.markdown(
                f"<div class='positive'>✅ <b>Sentimen Positif</b><br>Probabilitas: {pos:.2f}%</div>",
                unsafe_allow_html=True
            )
            st.progress(pos/100)
        else:
            st.markdown(
                f"<div class='negative'>❌ <b>Sentimen Negatif</b><br>Probabilitas: {neg:.2f}%</div>",
                unsafe_allow_html=True
            )
            st.progress(neg/100)

        st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# RIWAYAT
# =====================================================
if st.session_state.history:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🗂️ Riwayat Prediksi")
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================
st.caption(
    "ℹ️ Model: SVM (Kernel RBF) | Fitur: TF-IDF | Dataset: Google Play Store – JogjaKita"
)
