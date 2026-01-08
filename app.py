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
    page_icon="😊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# =====================================================
# STYLE (CSS) - DIPERBARUI DAN DIPERBAIKI
# =====================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

    .stApp {
        background-color: #f0f2f6;
        font-family: 'Poppins', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6, p, span, label {
        color: #333333 !important;
    }

    /* Gaya untuk "Kartu" yang Ditingkatkan */
    .card {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
        margin-bottom: 25px;
        border: 1px solid #e2e8f0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
    }

    .hero-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        border: none; /* Hero card tidak perlu border */
    }
    
    .hero-card h1, .hero-card h2, .hero-card p {
        color: white !important;
    }
    
    .hero-card .subtitle {
        font-size: 18px;
        font-weight: 400;
        color: rgba(255, 255, 255, 0.9) !important;
        margin-top: 5px;
    }

    div.stButton > button:first-child {
        background-color: #667eea !important;
        color: white !important;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 12px 28px;
        transition: all 0.3s ease;
        width: 100%;
    }

    div.stButton > button:hover {
        background-color: #5a67d8 !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    .secondary-btn-container > button {
        background-color: transparent !important;
        color: #e53e3e !important;
        font-weight: 500;
        border: 1px solid #e53e3e !important;
        border-radius: 8px;
        padding: 8px 16px;
    }
    .secondary-btn-container > button:hover {
        background-color: #fff5f5 !important;
    }

    .stTextArea textarea {
        background-color: #f7fafc !important;
        color: #2d3748 !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 10px !important;
        font-size: 16px;
    }
    
    /* Gaya untuk Hasil Prediksi */
    .result-container {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 20px;
        border-radius: 10px;
        margin-top: 15px;
        font-size: 24px;
        font-weight: 600;
    }
    .result-positive {
        background-color: #b8f5d1; /* Warna hijau lebih kontras */
        color: #22543d;
    }
    .result-negative {
        background-color: #ffcdd2; /* Warna merah lebih kontras */
        color: #742a2a;
    }

    /* --- GAYA BARU UNTUK PROBABILITAS GABUNGAN --- */
    .prob-container {
        width: 100%;
        background-color: #e2e8f0;
        border-radius: 10px;
        overflow: hidden;
        height: 35px;
        display: flex;
        margin-top: 10px;
    }
    .prob-positive {
        background-color: #48bb78; /* Hijau tua */
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        transition: width 0.5s ease-in-out;
    }
    .prob-negative {
        background-color: #f56565; /* Merah tua */
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        transition: width 0.5s ease-in-out;
    }

    /* --- GAYA TABEL YANG DIPERBAIKI --- */
    .dataframe-container {
        max-height: 400px;
        overflow-y: auto;
        border-radius: 8px;
    }
    
    /* Style untuk header tabel */
    .dataframe-container thead th {
        background-color: #4a5568;
        color: white;
        font-weight: 600;
        text-align: left !important;
    }
    
    /* Style untuk baris tabel yang diwarnai */
    .dataframe tbody tr:nth-child(even) {
        background-color: #f7fafc;
    }
    
    /* Aplikasi warna dari styler pandas */
    .dataframe tbody tr td {
        font-weight: 500;
    }
    
    div[data-testid="stToolbar"], header[data-testid="stHeader"], div[data-testid="stDecoration"] {
        display: none !important;
    }
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 1rem !important;
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
st.markdown("""
<div class="card hero-card">
    <h1>Analisis Sentimen Ulasan JogjaKita</h1>
    <p class="subtitle">Menggunakan Algoritma <b>Support Vector Machine (SVM)</b> untuk menganalisis pendapat pengguna.</p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# INPUT TEKS
# =====================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### Masukkan Ulasan Pengguna")

input_text = st.text_area(
    "Contoh: Aplikasi JogjaKita sangat membantu dan drivernya ramah",
    height=120,
    placeholder="Ketik ulasan pengguna di sini...",
    label_visibility="collapsed"
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🔍 Prediksi Sentimen"):
        if input_text.strip() == "":
            st.warning("Silakan masukkan teks ulasan terlebih dahulu.", icon="⚠️")
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

st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# PREDIKSI - TAMPILAN HASIL DIPERBARUI
# =====================================================
if st.session_state.history:
    last_result = st.session_state.history[-1]
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Hasil Prediksi")
    
    result_icon = "😊" if last_result["Sentimen"] == "Positif" else "😞"
    result_class = "result-positive" if last_result["Sentimen"] == "Positif" else "result-negative"
    
    st.markdown(f"""
    <div class="result-container {result_class}">
        {result_icon} Sentimen {last_result['Sentimen']}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Probabilitas Prediksi")
    
    # --- TAMPILAN BAR PROBABILITAS GABUNGAN ---
    prob_pos = last_result['Probabilitas Positif (%)']
    prob_neg = last_result['Probabilitas Negatif (%)']
    
    st.markdown(f"""
    <div class="prob-container">
        <div class="prob-positive" style="width: {prob_pos}%;">
            {prob_pos:.1f}%
        </div>
        <div class="prob-negative" style="width: {prob_neg}%;">
            {prob_neg:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# RIWAYAT PREDIKSI - TABEL DIPERBAIKI
# =====================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### Riwayat Prediksi")

if st.session_state.history:
    df_history = pd.DataFrame(st.session_state.history)
    
    # --- FUNGSI HIGHLIGHT YANG SUDAH DIPERBAIKI ---
    def highlight_sentiment(row):
        # Tentukan warna berdasarkan kolom 'Sentimen'
        if row['Sentimen'] == 'Positif':
            color = '#b8f5d1'  # Hijau muda
        else:
            color = '#ffcdd2'  # Merah muda
        
        # Kembalikan list dengan warna yang sama untuk setiap kolom di baris tersebut
        return [f'background-color: {color}; font-weight: 500;'] * len(row)
    
    # Terapkan fungsi ke setiap baris (axis=1)
    styled_df = df_history.style.apply(highlight_sentiment, axis=1)
    
    st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
    st.dataframe(styled_df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="secondary-btn-container">', unsafe_allow_html=True)
    if st.button("🗑️ Hapus Riwayat"):
        st.session_state.history = []
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Belum ada riwayat prediksi.", icon="ℹ️")
    
st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================
st.markdown("""
<div class="card" style="text-align: center; background-color: #edf2f7;">
    <p style="font-size: 14px; color: #4a5568;">
        ℹ️ <b>Informasi Model:</b> Support Vector Machine (SVM) dengan TF-IDF | Dataset: Ulasan Aplikasi JogjaKita di Google Play Store
    </p>
</div>
""", unsafe_allow_html=True)
