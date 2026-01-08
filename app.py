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
    layout="centered"
)

# =====================================================
# STYLE (CSS) - PERUBAHAN SIGNIFIKAN DI SINI
# =====================================================
st.markdown(
    """
    <style>
    /* Import Font Modern */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Background dan Font Utama */
    .stApp {
        background-color: #f4f7f9;
        font-family: 'Inter', sans-serif;
    }

    /* Membuat Konten Utama Lebih Rapi dan Terfokus */
    .main-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 2rem;
    }

    /* Gaya Header */
    .header-block {
        background: linear-gradient(90deg, #6B46C1 0%, #9333EA 100%);
        padding: 2rem 1.5rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(107, 70, 193, 0.2);
    }

    .header-block h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
    }

    .header-block p {
        font-size: 1rem;
        font-weight: 400;
        margin-top: 0.5rem;
        color: rgba(255, 255, 255, 0.9);
    }

    /* Gaya Input Area */
    .stTextArea > div > div > textarea {
        border-radius: 8px;
        border: 1px solid #d1d5db;
        background-color: #ffffff;
        font-size: 1rem;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #9333EA;
        box-shadow: 0 0 0 3px rgba(147, 51, 234, 0.1);
    }

    /* Gaya Tombol */
    .stButton > button {
        background-color: #3B82F6;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        transition: all 0.2s ease-in-out;
        width: 100%;
    }

    .stButton > button:hover {
        background-color: #2563EB;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    /* Gaya Kotak Hasil */
    .result-box {
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
        margin-top: 1rem;
        font-weight: 600;
        font-size: 1.2rem;
    }
    .result-positive {
        background-color: #D1FAE5;
        color: #065F46;
        border: 1px solid #6EE7B7;
    }
    .result-negative {
        background-color: #FEE2E2;
        color: #991B1B;
        border: 1px solid #FCA5A5;
    }

    /* Gaya Progress Bar */
    .stProgress > div > div > div > div {
        border-radius: 4px;
    }
    .stProgress .progress-bar {
        background-color: #9333EA; /* Warna ungu untuk konsistensi */
    }

    /* Gaya Tabel Riwayat */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
    .dataframe thead th {
        background-color: #F3F4F6;
        color: #374151;
        font-weight: 600;
        text-align: left;
    }
    .dataframe tbody tr:nth-child(even) {
        background-color: #F9FAFB;
    }
    
    /* Gaya Footer */
    .footer-info {
        background-color: #F9FAFB;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        border: 1px solid #E5E7EB;
        font-size: 0.9rem;
        color: #6B7280;
    }

    /* Hilangkan elemen default Streamlit */
    div[data-testid="stToolbar"], header[data-testid="stHeader"], div[data-testid="stDecoration"] {
        display: none !important;
    }
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 2rem !important;
        max-width: 800px !important; /* <<< PERUBAHAN: Membatasi lebar konten */
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
# HEADER - PERUBAHAN MENGGUNAKAN HTML/CSS
# =====================================================
st.markdown("""
<div class="header-block">
    <h1>Analisis Sentimen Ulasan JogjaKita</h1>
    <p>Menggunakan Algoritma <b>Support Vector Machine (SVM)</b></p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# INPUT TEKS
# =====================================================
st.subheader("Masukkan Ulasan Pengguna")

input_text = st.text_area(
    "Contoh: Aplikasi JogjaKita sangat membantu dan drivernya ramah",
    height=120,
    placeholder="Ketik ulasan pengguna di sini...",
    label_visibility="collapsed"
)

# =====================================================
# PREDIKSI
# =====================================================
if st.button("Prediksi Sentimen"):
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
        emoji = "😊" if pred_label == 1 else "😞"

        # Simpan riwayat
        st.session_state.history.append({
            "Ulasan": input_text,
            "Sentimen": label_text,
            "Probabilitas Positif (%)": round(prob_positif, 2),
            "Probabilitas Negatif (%)": round(prob_negatif, 2)
        })

        # Tampilkan Hasil
        st.divider()
        st.subheader("Hasil Prediksi")
        
        # <<< PERUBAHAN: Tampilan hasil dengan kustom CSS
        result_class = "result-positive" if pred_label == 1 else "result-negative"
        st.markdown(f"""
        <div class="result-box {result_class}">
            {emoji} Sentimen {label_text}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Probabilitas Prediksi")
        col_pos, col_neg = st.columns(2)
        with col_pos:
            st.write(f"**Positif : {prob_positif:.2f}%**")
            st.progress(prob_positif / 100)
        with col_neg:
            st.write(f"**Negatif : {prob_negatif:.2f}%**")
            st.progress(prob_negatif / 100)


# =====================================================
# RIWAYAT PREDIKSI - PERUBAHAN PADA TABEL
# =====================================================
st.divider()
st.subheader("Riwayat Prediksi")

if st.session_state.history:
    df_history = pd.DataFrame(st.session_state.history)
    
    # <<< PERUBAHAN: Mewarnai tabel agar lebih mudah dibaca
    def color_sentiment(val):
        color = '#C6F6D5' if val == 'Positif' else '#FED7D7'
        return f'background-color: {color}; font-weight: 500'

    styled_df = df_history.style.applymap(color_sentiment, subset=['Sentimen'])
    
    st.dataframe(styled_df, use_container_width=True)

    if st.button("Hapus Riwayat"):
        st.session_state.history = []
        st.rerun()
else:
    st.info("Belum ada riwayat prediksi.")

# =====================================================
# FOOTER - PERUBAHAN MENGGUNAKAN HTML/CSS
# =====================================================
st.markdown("""
<br>
<div class="footer-info">
    ℹ️ <b>Informasi Model:</b> Algoritma Support Vector Machine (SVM) | Ekstraksi Fitur : TF-IDF | Dataset : Google Play Store – Aplikasi JogjaKita
</div>
""", unsafe_allow_html=True)
