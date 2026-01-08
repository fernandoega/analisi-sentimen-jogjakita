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
    page_icon="😊", # Menggunakan emoji agar lebih universal
    layout="centered",
    initial_sidebar_state="collapsed"
)

# =====================================================
# STYLE (CSS) - DIPERBARUI
# =====================================================
st.markdown(
    """
    <style>
    /* Font Import (Opsional, untuk font yang lebih menarik) */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

    /* General Styling */
    .stApp {
        background-color: #f0f2f6; /* Background abu-abu muda yang lebih lembut */
        font-family: 'Poppins', sans-serif; /* Menggunakan font Poppins */
    }
    
    /* Semua teks */
    h1, h2, h3, h4, h5, h6, p, span, label {
        color: #333333 !important;
    }

    /* Gaya untuk "Kartu" */
    .card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        margin-bottom: 25px;
    }

    /* Gaya Khusus untuk Kartu Header */
    .hero-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); /* Gradien ungu-biru */
        color: white;
        text-align: center;
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

    /* Gaya untuk Tombol Utama */
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
    
    /* Gaya untuk Tombol Sekunder (Hapus Riwayat) */
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

    /* Text area & input */
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
    }
    .result-positive {
        background-color: #c6f6d5;
        color: #22543d;
        font-size: 24px;
        font-weight: 600;
    }
    .result-negative {
        background-color: #fed7d7;
        color: #742a2a;
        font-size: 24px;
        font-weight: 600;
    }
    
    /* Progress Bar Styling */
    .stProgress > div > div > div > div {
        background-color: #667eea;
    }
    
    /* Styling Tabel Riwayat */
    .dataframe-container {
        max-height: 400px;
        overflow-y: auto;
    }
    .dataframe-container div[data-testid="stVerticalBlock"] {
        padding-top: 0;
    }
    
    /* Hilangkan elemen default Streamlit */
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
# HEADER - DIPERBARUI MENGGUNAKAN KARTU
# =====================================================
st.markdown("""
<div class="card hero-card">
    <h1>Analisis Sentimen Ulasan JogjaKita</h1>
    <p class="subtitle">Menggunakan Algoritma <b>Support Vector Machine (SVM)</b> untuk menganalisis pendapat pengguna.</p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# INPUT TEKS - DIPERBARUI MENGGUNAKAN KARTU
# =====================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### Masukkan Ulasan Pengguna")

input_text = st.text_area(
    "Contoh: Aplikasi JogjaKita sangat membantu dan drivernya ramah",
    height=120,
    placeholder="Ketik ulasan pengguna di sini...",
    label_visibility="collapsed"
)

# Kolom untuk memusatkan tombol
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

            # Simpan riwayat
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
if st.session_state.history: # Cek jika ada prediksi yang sudah dilakukan
    last_result = st.session_state.history[-1]
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Hasil Prediksi")
    
    # Tampilkan hasil dengan gaya kustom
    result_icon = "😊" if last_result["Sentimen"] == "Positif" else "😞"
    result_class = "result-positive" if last_result["Sentimen"] == "Positif" else "result-negative"
    
    st.markdown(f"""
    <div class="result-container {result_class}">
        {result_icon} Sentimen {last_result['Sentimen']}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Probabilitas Prediksi")
    
    # Menggunakan metrik untuk tampilan yang lebih rapi
    col_pos, col_neg = st.columns(2)
    with col_pos:
        st.metric(label="Positif", value=f"{last_result['Probabilitas Positif (%)']:.2f}%")
        st.progress(last_result['Probabilitas Positif (%)'] / 100)
        
    with col_neg:
        st.metric(label="Negatif", value=f"{last_result['Probabilitas Negatif (%)']:.2f}%")
        st.progress(last_result['Probabilitas Negatif (%)'] / 100)
        
    st.markdown('</div>', unsafe_allow_html=True)


# =====================================================
# RIWAYAT PREDIKSI - TABEL DIPERBARUI
# =====================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### Riwayat Prediksi")

if st.session_state.history:
    df_history = pd.DataFrame(st.session_state.history)
    
    # Mewarnai tabel berdasarkan sentimen
    def highlight_sentiment(s):
        is_positive = s['Sentimen'] == 'Positif'
        return [
            'background-color: #c6f6d5' if is_positive else 'background-color: #fed7d7'
            for v in is_positive
        ]
    
    styled_df = df_history.style.apply(highlight_sentiment, axis=1)
    
    st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
    st.dataframe(styled_df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Tombol hapus riwayat dengan kustomisasi
    st.markdown('<div class="secondary-btn-container">', unsafe_allow_html=True)
    if st.button("🗑️ Hapus Riwayat"):
        st.session_state.history = []
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Belum ada riwayat prediksi.", icon="ℹ️")
    
st.markdown('</div>', unsafe_allow_html=True)


# =====================================================
# FOOTER - DIPERBARUI
# =====================================================
st.markdown("""
<div class="card" style="text-align: center; background-color: #edf2f7;">
    <p style="font-size: 14px; color: #4a5568;">
        ℹ️ <b>Informasi Model:</b> Support Vector Machine (SVM) dengan TF-IDF | Dataset: Ulasan Aplikasi JogjaKita di Google Play Store
    </p>
</div>
""", unsafe_allow_html=True)
