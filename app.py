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
col1, col2 = st.columns([1, 5])

with col1:
    st.image("logo.png", width=120)

with col2:
    st.markdown(
        """
        <h2>Analisis Sentimen Ulasan JogjaKita</h2>
        <p>Menggunakan Algoritma <b>Support Vector Machine (SVM)</b> </p>
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
st.subheader("🗂️ Riwayat Prediksi")

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
st.divider()
st.caption("""
ℹ️ **Informasi Model**
- Algoritma : Support Vector Machine (Kernel RBF)
- Ekstraksi Fitur : TF-IDF (Unigram & Bigram)
- Dataset : Google Play Store – Aplikasi JogjaKita
""")
