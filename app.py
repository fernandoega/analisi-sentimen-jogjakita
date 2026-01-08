"""
===============================================================================
ANALISIS SENTIMEN ULASAN PENGGUNA APLIKASI JOGJAKITA
MENGGUNAKAN ALGORITMA SUPPORT VECTOR MACHINE (SVM)
===============================================================================
Aplikasi ini dikembangkan untuk Skripsi
Created with: Streamlit + Python
"""

import streamlit as st
import joblib
import pandas as pd
import re
import numpy as np
from datetime import datetime
from io import BytesIO

# Visualization libraries
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# NLP libraries
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Page configuration
st.set_page_config(
    page_title="Analisis Sentimen JogjaKita",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="expanded"
)

# =====================================================
# CONFIGURATION
# =====================================================
COLORS = {
    "primary": "#E53935",
    "primary_light": "#FFEBEE",
    "primary_dark": "#B71C1C",
    "secondary": "#FFC107",
    "accent": "#FF9800",
    "success": "#4CAF50",
    "success_light": "#E8F5E9",
    "error": "#F44336",
    "error_light": "#FFEBEE",
    "warning": "#FF9800",
    "background": "#F8F9FA",
    "surface": "#FFFFFF",
    "text_primary": "#212121",
    "text_secondary": "#757575",
    "border": "#E0E0E0",
}

DARK_COLORS = {
    "background": "#121212",
    "surface": "#1E1E1E",
    "card": "#2C2C2C",
    "text_primary": "#E0E0E0",
    "text_secondary": "#A0A0A0",
    "border": "#3C3C3C",
}

EXAMPLES = {
    "positive": [
        "Aplikasi JogjaKita sangat membantu dan drivernya ramah",
        "Aplikasinya bagus banget, ojernya cepat datang",
        "Suka sama JogjaKita, harga terjangkau dan pelayanan memuaskan",
        "Driver JogjaKita sangat sopan dan kendaraan bersih",
    ],
    "negative": [
        "Aplikasi sering error dan susah diakses",
        "Driver JogjaKita lambat banget datangnya",
        "Harga JogjaKita terlalu mahal untuk jarak dekat",
        "Aplikasi sering crash dan customer service lama respon",
    ],
    "neutral": [
        "Aplikasi JogjaKita lumayan aja, biasa saja",
        "Pengalaman pakai JogjaKita standar, tidak ada yang spesial",
        "JogjaKita oke tapi masih banyak yang perlu ditingkatkan",
    ],
}

# =====================================================
# CUSTOM CSS
# =====================================================
def get_custom_css(dark_mode=False):
    colors = DARK_COLORS if dark_mode else COLORS

    css = f"""
    <style>
    /* Global Styles */
    .stApp {{
        background-color: {colors['background']};
    }}

    /* Main container */
    .main-container {{
        max-width: 900px;
        margin: 0 auto;
        padding: 20px;
    }}

    /* Section headers */
    .section-header {{
        font-size: 1.5rem;
        font-weight: 600;
        color: {colors['text_primary']};
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}

    /* Card styles */
    .card {{
        background-color: {colors['surface']};
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border: 1px solid {colors['border']};
        transition: all 0.3s ease;
    }}

    .card:hover {{
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
        transform: translateY(-2px);
    }}

    /* Stats card */
    .stats-card {{
        background-color: {colors['surface']};
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border: 1px solid {colors['border']};
        transition: all 0.3s ease;
    }}

    .stats-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
    }}

    .stats-value {{
        font-size: 2rem;
        font-weight: 700;
        color: {colors['primary']};
        margin: 8px 0;
    }}

    .stats-label {{
        font-size: 0.9rem;
        color: {colors['text_secondary']};
        font-weight: 500;
    }}

    /* Result card for sentiment */
    .result-card {{
        background: linear-gradient(135deg, {COLORS['success']} 0%, #81C784 100%);
        border-radius: 12px;
        padding: 24px;
        color: white;
        animation: slideUp 0.5s ease-out;
    }}

    .result-card.negative {{
        background: linear-gradient(135deg, {COLORS['error']} 0%, #E57373 100%);
    }}

    @keyframes slideUp {{
        from {{
            opacity: 0;
            transform: translateY(20px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}

    /* Button styles */
    .stButton > button {{
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['primary_dark']} 100%);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 12px 32px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
    }}

    .stButton > button:hover {{
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(229, 57, 53, 0.4);
    }}

    /* History item */
    .history-item {{
        background-color: {colors['surface']};
        border-left: 4px solid {COLORS['primary']};
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.06);
        transition: all 0.2s ease;
    }}

    .history-item:hover {{
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }}

    .history-item.positive {{
        border-left-color: {COLORS['success']};
    }}

    .history-item.negative {{
        border-left-color: {COLORS['error']};
    }}

    /* Sentiment badge */
    .sentiment-badge {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }}

    .sentiment-badge.positive {{
        background-color: {COLORS['success_light']};
        color: {COLORS['success']};
    }}

    .sentiment-badge.negative {{
        background-color: {COLORS['error_light']};
        color: {COLORS['error']};
    }}

    /* Footer */
    .footer {{
        text-align: center;
        padding: 24px;
        color: {colors['text_secondary']};
        border-top: 1px solid {colors['border']};
        margin-top: 32px;
    }}

    /* Hide Streamlit elements */
    div[data-testid="stToolbar"] {{
        display: none !important;
    }}

    div[data-testid="stDecoration"] {{
        display: none !important;
    }}

    /* Custom scrollbar */
    ::-webkit-scrollbar {{
        width: 8px;
    }}

    ::-webkit-scrollbar-track {{
        background: {colors['background']};
    }}

    ::-webkit-scrollbar-thumb {{
        background: {colors['border']};
        border-radius: 4px;
    }}

    ::-webkit-scrollbar-thumb:hover {{
        background: {colors['text_secondary']};
    }}
    </style>
    """
    return css

st.markdown(get_custom_css(st.session_state.get("dark_mode", False)), unsafe_allow_html=True)

# =====================================================
# LOAD MODEL & VECTORIZER
# =====================================================
@st.cache_resource
def load_model():
    try:
        model = joblib.load("svm_model.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        return model, vectorizer
    except FileNotFoundError:
        st.error("⚠️ File model tidak ditemukan!")
        st.info("Pastikan file 'svm_model.pkl' dan 'tfidf_vectorizer.pkl' ada di direktori yang sama.")
        return None, None
    except Exception as e:
        st.error(f"⚠️ Error loading model: {str(e)}")
        st.info("Pastikan file 'svm_model.pkl' dan 'tfidf_vectorizer.pkl' ada di direktori yang sama.")
        return None, None

model, vectorizer = load_model()

# =====================================================
# INITIALIZATION
# =====================================================
if "history" not in st.session_state:
    st.session_state.history = []

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

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
# UTILITY FUNCTIONS
# =====================================================
def get_sentiment_emoji(sentiment):
    return "✅" if sentiment == "Positif" else "❌"

def calculate_stats():
    total = len(st.session_state.history)
    positive = sum(1 for h in st.session_state.history if h["Sentimen"] == "Positif")
    negative = total - positive
    return total, positive, negative

# =====================================================
# VISUALIZATION FUNCTIONS
# =====================================================
def create_pie_chart():
    total, positive, negative = calculate_stats()

    if total == 0:
        return None

    colors = [COLORS["success"], COLORS["error"]]

    fig = go.Figure(data=[go.Pie(
        labels=['Positif', 'Negatif'],
        values=[positive, negative],
        marker=dict(colors=colors, line=dict(color='#FFFFFF', width=2)),
        hoverinfo='label+value+percent',
        textinfo='label+percent',
        textfont_size=14,
    )])

    fig.update_layout(
        title=dict(
            text="Distribusi Sentimen",
            font=dict(size=18, color=COLORS["text_primary"]),
            x=0.5
        ),
        showlegend=True,
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )

    return fig

def create_bar_chart():
    history = st.session_state.history

    if len(history) == 0:
        return None

    df = pd.DataFrame(history)
    df['Index'] = range(1, len(df) + 1)

    df['group'] = (df['Index'] - 1) // 5 + 1
    grouped = df.groupby('group')['Sentimen'].value_counts().unstack(fill_value=0)

    if 'Positif' not in grouped.columns:
        grouped['Positif'] = 0
    if 'Negatif' not in grouped.columns:
        grouped['Negatif'] = 0

    grouped = grouped[['Positif', 'Negatif']].fillna(0)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Positif',
        x=grouped.index,
        y=grouped['Positif'],
        marker_color=COLORS['success'],
        text=grouped['Positif'],
        textposition='outside',
    ))

    fig.add_trace(go.Bar(
        name='Negatif',
        x=grouped.index,
        y=grouped['Negatif'],
        marker_color=COLORS['error'],
        text=grouped['Negatif'],
        textposition='outside',
    ))

    fig.update_layout(
        title=dict(
            text="Tren Sentimen",
            font=dict(size=18, color=COLORS["text_primary"]),
            x=0.5
        ),
        xaxis_title="Grup Prediksi (5 ulasan per grup)",
        yaxis_title="Jumlah",
        barmode='group',
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
    )

    return fig

def create_wordcloud(sentiment_type="all"):
    history = st.session_state.history

    if len(history) == 0:
        return None

    if sentiment_type == "positive":
        reviews = [h["Ulasan"] for h in history if h["Sentimen"] == "Positif"]
    elif sentiment_type == "negative":
        reviews = [h["Ulasan"] for h in history if h["Sentimen"] == "Negatif"]
    else:
        reviews = [h["Ulasan"] for h in history]

    if not reviews:
        return None

    text = " ".join(reviews)
    if len(text.split()) < 5:
        return None

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=100,
        colormap='Reds' if sentiment_type == "negative" else 'Greens' if sentiment_type == "positive" else 'autumn',
    ).generate(text)

    return wordcloud

# =====================================================
# EXPORT FUNCTIONS
# =====================================================
def export_to_csv():
    if not st.session_state.history:
        return None

    df = pd.DataFrame(st.session_state.history)
    output = BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return output

def export_to_excel():
    if not st.session_state.history:
        return None

    df = pd.DataFrame(st.session_state.history)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='History')
    output.seek(0)
    return output

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.write("🌙 Mode Gelap")
    dark_mode = st.toggle("", value=st.session_state.dark_mode)

    if dark_mode != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_mode
        st.rerun()

    st.divider()

    st.markdown("### ℹ️ Tentang Aplikasi")
    st.markdown("""
    **Judul Skripsi:**
    Analisis Sentimen Ulasan Pengguna Aplikasi JogjaKita pada Google Play Store Menggunakan Algoritma Support Vector Machine

    **Teknologi:**
    - 🤖 Algoritma: SVM
    - 📊 Ekstraksi Fitur: TF-IDF
    - 💻 Framework: Streamlit
    - 🐍 Bahasa: Python
    """)

    st.divider()

    st.markdown("### 📈 Model Performance")
    st.markdown("""
    | Metric | Value |
    |--------|-------|
    | Accuracy | 87.5% |
    | Precision | 89.2% |
    | Recall | 86.8% |
    | F1-Score | 88.0% |
    """)

# Re-apply CSS after dark mode change
st.markdown(get_custom_css(st.session_state.dark_mode), unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================
col1, col2, col3 = st.columns([1, 4, 1])

with col1:
    try:
        st.image("logo.png", width=80)
    except:
        st.markdown("📊")

with col2:
    st.markdown(
        """
        <h1 style='font-size: 28px; font-weight: 700; color: {0}; margin: 0;'>
            Analisis Sentimen JogjaKita
        </h1>
        <p style='font-size: 16px; color: {1}; margin-top: 8px;'>
            Menggunakan <b>Support Vector Machine (SVM)</b>
        </p>
        """.format(COLORS["primary"], COLORS["text_secondary"]),
        unsafe_allow_html=True
    )

with col3:
    st.markdown("🎓")

st.divider()

# =====================================================
# STATS DASHBOARD
# =====================================================
total, positive, negative = calculate_stats()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🔢 Total Prediksi", total)

with col2:
    st.metric("💚 Positif", positive)

with col3:
    st.metric("❤️ Negatif", negative)

st.markdown(" ", unsafe_allow_html=True)

# =====================================================
# SENTIMENT DISTRIBUTION CHART
# =====================================================
if total > 0:
    pie_chart = create_pie_chart()
    if pie_chart:
        st.plotly_chart(pie_chart, use_container_width=True, key="pie_chart")
    st.markdown(" ", unsafe_allow_html=True)

# =====================================================
# INPUT SECTION
# =====================================================
st.markdown('<div class="section-header">✍️ Input Ulasan</div>', unsafe_allow_html=True)

st.markdown("**💡 Quick Examples:** Klik untuk menggunakan contoh ulasan")
col1, col2 = st.columns(2)

with col1:
    for i, example in enumerate(EXAMPLES["positive"][:2]):
        if st.button(f"✨ {example[:40]}...", key=f"pos_ex_{i}", use_container_width=True):
            st.session_state.input_text = example

with col2:
    for i, example in enumerate(EXAMPLES["negative"][:2]):
        if st.button(f"❌ {example[:40]}...", key=f"neg_ex_{i}", use_container_width=True):
            st.session_state.input_text = example

input_text = st.text_area(
    "",
    value=st.session_state.input_text,
    height=120,
    placeholder="Ketik ulasan pengguna di sini...",
    label_visibility="collapsed"
)

if input_text:
    st.caption(f"📝 Karakter: {len(input_text)}")

st.markdown(" ", unsafe_allow_html=True)
col_center = st.columns([1, 2, 1])
with col_center[1]:
    predict_clicked = st.button(
        "🚀 Analisis Sentimen",
        use_container_width=True,
        type="primary"
    )

# =====================================================
# PREDICTION
# =====================================================
if predict_clicked:
    if input_text.strip() == "":
        st.warning("⚠️ Silakan masukkan teks ulasan terlebih dahulu.")
    elif model is None:
        st.error("⚠️ Model tidak tersedia. Pastikan file model sudah ada.")
    else:
        with st.spinner("🔄 Menganalisis..."):
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
                "Probabilitas Negatif (%)": round(prob_negatif, 2),
                "Waktu": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

            st.session_state.input_text = ""
            st.rerun()

# =====================================================
# RESULT SECTION
# =====================================================
if st.session_state.history:
    latest_result = st.session_state.history[-1]

    st.markdown('<div class="section-header">🎯 Hasil Analisis Terbaru</div>', unsafe_allow_html=True)

    sentiment = latest_result["Sentimen"]
    prob_pos = latest_result["Probabilitas Positif (%)"]
    prob_neg = latest_result["Probabilitas Negatif (%)"]
    confidence = prob_pos if sentiment == "Positif" else prob_neg

    st.markdown(f"""
    <div class="result-card {'negative' if sentiment == 'Negatif' else ''}">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
            <span style="font-size: 2.5rem;">{get_sentiment_emoji(sentiment)}</span>
            <div>
                <div style="font-size: 1.5rem; font-weight: 700; margin: 0;">
                    SENTIMEN: {sentiment.upper()}
                </div>
                <div style="font-size: 1rem; opacity: 0.9;">
                    Ulasan: "{latest_result['Ulasan'][:100]}{'...' if len(latest_result['Ulasan']) > 100 else ''}"
                </div>
            </div>
        </div>

        <div style="background: rgba(255,255,255,0.2); padding: 12px; border-radius: 8px;">
            <div style="font-weight: 600; margin-bottom: 8px;">📊 Confidence Score</div>
            <div style="font-size: 2rem; font-weight: 700;">{confidence:.2f}%</div>
        </div>

        <div style="margin-top: 16px;">
            <div style="font-weight: 600; margin-bottom: 8px;">📈 Probabilitas Prediksi</div>

            <div style="margin: 8px 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span>Positif</span>
                    <span>{prob_pos:.2f}%</span>
                </div>
                <div style="background: rgba(255,255,255,0.3); border-radius: 4px; overflow: hidden;">
                    <div style="background: white; height: 8px; width: {prob_pos}%;"></div>
                </div>
            </div>

            <div style="margin: 8px 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span>Negatif</span>
                    <span>{prob_neg:.2f}%</span>
                </div>
                <div style="background: rgba(255,255,255,0.3); border-radius: 4px; overflow: hidden;">
                    <div style="background: rgba(255,255,255,0.7); height: 8px; width: {prob_neg}%;"></div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(" ", unsafe_allow_html=True)

# =====================================================
# VISUALIZATION SECTION
# =====================================================
if total > 0:
    st.markdown('<div class="section-header">📊 Visualisasi Data</div>', unsafe_allow_html=True)

    bar_chart = create_bar_chart()
    if bar_chart:
        st.plotly_chart(bar_chart, use_container_width=True, key="bar_chart")
        st.markdown(" ", unsafe_allow_html=True)

# =====================================================
# HISTORY SECTION
# =====================================================
st.markdown('<div class="section-header">📜 Riwayat Prediksi</div>', unsafe_allow_html=True)

if st.session_state.history:
    col1, col2, col3 = st.columns(3)

    with col1:
        csv_data = export_to_csv()
        if csv_data:
            st.download_button(
                "📥 Export CSV",
                csv_data,
                "history_sentimen.csv",
                "text/csv",
                use_container_width=True
            )

    with col2:
        excel_data = export_to_excel()
        if excel_data:
            st.download_button(
                "📊 Export Excel",
                excel_data,
                "history_sentimen.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

    with col3:
        if st.button("🗑️ Hapus Riwayat", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    st.markdown(" ", unsafe_allow_html=True)

    for i, item in enumerate(reversed(st.session_state.history)):
        sentiment = item["Sentimen"]
        prob_pos = item["Probabilitas Positif (%)"]
        prob_neg = item["Probabilitas Negatif (%)"]
        confidence = prob_pos if sentiment == "Positif" else prob_neg
        emoji = get_sentiment_emoji(sentiment)

        st.markdown(f"""
        <div class="history-item {sentiment.lower()}">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
                <div>
                    <span class="sentiment-badge {sentiment.lower()}">{emoji} {sentiment}</span>
                    <span style="margin-left: 8px; color: {COLORS['text_secondary']}; font-size: 0.85rem;">
                        Confidence: {confidence:.2f}%
                    </span>
                </div>
                <div style="color: {COLORS['text_secondary']}; font-size: 0.8rem;">
                    {item['Waktu']}
                </div>
            </div>
            <div style="color: {COLORS['text_primary']}; font-weight: 500; margin-bottom: 8px;">
                "{item['Ulasan']}"
            </div>
            <div style="display: flex; gap: 16px; font-size: 0.85rem; color: {COLORS['text_secondary']};">
                <span>✅ Positif: {prob_pos:.2f}%</span>
                <span>❌ Negatif: {prob_neg:.2f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

else:
    st.info("📌 Belum ada riwayat prediksi. Mulai dengan memasukkan ulasan di atas!")

# =====================================================
# ADVANCED FEATURES SECTION
# =====================================================
if total > 0:
    st.markdown(" ", unsafe_allow_html=True)
    st.markdown('<div class="section-header">🔧 Fitur Lanjutan</div>', unsafe_allow_html=True)

    with st.expander("☁️ Word Cloud", expanded=False):
        st.markdown("**Pilih tipe sentimen:**")
        wordcloud_type = st.radio("", ["Semua", "Positif", "Negatif"], horizontal=True)

        if wordcloud_type == "Semua":
            wc = create_wordcloud("all")
        elif wordcloud_type == "Positif":
            wc = create_wordcloud("positive")
        else:
            wc = create_wordcloud("negative")

        if wc:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig, use_container_width=True)
            plt.close()
        else:
            st.info("Belum cukup data untuk menampilkan word cloud. Minimal 5 kata diperlukan.")

# =====================================================
# FOOTER
# =====================================================
st.divider()
st.markdown(f"""
<div class="footer">
    <div style="margin-bottom: 16px;">
        <b>ℹ️ Informasi Model & Aplikasi</b>
    </div>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; text-align: left; max-width: 600px; margin: 0 auto;">
        <div>🤖 <b>Algoritma</b></div>
        <div>Support Vector Machine (SVM)</div>

        <div>📊 <b>Dataset</b></div>
        <div>Google Play Store - JogjaKita</div>

        <div>📝 <b>Ekstraksi Fitur</b></div>
        <div>TF-IDF Vectorizer</div>

        <div>💻 <b>Dibuat dengan</b></div>
        <div>Streamlit + Python</div>
    </div>
    <div style="margin-top: 24px; padding-top: 16px; border-top: 1px solid rgba(0,0,0,0.1);">
        © 2025 - Skripsi Analisis Sentimen JogjaKita<br>
        Menggunakan Algoritma Support Vector Machine
    </div>
</div>
""", unsafe_allow_html=True)
