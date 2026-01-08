
# =====================================================
st.markdown("""
<div class="card">
<b>Deskripsi Sistem</b><br><br>
Sistem ini menganalisis sentimen ulasan pengguna aplikasi <b>JogjaKita</b>
menjadi <b>positif</b> atau <b>negatif</b> menggunakan algoritma
<b>Support Vector Machine (SVM)</b> dengan fitur <b>TF-IDF</b>.
</div>
""", unsafe_allow_html=True)

# =====================================================
# INPUT
# =====================================================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>📝 Masukkan Ulasan Pengguna</div>", unsafe_allow_html=True)

input_text = st.text_area(
    "",
    height=120,
    placeholder="Contoh: Aplikasi JogjaKita sangat membantu dan drivernya ramah",
    label_visibility="collapsed"
)

st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# PREDIKSI
# =====================================================
if st.button("🔍 Prediksi Sentimen"):
    if input_text.strip() == "":
        st.warning("Silakan masukkan teks ulasan.")
    else:
        clean = preprocess(input_text)
        vec = vectorizer.transform([clean])
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]

        label = "Positif" if pred == 1 else "Negatif"
        pos, neg = proba[1]*100, proba[0]*100

        st.session_state.history.append({
            "Ulasan": input_text,
            "Sentimen": label,
            "Positif (%)": round(pos,2),
            "Negatif (%)": round(neg,2),
            "Waktu": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>📌 Hasil Prediksi</div>", unsafe_allow_html=True)

        if label == "Positif":
            st.markdown(f"<div class='result-positive'>✅ <b>Sentimen Positif</b><br>Probabilitas: {pos:.2f}%</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result-negative'>❌ <b>Sentimen Negatif</b><br>Probabilitas: {neg:.2f}%</div>", unsafe_allow_html=True)

        st.progress(pos/100 if label=="Positif" else neg/100)
        st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# RIWAYAT
# =====================================================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>📂 Riwayat Prediksi</div>", unsafe_allow_html=True)

if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True)

    if st.button("🗑️ Hapus Riwayat"):
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
Algoritma: Support Vector Machine (Kernel RBF)  
Fitur: TF-IDF  
Dataset: Google Play Store – JogjaKita
""")
