import streamlit as st
st.image("logo.png", width=80)
with col2:
st.markdown(
"""
<h2 style="margin-bottom:6px;">Analisis Sentimen Ulasan JogjaKita</h2>
<p style="margin-top:0; color:#555;">Menggunakan Algoritma <b>Support Vector Machine (SVM)</b></p>
""",
unsafe_allow_html=True
)


st.markdown("<hr>", unsafe_allow_html=True)


# ================= INPUT =================
st.markdown("<h4 class='section-title'>📝 Masukkan Ulasan Pengguna</h4>", unsafe_allow_html=True)
input_text = st.text_area(
"Contoh: Aplikasi JogjaKita sangat membantu dan drivernya ramah",
height=120
)


# ================= PREDIKSI =================
if st.button("🔍 Prediksi Sentimen"):
if input_text.strip() == "":
st.warning("Silakan masukkan teks ulasan terlebih dahulu.")
else:
clean_text = preprocess(input_text)
vector = vectorizer.transform([clean_text])
pred_label = model.predict(vector)[0]


if hasattr(model, "predict_proba"):
proba = model.predict_proba(vector)[0]
prob_negatif = proba[0] * 100
prob_positif = proba[1] * 100
else:
prob_negatif = prob_positif = None


label_text = "Positif" if pred_label == 1 else "Negatif"
st.session_state.history.append({
"Ulasan": input_text,
"Sentimen": label_text,
"Probabilitas Positif (%)": round(prob_positif, 2) if prob_positif is not None else "-",
"Probabilitas Negatif (%)": round(prob_negatif, 2) if prob_negatif is not None else "-",
})


st.markdown("<hr>", unsafe_allow_html=True)
st.subheader("📌 Hasil Prediksi")
st.success("✅ Sentimen Positif" if pred_label == 1 else "❌ Sentimen Negatif")


if prob_positif is not None:
st.markdown("### 📈 Probabilitas Prediksi")
st.write(f"**Positif : {prob_positif:.2f}%**")
st.progress(prob_positif / 100)
st.write(f"**Negatif : {prob_negatif:.2f}%**")
st.progress(prob_negatif / 100)


# ================= RIWAYAT =================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h4 class='section-title'>📂 Riwayat Prediksi</h4>", unsafe_allow_html=True)


if len(st.session_state.history) > 0:
df_history = pd.DataFrame(st.session_state.history)
st.dataframe(df_history, use_container_width=True)
if st.button("🧹 Hapus Riwayat"):
st.session_state.history = []
st.rerun()
else:
st.info("Belum ada riwayat prediksi.")


# ================= FOOTER =================
st.markdown("<hr>", unsafe_allow_html=True)
st.caption(
"""
ℹ️ **Informasi Model**
- Algoritma : Support Vector Machine (Kernel RBF)
- Ekstraksi Fitur : TF-IDF (Unigram & Bigram)
- Dataset : Google Play Store – Aplikasi JogjaKita
"""
)


# =====================================================
# CARD END
# =====================================================
st.markdown("</div>", unsafe_allow_html=True)
