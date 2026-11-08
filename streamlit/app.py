"""Router Streamlit. Jalankan dari root project:  streamlit run streamlit/app.py"""
import streamlit as st

import api_client
from prediction_page import run_prediction_page
from profile_page import run_profile_page

st.set_page_config(page_title="HR Promotion Predictor", page_icon="🏆", layout="centered")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Prediction", "Employee Profiles"])

st.sidebar.divider()
st.sidebar.subheader("Pengaturan API")
api_url = st.sidebar.text_input("API URL", api_client.default_api_url()).rstrip("/")
if st.sidebar.button("Cek koneksi"):
    try:
        st.sidebar.success("API terhubung")
        st.sidebar.json(api_client.health(api_url))
    except Exception as e:
        st.sidebar.error(f"Gagal: {e}")

if page == "Home":
    st.title("Employee Promotion Predictor")
    st.caption("Streamlit (UI) → Flask API (model) → hasil prediksi")
    st.divider()
    st.markdown(
        """
        UI ini **tidak memuat model sendiri**. Semua prediksi dikirim ke Flask API
        lewat HTTP (`POST /predict`), sehingga model bisa dijalankan di mana saja:
        lokal, tunnel ngrok, atau Vercel. Cukup ganti **API URL** di sidebar.

        1. **Prediction** — isi form manual lalu prediksi.
        2. **Employee Profiles** — pilih karyawan dari `employees.csv` lalu prediksi.
        """
    )
    try:
        info = api_client.health(api_url)
        c1, c2, c3 = st.columns(3)
        c1.metric("API", "Online")
        c2.metric("ROC-AUC", info["metrics"]["roc_auc"])
        c3.metric("Threshold", f"{info['threshold']:.2f}")
    except Exception:
        st.warning(f"API di `{api_url}` belum terhubung. Jalankan `python api/index.py`.")
elif page == "Prediction":
    run_prediction_page(api_url)
else:
    run_profile_page(api_url)
