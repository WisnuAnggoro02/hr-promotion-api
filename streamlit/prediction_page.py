import streamlit as st

from api_client import fetch_schema, predict


def run_prediction_page(api_url: str) -> None:
    st.title("Promotion Prediction")
    st.caption("Isi data karyawan lalu klik Predict. Form dibangun dari `/schema` API.")
    st.divider()

    try:
        schema = fetch_schema(api_url)
    except Exception as e:
        st.error(f"Tidak bisa mengambil schema dari `{api_url}`.\n\n{e}")
        st.info("Pastikan API sudah jalan: `python api/index.py`")
        return

    cat, rng = schema["categories"], schema["numeric_ranges"]

    col1, col2 = st.columns(2)
    with col1:
        department = st.selectbox("Department", cat["department"])
        region = st.selectbox("Region", cat["region"])
        education = st.selectbox("Education", cat["education"])
        gender = st.radio("Gender", cat["gender"], horizontal=True)
        recruitment = st.selectbox("Recruitment Channel", cat["recruitment_channel"])
        awards = st.radio("Awards Won", ["No", "Yes"], horizontal=True)
    with col2:
        age = st.slider("Age", *rng["age"], 30)
        service = st.slider("Length of Service (years)", *rng["length_of_service"], 5)
        trainings = st.slider("No. of Trainings", *rng["no_of_trainings"], 2)
        rating = st.slider("Previous Year Rating", *rng["previous_year_rating"], 3)
        avg_score = st.slider("Avg Training Score", *rng["avg_training_score"], 60)

    st.divider()

    if st.button("Predict Promotion", type="primary", width="stretch"):
        payload = {
            "department": department, "region": region, "education": education,
            "gender": gender, "recruitment_channel": recruitment,
            "no_of_trainings": trainings, "age": age,
            "previous_year_rating": rating, "length_of_service": service,
            "awards_won": 1 if awards == "Yes" else 0,
            "avg_training_score": avg_score,
        }
        try:
            result = predict(api_url, payload)
        except Exception as e:
            st.error(f"Prediksi gagal: {e}")
            return

        if result["prediction"] == 1:
            st.success("**Promoted** — kemungkinan besar dipromosikan.")
        else:
            st.warning("**Not Promoted** — kecil kemungkinan dipromosikan.")
        c1, c2 = st.columns(2)
        c1.metric("Promotion Probability", f"{result['probability'] * 100:.1f}%")
        c2.metric("Decision Threshold", f"{result['threshold'] * 100:.1f}%")
        st.progress(min(int(result["probability"] * 100), 100))
        with st.expander("Request / response (untuk debugging)"):
            st.json({"request": payload, "response": result})
