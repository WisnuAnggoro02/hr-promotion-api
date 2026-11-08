"""Halaman Employee Profiles: data dari employees.csv, prediksi lewat POST /predict."""
import html
from pathlib import Path

import pandas as pd
import streamlit as st

from api_client import predict

CSV_PATH = Path(__file__).parent / "employees.csv"

# Kolom CSV yang dikirim ke API (harus sama dengan `features` di metadata.json)
FEATURES = [
    "department", "region", "education", "gender", "recruitment_channel",
    "no_of_trainings", "age", "previous_year_rating", "length_of_service",
    "awards_won", "avg_training_score",
]
STR_FEATURES = {"department", "region", "education", "gender", "recruitment_channel"}


@st.cache_data
def load_employees() -> pd.DataFrame:
    return pd.read_csv(CSV_PATH)


def to_payload(emp: pd.Series) -> dict:
    """Ubah satu baris CSV menjadi JSON body untuk /predict (tipe Python murni)."""
    return {f: (str(emp[f]) if f in STR_FEATURES else int(emp[f])) for f in FEATURES}


def render_html(content: str, height: int) -> None:
    """st.iframe (Streamlit baru); components.html sudah deprecated. Semua data sudah di-escape."""
    if hasattr(st, "iframe"):
        st.iframe(content, height=height)
    else:  # Streamlit lama
        import streamlit.components.v1 as components
        components.html(content, height=height)


# ── Komponen HTML ──────────────────────────────────────────────────────────────
def profile_card_html(emp: pd.Series) -> str:
    e = lambda v: html.escape(str(v))
    initials = "".join(w[0] for w in str(emp["name"]).split()[:2]).upper()
    awards = "Yes" if int(emp["awards_won"]) == 1 else "No"
    region = str(emp["region"]).replace("_", " ").title()
    stats = [("#48cae4", emp["no_of_trainings"], "Trainings"),
             ("#6c63ff", emp["previous_year_rating"], "Rating"),
             ("#f4a261", emp["length_of_service"], "Yrs Service"),
             ("#2ec4b6", emp["avg_training_score"], "Avg Score")]
    stats_html = "".join(
        f'<div class="stat" style="background:{c};"><div class="stat-v">{e(v)}</div>'
        f'<div class="stat-l">{l}</div></div>' for c, v, l in stats)
    return f"""<!DOCTYPE html><html><head><style>
* {{ box-sizing:border-box; margin:0; padding:0;
     font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; }}
body {{ background:transparent; padding:2px 0; }}
.card {{ background:white; border-radius:16px; padding:22px 24px;
  box-shadow:0 2px 12px rgba(0,0,0,.09); display:flex; flex-direction:column;
  align-items:center; text-align:center; gap:10px; }}
.avatar {{ width:72px; height:72px; border-radius:50%;
  background:linear-gradient(135deg,#6c63ff,#48cae4); display:flex;
  align-items:center; justify-content:center; font-size:26px; font-weight:700; color:white; }}
.name {{ font-size:20px; font-weight:700; color:#1a1a2e; }}
.role {{ font-size:13px; color:#6b7280; margin-top:3px; }}
.dept {{ font-size:12px; color:#9ca3af; margin-top:1px; }}
.meta {{ font-size:12px; color:#9ca3af; }}
.stats {{ display:flex; gap:10px; width:100%; margin-top:6px; }}
.stat {{ flex:1; border-radius:10px; padding:12px 8px; text-align:center; }}
.stat-v {{ font-size:22px; font-weight:700; color:white; }}
.stat-l {{ font-size:10px; color:rgba(255,255,255,.85); margin-top:2px; }}
</style></head><body><div class="card">
  <div class="avatar">{e(initials)}</div>
  <div><div class="name">{e(emp['name'])}</div><div class="role">{e(emp['position'])}</div>
    <div class="dept">{e(emp['department'])} &middot; {e(region)}</div></div>
  <div class="meta">Age {e(emp['age'])} &middot; {e(emp['gender'])} &middot; {e(emp['education'])}
    &nbsp;|&nbsp; &#127941; Award: <strong>{awards}</strong>
    &nbsp;|&nbsp; &#128229; {e(emp['recruitment_channel'])}</div>
  <div class="stats">{stats_html}</div>
</div></body></html>"""


def gauge_html(probability: float, emp_name: str) -> str:
    value = int(round(probability * 100))
    if value > 80:
        gradient, shadow, label = "linear-gradient(135deg,#11998e,#38ef7d)", "rgba(17,153,142,.30)", "Highly Likely"
    elif value >= 55:
        gradient, shadow, label = "linear-gradient(135deg,#f7971e,#ffd200)", "rgba(247,151,30,.30)", "Moderate Chance"
    else:
        gradient, shadow, label = "linear-gradient(135deg,#c0392b,#e74c3c)", "rgba(192,57,43,.30)", "Low Likelihood"
    radius = 36
    circ = 2 * 3.14159265 * radius
    fill = circ * value / 100
    return f"""<!DOCTYPE html><html><head><style>
* {{ box-sizing:border-box; margin:0; padding:0;
     font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; }}
body {{ background:transparent; padding:2px 0; }}
.card {{ background:{gradient}; border-radius:16px; padding:22px 20px;
  box-shadow:0 4px 18px {shadow}; display:flex; flex-direction:column;
  align-items:center; color:white; text-align:center; gap:6px; }}
h3 {{ font-size:15px; font-weight:700; }} p {{ font-size:11px; opacity:.8; }}
.verdict {{ font-size:13px; font-weight:600; margin-top:8px;
  background:rgba(255,255,255,.2); border-radius:20px; padding:4px 14px; }}
</style></head><body><div class="card">
  <h3>Promotion Likelihood</h3><p>{html.escape(emp_name)}</p>
  <svg width="100" height="100" viewBox="0 0 100 100">
    <circle cx="50" cy="50" r="{radius}" fill="none" stroke="rgba(255,255,255,.25)" stroke-width="10"/>
    <circle cx="50" cy="50" r="{radius}" fill="none" stroke="white" stroke-width="10"
      stroke-dasharray="{fill:.1f} {circ - fill:.1f}" stroke-linecap="round" transform="rotate(-90 50 50)"/>
    <text x="50" y="46" text-anchor="middle" font-size="19" font-weight="bold" fill="white">{value}%</text>
    <text x="50" y="62" text-anchor="middle" font-size="9" fill="rgba(255,255,255,.75)">probability</text>
  </svg>
  <div class="verdict">{label}</div>
</div></body></html>"""


def score_bar(label: str, value: int, color: str) -> None:
    st.markdown(
        f"""<div style="margin-bottom:12px;">
          <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
            <span style="font-size:13px;opacity:.7;">{label}</span>
            <span style="font-size:13px;font-weight:600;color:{color};">{value}</span></div>
          <div style="background:rgba(128,128,128,.2);border-radius:6px;height:8px;">
            <div style="background:{color};width:{value}%;height:8px;border-radius:6px;"></div>
          </div></div>""",
        unsafe_allow_html=True,
    )


# ── Halaman ────────────────────────────────────────────────────────────────────
def run_profile_page(api_url: str) -> None:
    st.title("Employee Profiles")
    st.caption("Data dari `employees.csv`. Setiap prediksi = satu request `POST /predict` ke API.")
    st.divider()

    df = load_employees()

    with st.expander("All Employees", expanded=False):
        cols = ["name", "position", "department", "age", "gender",
                "previous_year_rating", "avg_training_score"]
        st.dataframe(df[cols], width="stretch", hide_index=True)

    # Prediksi massal (opsional): 1 request per karyawan
    with st.expander("Prediksi semua karyawan (40 request ke API)"):
        if st.button("Jalankan untuk semua"):
            rows, bar = [], st.progress(0.0)
            try:
                for i, (_, emp) in enumerate(df.iterrows(), start=1):
                    res = predict(api_url, to_payload(emp))
                    rows.append({"Name": emp["name"], "Department": emp["department"],
                                 "Probability": res["probability"], "Result": res["label"]})
                    bar.progress(i / len(df))
                st.session_state["bulk_result"] = pd.DataFrame(rows)
            except Exception as e:
                st.error(f"Gagal pada karyawan ke-{len(rows) + 1}: {e}")
        if "bulk_result" in st.session_state:
            st.dataframe(
                st.session_state["bulk_result"], width="stretch", hide_index=True,
                column_config={"Probability": st.column_config.ProgressColumn(
                    "Probability", min_value=0.0, max_value=1.0, format="percent")},
            )

    st.markdown("<br>", unsafe_allow_html=True)

    by_name = df.set_index("name")
    selected = st.selectbox(
        "Select Employee", df["name"].tolist(),
        format_func=lambda n: f"{n}  —  {by_name.loc[n, 'position']}, {by_name.loc[n, 'department']}",
    )
    emp = df[df["name"] == selected].iloc[0]

    render_html(profile_card_html(emp), height=290)

    st.subheader("Assessment Breakdown")
    c1, c2 = st.columns(2)
    with c1:
        score_bar("Performance Score", int(emp["performance_score"]), "#6c63ff")
        score_bar("Technical Score", int(emp["technical_score"]), "#48cae4")
    with c2:
        score_bar("Leadership Score", int(emp["leadership_score"]), "#f4a261")
        score_bar("Potential Score", int(emp["potential_score"]), "#2ec4b6")
    score_bar("Avg Training Score", int(emp["avg_training_score"]), "#e63946")

    st.divider()
    st.subheader("Promotion Prediction")
    st.caption("Output model dari API berdasarkan data profil karyawan ini.")

    if st.button("Run Prediction", type="primary", width="stretch"):
        payload = to_payload(emp)
        try:
            result = predict(api_url, payload)
        except Exception as e:
            st.error(f"Prediksi gagal: {e}")
            return

        if result["prediction"] == 1:
            st.success(f"**Promoted** — {emp['name']} kemungkinan besar dipromosikan.")
        else:
            st.warning(f"**Not Promoted** — {emp['name']} kecil kemungkinan dipromosikan.")

        col_a, col_b = st.columns([2, 3])
        with col_a:
            st.metric("Promotion Probability", f"{result['probability'] * 100:.1f}%")
            st.metric("Decision Threshold", f"{result['threshold'] * 100:.1f}%")
        with col_b:
            render_html(gauge_html(result["probability"], emp["name"]), height=260)
        with st.expander("Request / response (untuk debugging)"):
            st.json({"request": payload, "response": result})
