"""
app_streamlit.py - Monolithic Streamlit Deployment
DTSC6012001 Model Deployment - Mid Exam 2026

Cara menjalankan:
    streamlit run app_streamlit.py

Deploy ke Streamlit Cloud:
    1. Push ke GitHub repo
    2. Buka share.streamlit.io, New app
    3. Pilih repo & file ini
    4. Tambahkan requirements.txt
"""

import pickle
import numpy as np
import pandas as pd
import streamlit as st

# CONFIG & STYLE

st.set_page_config(
    page_title="Student Placement Predictor",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1f4e79; }
    .subtitle   { color: #666; margin-bottom: 1rem; }
    .result-box { padding: 1rem; border-radius: 8px; font-size: 1.1rem; }
    .placed     { background-color: #d4edda; color: #155724; }
    .not-placed { background-color: #f8d7da; color: #721c24; }
    .salary     { background-color: #d1ecf1; color: #0c5460; }
</style>
""", unsafe_allow_html=True)

# LOAD MODEL

@st.cache_resource
def load_model(path='best_models.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)


try:
    artifacts = load_model()
    clf_pipeline = artifacts['clf_pipeline']
    reg_pipeline = artifacts['reg_pipeline']
    clf_name     = artifacts['clf_name']
    reg_name     = artifacts['reg_name']
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Gagal memuat model: {e}")
    st.info("Pastikan file best_models.pkl ada di directory yang sama.")


# INFERENCE FUNCTION

def predict(input_dict: dict):
    """
    Jalankan prediksi klasifikasi & regresi dari input user.
    """
    df_input = pd.DataFrame([input_dict])
    placement_pred = clf_pipeline.predict(df_input)[0]
    placement_prob = clf_pipeline.predict_proba(df_input)[0]
    salary_pred    = reg_pipeline.predict(df_input)[0]
    return placement_pred, placement_prob, salary_pred


# UI LAYOUT

st.markdown('<p class="main-title">🎓 Student Placement Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Prediksi status penempatan kerja dan estimasi gaji mahasiswa</p>',
            unsafe_allow_html=True)

if model_loaded:
    st.sidebar.header(" Informasi Model")
    st.sidebar.success(f" Model dimuat")
    st.sidebar.info(f"**Klasifikasi:** {clf_name}")
    st.sidebar.info(f"**Regresi:** {reg_name}")

    st.sidebar.markdown("---")
    st.sidebar.header("ℹ Tentang Aplikasi")
    st.sidebar.write("""
    Aplikasi ini memprediksi:
    - **Placement Status**: Placed / Not Placed
    - **Salary Estimate**: dalam LPA (Lakhs Per Annum)
    """)

    # Form Input 
    st.subheader(" Masukkan Data Mahasiswa")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Profil Akademik**")
            cgpa = st.slider("CGPA", 5.0, 10.0, 7.5, 0.1)
            tenth_pct   = st.number_input("Tenth Percentage (%)", 40.0, 100.0, 75.0)
            twelfth_pct = st.number_input("Twelfth Percentage (%)", 40.0, 100.0, 72.0)
            backlogs    = st.number_input("Jumlah Backlog", 0, 10, 0)
            attendance  = st.slider("Attendance (%)", 50.0, 100.0, 85.0, 1.0)

        with col2:
            st.markdown("**Skills & Kegiatan**")
            coding_skill    = st.slider("Coding Skill Rating", 1, 10, 6)
            comm_skill      = st.slider("Communication Skill Rating", 1, 10, 6)
            apt_skill       = st.slider("Aptitude Skill Rating", 1, 10, 6)
            projects        = st.number_input("Projects Completed", 0, 10, 2)
            internships     = st.number_input("Internships Completed", 0, 5, 1)
            hackathons      = st.number_input("Hackathons Participated", 0, 10, 1)
            certifications  = st.number_input("Certifications Count", 0, 20, 3)

        with col3:
            st.markdown("**Profil Personal**")
            gender       = st.selectbox("Gender", ["Male", "Female"])
            branch       = st.selectbox("Branch", ["CSE", "ECE", "IT", "Mechanical", "Civil", "EEE"])
            study_hours  = st.slider("Study Hours per Day", 1.0, 12.0, 5.0, 0.5)
            sleep_hours  = st.slider("Sleep Hours", 4.0, 10.0, 7.0, 0.5)
            stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
            part_time    = st.selectbox("Part Time Job", ["Yes", "No"])
            fam_income   = st.selectbox("Family Income Level", ["Low", "Medium", "High"])
            city_tier    = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
            internet     = st.selectbox("Internet Access", ["Yes", "No"])
            extracurr    = st.selectbox("Extracurricular Involvement", ["Low", "Medium", "High"])

        submitted = st.form_submit_button(" Prediksi Sekarang", use_container_width=True)

    # ---- Prediction Output ----
    if submitted:
        input_dict = {
            'gender': gender,
            'branch': branch,
            'cgpa': cgpa,
            'tenth_percentage': tenth_pct,
            'twelfth_percentage': twelfth_pct,
            'backlogs': backlogs,
            'study_hours_per_day': study_hours,
            'attendance_percentage': attendance,
            'projects_completed': projects,
            'internships_completed': internships,
            'coding_skill_rating': coding_skill,
            'communication_skill_rating': comm_skill,
            'aptitude_skill_rating': apt_skill,
            'hackathons_participated': hackathons,
            'certifications_count': certifications,
            'sleep_hours': sleep_hours,
            'stress_level': stress_level,
            'part_time_job': part_time,
            'family_income_level': fam_income,
            'city_tier': city_tier,
            'internet_access': internet,
            'extracurricular_involvement': extracurr
        }

        with st.spinner("Memproses prediksi..."):
            placement_pred, placement_prob, salary_pred = predict(input_dict)

        st.markdown("---")
        st.subheader(" Hasil Prediksi")

        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            label = " PLACED" if placement_pred == 1 else " NOT PLACED"
            css   = "placed" if placement_pred == 1 else "not-placed"
            st.markdown(f'<div class="result-box {css}"><b>Status:</b><br>{label}</div>',
                        unsafe_allow_html=True)

        with res_col2:
            prob_placed = placement_prob[1] * 100
            st.metric("Probabilitas Placed", f"{prob_placed:.1f}%")
            st.progress(int(prob_placed))

        with res_col3:
            st.markdown(f'<div class="result-box salary"><b>Estimasi Gaji:</b><br> {salary_pred:.2f} LPA</div>',
                        unsafe_allow_html=True)

        # Show input summary
        with st.expander("Lihat Data Input"):
            st.json(input_dict)

else:
    st.warning(" Model belum dimuat. Jalankan `pipeline.py` terlebih dahulu untuk menghasilkan `best_models.pkl`.")
