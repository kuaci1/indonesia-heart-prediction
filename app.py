import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="HeartGuard AI Indonesia",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_assets():
    try:
        model = joblib.load('heart_attack_rf_model.pkl') 
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        return None, None

model, scaler = load_assets()

EXPECTED_COLS = [
    'age', 'gender', 'region', 'income_level', 'hypertension', 'diabetes', 
    'cholesterol_level', 'waist_circumference', 'family_history', 'smoking_status', 
    'alcohol_consumption', 'physical_activity', 'dietary_habits', 'air_pollution_exposure', 
    'stress_level', 'sleep_hours', 'blood_pressure_systolic', 'blood_pressure_diastolic', 
    'fasting_blood_sugar', 'cholesterol_hdl', 'cholesterol_ldl', 'triglycerides', 
    'EKG_results', 'previous_heart_disease', 'medication_usage'
]
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966486.png", width=100)
    st.title("Profil Pasien")
    st.markdown("Isi data demografis dasar di sini.")
    
    age = st.slider("Usia (Tahun)", 20, 90, 45)
    gender = st.radio("Jenis Kelamin", ["Male", "Female"], horizontal=True)
    region = st.selectbox("Wilayah Tinggal", ["Urban (Kota)", "Rural (Desa)"])
    income = st.select_slider("Tingkat Ekonomi", ["Low", "Middle", "High"], value="Middle")
    
    st.markdown("---")
    st.subheader("Riwayat Medis")
    fam_hist = st.checkbox("Ada Riwayat Keluarga Sakit Jantung?")
    prev_heart = st.checkbox("Pernah Sakit Jantung Sebelumnya?")
    meds = st.checkbox("Sedang Mengonsumsi Obat Jantung?")

# --- 3. JUDUL & DESKRIPSI APLIKASI ---
st.title("❤️ HeartGuard AI: Deteksi Dini Risiko Jantung")
st.markdown("""
Aplikasi ini menggunakan **Artificial Intelligence (Random Forest)** yang dilatih dengan data kesehatan Indonesia 
untuk memprediksi risiko serangan jantung berdasarkan profil klinis & gaya hidup.
""")

# Membagi layar menjadi 3 Tab agar rapi
tab1, tab2, tab3 = st.tabs(["📊 Data Klinis (Lab)", "🏃 Gaya Hidup & Lingkungan", "🩺 Penyakit Penyerta"])

with tab1:
    st.info("Masukkan hasil pemeriksaan laboratorium terakhir.")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Tanda Vital")
        sys_bp = st.number_input("Tensi Sistolik (Atas)", 90, 220, 120)
        dia_bp = st.number_input("Tensi Diastolik (Bawah)", 60, 140, 80)
        ekg = st.selectbox("Hasil EKG", ["Normal", "Abnormal"])
        
    with col2:
        st.markdown("#### Profil Lemak")
        chol = st.number_input("Kolesterol Total", 100, 400, 200)
        ldl = st.number_input("LDL (Jahat)", 50, 250, 100)
        hdl = st.number_input("HDL (Baik)", 20, 100, 50)
        
    with col3:
        st.markdown("#### Metabolisme")
        sugar = st.number_input("Gula Darah Puasa", 70, 300, 100)
        trig = st.number_input("Trigliserida", 50, 400, 150)
        waist = st.number_input("Lingkar Pinggang (cm)", 50, 150, 80)

with tab2:
    col_life1, col_life2 = st.columns(2)
    with col_life1:
        smoking = st.selectbox("Status Merokok", ["Never", "Past", "Current"])
        alcohol = st.selectbox("Konsumsi Alkohol", ["None", "Moderate", "High"])
        diet = st.radio("Pola Makan (Diet)", ["Healthy", "Unhealthy"], horizontal=True)
    
    with col_life2:
        activity = st.select_slider("Aktivitas Fisik", ["Low", "Moderate", "High"])
        stress = st.select_slider("Tingkat Stres", ["Low", "Moderate", "High"])
        pollution = st.select_slider("Paparan Polusi Udara", ["Low", "Moderate", "High"])
        sleep = st.slider("Jam Tidur per Hari", 3, 12, 7)

with tab3:
    col_comor1, col_comor2 = st.columns(2)
    with col_comor1:
        hyp = st.toggle("Menderita Hipertensi?", value=True if sys_bp > 140 else False)
    with col_comor2:
        dia = st.toggle("Menderita Diabetes?", value=True if sugar > 126 else False)


# --- 5. LOGIC PEMROSESAN DATA (Backend di Frontend) ---
if st.button("🔍 ANALISIS RISIKO SEKARANG", use_container_width=True, type="primary"):
    
    if model is None or scaler is None:
        st.error("Model tidak ditemukan! Pastikan file .pkl ada di folder yang sama.")
    else:
        # 1. Kumpulkan Inputan ke dalam Dictionary
        input_data = {
            'age': age,
            'gender': 1 if gender == 'Male' else 0,
            'region': 1 if "Urban" in region else 0,
            'income_level': {'Low': 0, 'Middle': 1, 'High': 2}[income],
            'hypertension': int(hyp),
            'diabetes': int(dia),
            'family_history': int(fam_hist),
            'previous_heart_disease': int(prev_heart),
            'smoking_status': {'Never': 0, 'Past': 1, 'Current': 2}[smoking],
            'alcohol_consumption': {'None': 0, 'Moderate': 1, 'High': 2}[alcohol],
            'physical_activity': {'Low': 0, 'Moderate': 1, 'High': 2}[activity],
            'dietary_habits': 1 if diet == 'Unhealthy' else 0,
            'air_pollution_exposure': {'Low': 0, 'Moderate': 1, 'High': 2}[pollution],
            'stress_level': {'Low': 0, 'Moderate': 1, 'High': 2}[stress],
            'sleep_hours': sleep,
            'blood_pressure_systolic': sys_bp,
            'blood_pressure_diastolic': dia_bp,
            'cholesterol_level': chol,
            'cholesterol_ldl': ldl,
            'cholesterol_hdl': hdl,
            'triglycerides': trig,
            'fasting_blood_sugar': sugar,
            'waist_circumference': waist,
            'EKG_results': 1 if ekg == 'Abnormal' else 0,
            'medication_usage': int(meds)
        }
        # 2. Buat DataFrame dari inputan
        df_input = pd.DataFrame([input_data])
        
        # Pastikan urutan kolom sesuai ekspektasi model
        df_input = df_input[EXPECTED_COLS] 
        
        # 3. Scaling
        X_scaled = scaler.transform(df_input)
        
        # 4. Prediksi
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0][1]
        prob_percent = round(probability * 100, 1)
        
        # --- 6. TAMPILAN HASIL (RESULT UI) ---
        st.markdown("---")
        
        # Layout Hasil
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            if prediction == 1:
                st.markdown("### Status Risiko:")
                st.error("⚠️ TINGGI (HIGH RISK)")
                st.metric(label="Probabilitas Serangan Jantung", value=f"{prob_percent}%", delta="Bahaya")
            else:
                st.markdown("### Status Risiko:")
                st.success("✅ RENDAH (LOW RISK)")
                st.metric(label="Probabilitas Serangan Jantung", value=f"{prob_percent}%", delta="-Aman")
        
        with res_col2:
            st.markdown("### Analisis Detail")
            # Progress bar dengan warna custom
            st.progress(probability, text=f"Skor Risiko: {prob_percent}/100")
            
            # Saran Kesehatan Dinamis (Smart Advice)
            st.markdown("#### 💡 Saran Kesehatan Personal:")
            advice_list = []
            
            if smoking == "Current":
                advice_list.append("- **Berhenti Merokok:** Ini adalah faktor risiko terbesar yang bisa Anda ubah saat ini.")
            if chol > 200 or ldl > 130:
                advice_list.append("- **Perbaiki Diet:** Kurangi gorengan dan santan. Kolesterol Anda di atas batas normal.")
            if sleep < 6:
                advice_list.append("- **Tidur Cukup:** Kurang tidur memicu stres jantung. Usahakan 7-8 jam.")
            if "Urban" in region and pollution == "High":
                advice_list.append("- **Polusi Udara:** Gunakan masker saat di luar ruangan, polusi memicu peradangan pembuluh darah.")
            
            if not advice_list:
                st.write("Gaya hidup Anda sudah cukup baik! Pertahankan pola makan dan olahraga rutin.")
            else:
                for advice in advice_list:
                    st.write(advice)

        # Disclaimer
        st.caption("Disclaimer: Hasil ini adalah prediksi AI dan bukan pengganti diagnosis medis dokter profesional.")