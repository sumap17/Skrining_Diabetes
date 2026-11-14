import streamlit as st
import numpy as np
import pandas as pd
import json
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import tensorflow as tf
from datetime import datetime

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="GlucoGuard AI | Clinical Diabetes Screening",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS UNTUK TAMPILAN MODERN ---
st.markdown("""
<style>
    /* Hero Banner */
    .hero-box {
        background: linear-gradient(135deg, #1C2541 0%, #0B132B 60%, #114B5F 100%);
        border: 1px solid rgba(0, 212, 182, 0.3);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 20px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #FFFFFF;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        font-size: 1.02rem;
        color: #B8C7D9;
        margin-bottom: 12px;
    }
    .badge {
        display: inline-block;
        background-color: rgba(0, 212, 182, 0.15);
        color: #00D4B6;
        border: 1px solid rgba(0, 212, 182, 0.4);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 8px;
    }
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #1C2541;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 16px 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    /* Status Banners */
    .status-high {
        background: linear-gradient(90deg, rgba(239, 68, 68, 0.2), rgba(153, 27, 27, 0.15));
        border-left: 6px solid #EF4444;
        padding: 18px;
        border-radius: 10px;
        margin-bottom: 16px;
    }
    .status-low {
        background: linear-gradient(90deg, rgba(16, 185, 129, 0.2), rgba(6, 95, 70, 0.15));
        border-left: 6px solid #10B981;
        padding: 18px;
        border-radius: 10px;
        margin-bottom: 16px;
    }
    /* Placeholder Card sebelum tombol diklik */
    .empty-state-box {
        background-color: rgba(28, 37, 65, 0.65);
        border: 1px dashed rgba(0, 212, 182, 0.45);
        border-radius: 14px;
        padding: 40px 24px;
        text-align: center;
        margin-top: 10px;
    }
    /* Info Cards untuk Tab Tentang */
    .about-card {
        background-color: #1C2541;
        border: 1px solid rgba(0, 212, 182, 0.22);
        border-radius: 14px;
        padding: 20px 22px;
        height: 100%;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.2);
    }
    .about-card h4 {
        color: #00D4B6;
        margin-top: 0;
        margin-bottom: 10px;
    }
    .team-card {
        background: linear-gradient(145deg, #1C2541, #141E36);
        border: 1px solid rgba(0, 212, 182, 0.35);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.25);
    }
    .team-name {
        font-size: 1.12rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-top: 8px;
        margin-bottom: 4px;
    }
    .team-nim {
        font-size: 0.9rem;
        color: #00D4B6;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .team-role {
        font-size: 0.85rem;
        color: #B8C7D9;
    }
</style>
""", unsafe_allow_html=True)

# --- INISIALISASI SESSION STATE ---
if 'has_predicted' not in st.session_state:
    st.session_state.has_predicted = False
if 'patient_data' not in st.session_state:
    st.session_state.patient_data = {}
if 'prob_diabetes' not in st.session_state:
    st.session_state.prob_diabetes = 0.0
if 'shap_vals' not in st.session_state:
    st.session_state.shap_vals = None
if 'X_transformed' not in st.session_state:
    st.session_state.X_transformed = None
if 'form_reset_counter' not in st.session_state:
    st.session_state.form_reset_counter = 0

def reset_all_data():
    st.session_state.has_predicted = False
    st.session_state.patient_data = {}
    st.session_state.prob_diabetes = 0.0
    st.session_state.shap_vals = None
    st.session_state.X_transformed = None
    st.session_state.form_reset_counter += 1

# --- LOAD ARTEFAK MODEL ---
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model('diabetes_mlp_model.keras')
    background_data = np.load('shap_background.npy')
    with open('model_config.json', 'r') as f:
        config = json.load(f)
    return model, background_data, config

model, background_data, config = load_artifacts()
optimal_threshold = config['optimal_threshold']
feature_names = config['feature_names']
scaler_mean = np.array(config['scaler_mean'])
scaler_scale = np.array(config['scaler_scale'])
thresh_percent = optimal_threshold * 100

friendly_names = {
    'age': 'Usia',
    'hypertension': 'Riwayat Hipertensi',
    'heart_disease': 'Riwayat Penyakit Jantung',
    'bmi': 'Indeks Massa Tubuh (BMI)',
    'HbA1c_level': 'Kadar HbA1c (%)',
    'blood_glucose_level': 'Glukosa Darah (mg/dL)',
    'gender_Male': 'Gender: Laki-laki',
    'gender_Other': 'Gender: Lainnya',
    'smoking_history_current': 'Perokok Aktif',
    'smoking_history_ever': 'Pernah Merokok',
    'smoking_history_former': 'Mantan Perokok',
    'smoking_history_never': 'Tidak Pernah Merokok',
    'smoking_history_not current': 'Bukan Perokok Saat Ini'
}

def preprocess_input(gender, age, hypertension, heart_disease, smoking_history, bmi, hba1c, glucose):
    num_raw = np.array([age, hypertension, heart_disease, bmi, hba1c, glucose], dtype=float)
    num_scaled = (num_raw - scaler_mean) / scaler_scale

    gender_male = 1.0 if gender == "Male" else 0.0
    gender_other = 1.0 if gender == "Other" else 0.0

    smoking_current = 1.0 if smoking_history == "current" else 0.0
    smoking_ever = 1.0 if smoking_history == "ever" else 0.0
    smoking_former = 1.0 if smoking_history == "former" else 0.0
    smoking_never = 1.0 if smoking_history == "never" else 0.0
    smoking_not_current = 1.0 if smoking_history == "not current" else 0.0

    cat_encoded = np.array([
        gender_male, gender_other,
        smoking_current, smoking_ever, smoking_former, smoking_never, smoking_not_current
    ], dtype=float)

    return np.concatenate([num_scaled, cat_encoded]).reshape(1, -1)

def model_predict(data):
    return model.predict(data, verbose=0).ravel()

@st.cache_resource
def get_shap_explainer(_bg_data):
    return shap.KernelExplainer(model_predict, _bg_data)

explainer = get_shap_explainer(background_data)

def create_gauge_chart(prob_percent, thresh_percent):
    bar_color = "#EF4444" if prob_percent >= thresh_percent else "#10B981"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob_percent,
        number={'suffix': "%", 'font': {'size': 36, 'color': '#FFFFFF'}},
        delta={'reference': thresh_percent, 'increasing': {'color': "#EF4444"}, 'decreasing': {'color': "#10B981"}},
        title={'text': "Skor Probabilitas Risiko Diabetes", 'font': {'size': 16, 'color': '#B8C7D9'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#B8C7D9"},
            'bar': {'color': bar_color, 'thickness': 0.28},
            'bgcolor': "rgba(255,255,255,0.05)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, thresh_percent], 'color': 'rgba(16, 185, 129, 0.22)'},
                {'range': [thresh_percent, 100], 'color': 'rgba(239, 68, 68, 0.22)'}
            ],
            'threshold': {
                'line': {'color': "#F59E0B", 'width': 4},
                'thickness': 0.8,
                'value': thresh_percent
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=270,
        margin=dict(l=20, r=20, t=45, b=10)
    )
    return fig

# --- HERO HEADER ---
st.markdown("""
<div class="hero-box">
    <div class="hero-title">🩺 GlucoGuard AI — Sistem Skrining Dini Risiko Diabetes</div>
    <div class="hero-subtitle">
        Clinical Decision Support System berbasis <b>Multi-Layer Perceptron (MLP)</b> dengan Kalibrasi 
        <b>Decision Threshold</b> dan Transparansi <b>Explainable AI (SHAP)</b>.
    </div>
    <span class="badge">🧠 Deep Learning MLP</span>
    <span class="badge">🎯 Optimal Threshold: 35.62%</span>
    <span class="badge">🔍 SHAP Explainability</span>
    <span class="badge">📊 ROC-AUC: 0.9740</span>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR: FORM INPUT TERSTRUKTUR & DINAMIS ---
st.sidebar.markdown("## 📋 Profil Klinis Pasien")

if st.sidebar.button("🗑️ Kosongkan / Reset Data", use_container_width=True, on_click=reset_all_data):
    st.toast("Formulir berhasil dikosongkan!", icon="🧹")

rc = st.session_state.form_reset_counter

smoking_options = {
    "Tidak Pernah (never)": "never",
    "Tidak Ada Informasi (No Info)": "No Info",
    "Mantan Perokok (former)": "former",
    "Perokok Aktif (current)": "current",
    "Bukan Perokok Saat Ini (not current)": "not current",
    "Pernah Merokok (ever)": "ever"
}

# Bagian 1: Demografis & Gaya Hidup
with st.sidebar.container(border=True):
    st.markdown("### 👤 1. Demografis & Gaya Hidup")
    patient_name = st.text_input("Nama / ID Pasien", value="", placeholder="Masukkan nama atau ID pasien...", key=f"name_{rc}")
    gender = st.selectbox("Jenis Kelamin", ["Female", "Male", "Other"], key=f"gender_{rc}")
    age = st.slider("Usia (Tahun)", min_value=1, max_value=100, value=30, key=f"age_{rc}")
    smoking_choice = st.selectbox("Riwayat Merokok", list(smoking_options.keys()), key=f"smoke_{rc}")

# Bagian 2: Antropometri (BMI)
with st.sidebar.container(border=True):
    st.markdown("### ⚖️ 2. Antropometri (BMI)")
    bmi_mode = st.radio(
        "Metode Input BMI",
        ["Hitung dari Tinggi & Berat", "Input BMI Langsung"],
        key=f"bmi_mode_{rc}"
    )
    if bmi_mode == "Hitung dari Tinggi & Berat":
        tb = st.number_input("Tinggi Badan (cm)", min_value=100, max_value=230, value=165, key=f"tb_{rc}")
        bb = st.number_input("Berat Badan (kg)", min_value=25.0, max_value=200.0, value=60.0, step=0.5, key=f"bb_{rc}")
        bmi_final = round(bb / ((tb / 100) ** 2), 2)
        st.caption(f"💡 Estimasi BMI: **{bmi_final} kg/m²**")
    else:
        bmi_final = st.number_input("Indeks Massa Tubuh / BMI (kg/m²)", min_value=10.0, max_value=70.0, value=22.0, step=0.1, key=f"bmi_man_{rc}")

# Bagian 3: Parameter Laboratorium & Medis
with st.sidebar.container(border=True):
    st.markdown("### 🩸 3. Parameter Laboratorium & Medis")
    hypertension_label = st.selectbox("Riwayat Hipertensi", ["Tidak", "Ya"], key=f"ht_{rc}")
    heart_label = st.selectbox("Riwayat Penyakit Jantung", ["Tidak", "Ya"], key=f"hd_{rc}")
    hba1c = st.slider("Kadar HbA1c (%)", min_value=3.5, max_value=12.0, value=5.2, step=0.1, key=f"hba1c_{rc}")
    glucose = st.slider("Kadar Glukosa Darah (mg/dL)", min_value=70, max_value=300, value=100, step=1, key=f"gluc_{rc}")

# Tombol Eksekusi Prediksi
submit_btn = st.sidebar.button(
    label="🔍 Mulai Analisis & Prediksi",
    type="primary",
    use_container_width=True
)

# --- EKSEKUSI KOMPUTASI HANYA SAAT TOMBOL PREDIKSI DIKLIK ---
if submit_btn:
    display_name = patient_name.strip() if patient_name.strip() != "" else "Pasien-Anonim"
    hypertension_val = 1 if hypertension_label == "Ya" else 0
    heart_val = 1 if heart_label == "Ya" else 0
    smoking_val = smoking_options[smoking_choice]

    X_input = preprocess_input(
        gender, age, hypertension_val, heart_val, smoking_val, bmi_final, hba1c, glucose
    )

    with st.spinner("Sedang menjalankan model MLP & menghitung kontribusi variabel SHAP..."):
        prob_val = float(model_predict(X_input)[0])
        shap_val = explainer.shap_values(X_input)[0]

    st.session_state.has_predicted = True
    st.session_state.prob_diabetes = prob_val
    st.session_state.shap_vals = shap_val
    st.session_state.X_transformed = X_input
    st.session_state.patient_data = {
        'patient_name': display_name,
        'gender': gender,
        'age': age,
        'smoking_choice': smoking_choice,
        'smoking_val': smoking_val,
        'bmi': bmi_final,
        'hypertension_label': hypertension_label,
        'hypertension_val': hypertension_val,
        'heart_label': heart_label,
        'heart_val': heart_val,
        'hba1c': hba1c,
        'glucose': glucose
    }

# --- NAVIGASI 5 TAB UTAMA ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🩺 Skrining & Analisis SHAP",
    "🔄 Simulasi Intervensi (What-If)",
    "📂 Skrining Massal (Batch CSV)",
    "📈 Performa & Arsitektur Model",
    "ℹ️ Tentang Proyek"
])

# =====================================================================
# TAB 1: SKRINING INDIVIDU & EXPLAINABLE AI (SHAP)
# =====================================================================
with tab1:
    if not st.session_state.has_predicted:
        st.markdown("""
        <div class="empty-state-box">
            <h3 style="color:#00D4B6; margin-bottom:8px;">Menunggu Input Data Pasien</h3>
            <p style="color:#B8C7D9; font-size:1.05rem; max-width:620px; margin:0 auto;">
                Silakan lengkapi seluruh parameter klinis dan gaya hidup pasien pada panel <b>Profil Klinis Pasien</b> 
                di sebelah kiri, kemudian tekan tombol <b>"🔍 Mulai Analisis & Prediksi"</b> untuk menampilkan hasil skrining dan penjelasan SHAP.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        pdata = st.session_state.patient_data
        prob_diabetes = st.session_state.prob_diabetes
        prob_percent = prob_diabetes * 100
        is_high_risk = prob_diabetes >= optimal_threshold

        col_left, col_right = st.columns([1.05, 1.25], gap="large")

        with col_left:
            st.subheader(f"📊 Hasil Skrining: {pdata['patient_name']}")
            if is_high_risk:
                st.markdown(f"""
                <div class="status-high">
                    <h3 style="color:#FCA5A5; margin:0;">⚠️ TERINDIKASI RISIKO TINGGI DIABETES</h3>
                    <p style="color:#E2E8F0; margin-top:6px; margin-bottom:0;">
                        Probabilitas pasien (<b>{prob_percent:.2f}%</b>) melampaui ambang batas klinis terkalibrasi (<b>{thresh_percent:.2f}%</b>). 
                        Disarankan segera melakukan pemeriksaan laboratorium lanjutan (GDP/TTGO) dan konsultasi dokter.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="status-low">
                    <h3 style="color:#6EE7B7; margin:0;">✅ TERINDIKASI RISIKO RENDAH (NORMAL)</h3>
                    <p style="color:#E2E8F0; margin-top:6px; margin-bottom:0;">
                        Probabilitas pasien (<b>{prob_percent:.2f}%</b>) berada di bawah ambang batas risiko (<b>{thresh_percent:.2f}%</b>). 
                        Tetap pertahankan pola makan seimbang dan aktivitas fisik rutin.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            st.plotly_chart(create_gauge_chart(prob_percent, thresh_percent), use_container_width=True)

            st.markdown("#### 🧾 Ringkasan Indikator Klinis Pasien")
            k1, k2, k3 = st.columns(3)
            hba1c_status = "Normal" if pdata['hba1c'] < 5.7 else ("Prediabetes" if pdata['hba1c'] < 6.5 else "Tinggi")
            gluc_status = "Normal" if pdata['glucose'] < 140 else ("Waspada" if pdata['glucose'] < 200 else "Tinggi")
            bmi_status = "Normal" if pdata['bmi'] < 25 else ("Overweight" if pdata['bmi'] < 30 else "Obesitas")

            k1.metric("HbA1c", f"{pdata['hba1c']}%", hba1c_status, delta_color="inverse" if pdata['hba1c'] >= 5.7 else "normal")
            k2.metric("Glukosa", f"{pdata['glucose']} mg/dL", gluc_status, delta_color="inverse" if pdata['glucose'] >= 140 else "normal")
            k3.metric("BMI", f"{pdata['bmi']}", bmi_status, delta_color="inverse" if pdata['bmi'] >= 25 else "normal")

        with col_right:
            st.subheader("🧠 Penjelasan Keputusan AI (SHAP Waterfall)")
            readable_features = [friendly_names.get(f, f) for f in feature_names]

            explanation = shap.Explanation(
                values=st.session_state.shap_vals,
                base_values=explainer.expected_value,
                data=np.round(st.session_state.X_transformed[0], 2),
                feature_names=readable_features
            )

            fig, ax = plt.subplots(figsize=(8.5, 5.2))
            shap.plots.waterfall(explanation, show=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            shap_df = pd.DataFrame({
                'Fitur': readable_features,
                'Kontribusi_SHAP': st.session_state.shap_vals
            }).sort_values(by='Kontribusi_SHAP', ascending=False)

            top_risk_factors = shap_df[shap_df['Kontribusi_SHAP'] > 0.005].head(3)

            st.markdown("#### 💡 Interpretasi & Rekomendasi Personal")
            if not top_risk_factors.empty:
                factors_str = ", ".join([f"**{row['Fitur']}** (+{row['Kontribusi_SHAP']:.3f})" for _, row in top_risk_factors.iterrows()])
                st.warning(f"**Faktor Dominan Peningkat Risiko:** {factors_str}.")
            else:
                st.success("Tidak ditemukan faktor klinis yang mendorong lonjakan skor risiko secara signifikan.")

            report_text = f"""LAPORAN SKRINING RISIKO DIABETES (GLUCOGUARD AI)
Tanggal Pemeriksaan : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
ID / Nama Pasien    : {pdata['patient_name']}
---------------------------------------------------------
PARAMETER PASIEN:
- Jenis Kelamin     : {pdata['gender']}
- Usia              : {pdata['age']} tahun
- BMI               : {pdata['bmi']} kg/m2 ({bmi_status})
- Hipertensi        : {pdata['hypertension_label']}
- Penyakit Jantung  : {pdata['heart_label']}
- Riwayat Merokok   : {pdata['smoking_choice']}
- Kadar HbA1c       : {pdata['hba1c']}% ({hba1c_status})
- Glukosa Darah     : {pdata['glucose']} mg/dL ({gluc_status})
---------------------------------------------------------
HASIL ANALISIS MODEL DEEP LEARNING (MLP):
- Probabilitas Risiko : {prob_percent:.2f}%
- Ambang Batas Klinis : {thresh_percent:.2f}%
- Status Skrining     : {'RISIKO TINGGI DIABETES' if is_high_risk else 'RISIKO RENDAH (NORMAL)'}
---------------------------------------------------------
Catatan: Hasil ini merupakan skrining awal berbasis kecerdasan buatan dan bukan pengganti diagnosis medis dokter.
"""
            st.download_button(
                label="📥 Unduh Laporan Hasil Skrining (.TXT)",
                data=report_text,
                file_name=f"Laporan_Skrining_{pdata['patient_name']}.txt",
                mime="text/plain",
                use_container_width=True
            )

# =====================================================================
# TAB 2: SIMULASI INTERVENSI GAYA HIDUP (WHAT-IF ANALYSIS)
# =====================================================================
with tab2:
    if not st.session_state.has_predicted:
        st.info("Jalankan prediksi pasien terlebih dahulu pada panel di sebelah kiri sebelum menggunakan simulasi intervensi.")
    else:
        pdata = st.session_state.patient_data
        prob_percent = st.session_state.prob_diabetes * 100

        st.subheader(f"🔄 Simulasi Intervensi Klinis: {pdata['patient_name']}")
        st.markdown(
            "Atur target perubahan parameter di bawah ini, lalu klik **Hitung Simulasi** untuk melihat "
            "penurunan risiko diabetes apabila pasien memperbaiki gaya hidup."
        )

        sim_col1, sim_col2 = st.columns([1, 1.4], gap="large")

        with sim_col1:
            with st.form(key="what_if_form"):
                st.markdown("#### 🎛️ Target Perubahan Parameter")
                sim_hba1c = st.slider("Target Kadar HbA1c Baru (%)", 3.5, 12.0, float(pdata['hba1c']), 0.1)
                sim_glucose = st.slider("Target Glukosa Darah Baru (mg/dL)", 70, 300, int(pdata['glucose']), 1)
                sim_bmi = st.slider("Target BMI Baru (kg/m²)", 15.0, 60.0, float(pdata['bmi']), 0.5)
                sim_smoking = st.selectbox(
                    "Target Status Merokok",
                    list(smoking_options.keys()),
                    index=list(smoking_options.keys()).index(pdata['smoking_choice'])
                )
                run_sim_btn = st.form_submit_button("⚡ Hitung Simulasi Perubahan", type="primary", use_container_width=True)

        with sim_col2:
            if run_sim_btn:
                X_sim = preprocess_input(
                    pdata['gender'], pdata['age'], pdata['hypertension_val'], pdata['heart_val'],
                    smoking_options[sim_smoking], sim_bmi, sim_hba1c, sim_glucose
                )
                sim_prob_percent = float(model_predict(X_sim)[0]) * 100
                diff_prob = sim_prob_percent - prob_percent

                m1, m2, m3 = st.columns(3)
                m1.metric("Risiko Awal", f"{prob_percent:.2f}%")
                m2.metric("Risiko Setelah Simulasi", f"{sim_prob_percent:.2f}%", f"{diff_prob:+.2f}%", delta_color="inverse")
                m3.metric("Status Baru", "RISIKO TINGGI" if sim_prob_percent >= thresh_percent else "NORMAL")

                comp_fig = go.Figure(data=[
                    go.Bar(
                        name='Kondisi Awal',
                        x=['Probabilitas Risiko (%)'],
                        y=[prob_percent],
                        marker_color='#EF4444' if prob_percent >= thresh_percent else '#3B82F6',
                        text=[f"{prob_percent:.2f}%"],
                        textposition='auto'
                    ),
                    go.Bar(
                        name='Setelah Simulasi',
                        x=['Probabilitas Risiko (%)'],
                        y=[sim_prob_percent],
                        marker_color='#10B981' if sim_prob_percent < thresh_percent else '#F59E0B',
                        text=[f"{sim_prob_percent:.2f}%"],
                        textposition='auto'
                    )
                ])
                comp_fig.add_hline(
                    y=thresh_percent, line_dash="dash", line_color="#F59E0B",
                    annotation_text=f"Batas Ambang ({thresh_percent:.2f}%)"
                )
                comp_fig.update_layout(
                    title="Perbandingan Skor Risiko Sebelum & Sesudah Intervensi",
                    yaxis_range=[0, 105],
                    barmode='group',
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=330
                )
                st.plotly_chart(comp_fig, use_container_width=True)
            else:
                st.info("Atur target HbA1c, Glukosa, atau BMI pada panel kiri, lalu tekan **⚡ Hitung Simulasi Perubahan**.")

# =====================================================================
# TAB 3: SKRINING MASSAL (BATCH PREDICTION VIA CSV)
# =====================================================================
with tab3:
    st.subheader("📂 Skrining Massal Pasien (Batch Prediction via Upload CSV)")
    st.markdown(
        "Unggah file CSV berisi banyak data pasien sekaligus untuk melakukan prediksi massal. "
        "Pastikan file memiliki kolom: `gender`, `age`, `hypertension`, `heart_disease`, `smoking_history`, `bmi`, `HbA1c_level`, `blood_glucose_level`."
    )

    sample_batch = pd.DataFrame([
        {"gender": "Female", "age": 54, "hypertension": 1, "heart_disease": 0, "smoking_history": "never", "bmi": 31.2, "HbA1c_level": 7.4, "blood_glucose_level": 185},
        {"gender": "Male", "age": 29, "hypertension": 0, "heart_disease": 0, "smoking_history": "never", "bmi": 22.4, "HbA1c_level": 5.1, "blood_glucose_level": 95},
        {"gender": "Male", "age": 62, "hypertension": 1, "heart_disease": 1, "smoking_history": "former", "bmi": 28.9, "HbA1c_level": 6.6, "blood_glucose_level": 160}
    ])
    st.download_button(
        "📄 Unduh Template CSV Contoh (3 Pasien)",
        data=sample_batch.to_csv(index=False),
        file_name="template_pasien_diabetes.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Pilih file CSV data pasien", type=["csv"])
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Pratinjau Data Terunggah:", batch_df.head())

        if st.button("Proses Prediksi Massal", type="primary"):
            required_cols = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']
            if all(col in batch_df.columns for col in required_cols):
                transformed_rows = np.vstack([
                    preprocess_input(
                        row['gender'], row['age'], row['hypertension'], row['heart_disease'],
                        row['smoking_history'], row['bmi'], row['HbA1c_level'], row['blood_glucose_level']
                    )
                    for _, row in batch_df.iterrows()
                ])
                probs = model_predict(transformed_rows)
                batch_df['Probabilitas_Diabetes (%)'] = np.round(probs * 100, 2)
                batch_df['Status_Prediksi'] = np.where(probs >= optimal_threshold, 'Risiko Tinggi', 'Risiko Rendah')

                b1, b2, b3 = st.columns(3)
                total_p = len(batch_df)
                high_p = (batch_df['Status_Prediksi'] == 'Risiko Tinggi').sum()
                b1.metric("Total Pasien Diperiksa", f"{total_p} Pasien")
                b2.metric("Terindikasi Risiko Tinggi", f"{high_p} Pasien", f"{(high_p/total_p)*100:.1f}%", delta_color="inverse")
                b3.metric("Terindikasi Risiko Rendah", f"{total_p - high_p} Pasien")

                st.dataframe(batch_df, use_container_width=True)

                st.download_button(
                    "📥 Unduh Hasil Skrining Massal (.CSV)",
                    data=batch_df.to_csv(index=False),
                    file_name="hasil_skrining_massal_diabetes.csv",
                    mime="text/csv"
                )
            else:
                st.error(f"Format kolom tidak sesuai! Kolom wajib ada: {', '.join(required_cols)}")

# =====================================================================
# TAB 4: DASHBOARD EVALUASI & METODOLOGI MODEL
# =====================================================================
with tab4:
    st.subheader("📈 Ringkasan Evaluasi Eksperimen & Arsitektur Model")
    st.markdown(
        "Halaman ini merangkum hasil pengujian empiris dari metodologi yang digunakan dalam membangun sistem ini "
        "pada **Diabetes Prediction Dataset** (100.000 rekam medis klinis)."
    )

    eval_col1, eval_col2 = st.columns(2, gap="large")

    with eval_col1:
        st.markdown("#### 1️⃣ Komparasi Penanganan Imbalanced Data (Threshold 0.50)")
        comp_table = pd.DataFrame({
            "Skenario Eksperimen": ["Baseline (Data Asli)", "SMOTE Oversampling", "Class Weighting"],
            "Val Loss (BCE)": [0.0892, 0.2595, 0.2155],
            "Precision": [1.0000, 0.3651, 0.4944],
            "Recall": [0.6489, 0.9293, 0.8295],
            "F1-Score": [0.7870, 0.5243, 0.6195],
            "ROC-AUC": [0.9740, 0.9646, 0.9538]
        })
        st.dataframe(comp_table, use_container_width=True, hide_index=True)

        st.markdown("#### 2️⃣ Dampak Optimalisasi Decision Threshold pada Test Set")
        thresh_table = pd.DataFrame({
            "Ambang Batas (Threshold)": ["Default (0.5000)", f"Optimal PR-Curve ({optimal_threshold:.4f})"],
            "Precision (Kelas 1)": ["99.77%", "92.38%"],
            "Recall / Sensitivitas": ["69.42%", "72.48% (+3.06%)"],
            "F1-Score": ["81.87%", "81.23%"],
            "Akurasi Total": ["97.29%", "97.05%"]
        })
        st.dataframe(thresh_table, use_container_width=True, hide_index=True)

    with eval_col2:
        st.markdown("#### 3️⃣ Visualisasi Komparasi Metrik Antar Skenario")
        fig_eval = go.Figure(data=[
            go.Bar(name='Precision', x=comp_table['Skenario Eksperimen'], y=comp_table['Precision'], marker_color='#3B82F6'),
            go.Bar(name='Recall', x=comp_table['Skenario Eksperimen'], y=comp_table['Recall'], marker_color='#10B981'),
            go.Bar(name='F1-Score', x=comp_table['Skenario Eksperimen'], y=comp_table['F1-Score'], marker_color='#00D4B6'),
            go.Bar(name='ROC-AUC', x=comp_table['Skenario Eksperimen'], y=comp_table['ROC-AUC'], marker_color='#F59E0B')
        ])
        fig_eval.update_layout(
            barmode='group',
            yaxis_range=[0, 1.08],
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=340,
            margin=dict(l=10, r=10, t=20, b=10)
        )
        st.plotly_chart(fig_eval, use_container_width=True)

    st.info(
        "**Arsitektur Jaringan Saraf (MLP):** Input Layer (13 Fitur Terenkode) $\\rightarrow$ Dense(64, ReLU) + BatchNorm + Dropout(0.3) "
        "$\\rightarrow$ Dense(32, ReLU) + BatchNorm + Dropout(0.2) $\\rightarrow$ Dense(16, ReLU) $\\rightarrow$ Output Layer Dense(1, Sigmoid). "
        "Dilatih dengan Adam Optimizer ($lr = 0.001$) dan Early Stopping."
    )

# =====================================================================
# TAB 5: TENTANG PROYEK (INFORMASI APLIKASI, MODEL, TECH STACK & TIM)
# =====================================================================
with tab5:
    st.subheader("ℹ️ Tentang Proyek — GlucoGuard AI")
    st.markdown(
        "**Optimalisasi Deteksi Dini Risiko Penyakit Diabetes Menggunakan Arsitektur Multi-Layer Perceptron (MLP) "
        "dengan Penanganan Ketimpangan Data, Threshold Tuning, dan Explainable AI (SHAP)**"
    )

    # Baris 1: Deskripsi Proyek & Fitur Utama
    about_c1, about_c2 = st.columns(2, gap="large")
    with about_c1:
        st.markdown("""
        <div class="about-card">
            <h4>🎯 Deskripsi Singkat Proyek</h4>
            <p style="color:#E2E8F0; line-height:1.6;">
                <b>GlucoGuard AI</b> adalah sistem pendukung keputusan klinis (<i>Clinical Decision Support System</i>) 
                berbasis web yang dirancang untuk membantu tenaga kesehatan maupun masyarakat umum melakukan skrining dini risiko diabetes melitus secara cepat, objektif, dan terukur.
            </p>
            <p style="color:#B8C7D9; line-height:1.6; margin-bottom:0;">
                Sistem ini mengatasi dua tantangan utama dalam pemodelan data medis: <b>ketimpangan kelas (imbalanced data)</b> 
                yang sering memicu kesalahan negatif palsu (<i>false negative</i>), serta sifat kotak hitam (<i>black-box</i>) 
                pada jaringan saraf tiruan dengan menyajikan penjelasan matematis di balik setiap prediksi.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with about_c2:
        st.markdown("""
        <div class="about-card">
            <h4>✨ Fitur Utama Aplikasi</h4>
            <ul style="color:#E2E8F0; line-height:1.75; padding-left:20px; margin-bottom:0;">
                <li><b>Skrining Risiko Real-Time & Gauge Meter:</b> Menghitung probabilitas risiko diabetes berbasis ambang batas terkalibrasi (35.62%) serta evaluasi indikator HbA1c, Glukosa, dan BMI.</li>
                <li><b>Explainable AI (SHAP Waterfall):</b> Memvisualisasikan kontribusi setiap variabel klinis pasien terhadap keputusan akhir model.</li>
                <li><b>Simulasi Intervensi (What-If Analysis):</b> Menguji proyeksi penurunan risiko apabila pasien menurunkan HbA1c, glukosa darah, atau berat badan.</li>
                <li><b>Skrining Massal (Batch CSV):</b> Memproses prediksi untuk puluhan hingga ratusan data pasien sekaligus dalam satu unggahan file CSV.</li>
                <li><b>Ekspor Laporan Otomatis:</b> Mengunduh ringkasan hasil skrining individu (.TXT) maupun rekap massal (.CSV).</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Baris 2: Informasi Model & Dataset
    about_c3, about_c4 = st.columns(2, gap="large")
    with about_c3:
        st.markdown("""
        <div class="about-card">
            <h4>🧠 Informasi Model & Metodologi</h4>
            <ul style="color:#E2E8F0; line-height:1.7; padding-left:20px; margin-bottom:0;">
                <li><b>Algoritma Utama:</b> Deep Learning — Multi-Layer Perceptron (MLP) Classifier dengan 3 <i>Hidden Layers</i> (64-32-16 neuron), aktivasi ReLU, <i>Batch Normalization</i>, dan <i>Dropout</i>.</li>
                <li><b>Penanganan Imbalanced Data:</b> Komparasi empiris antara <i>Baseline</i>, <i>SMOTE Oversampling</i>, dan <i>Class Weighting</i> secara ketat pada <i>Training Set</i> (anti <i>data leakage</i>).</li>
                <li><b>Decision Threshold Tuning:</b> Kalibrasi ambang batas keputusan dari bawaan <code>0.5000</code> menjadi <code>0.3562</code> berdasarkan kurva <i>Precision-Recall</i> untuk menekan <i>False Negative</i>.</li>
                <li><b>Interpretabilitas Model:</b> Integrasi <i>SHAP KernelExplainer</i> untuk transparansi bobot fitur lokal maupun global.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with about_c4:
        st.markdown("""
        <div class="about-card">
            <h4>📊 Informasi Dataset</h4>
            <p style="color:#E2E8F0; line-height:1.6;">
                Model dilatih dan dievaluasi menggunakan <b>Diabetes Prediction Dataset</b> (open-source benchmark dataset dari Kaggle) 
                yang terdiri dari <b>100.000 rekam medis klinis</b> pasien yang telah dianonimkan (8.5% kasus positif dan 91.5% kasus negatif).
            </p>
            <ul style="color:#B8C7D9; line-height:1.65; padding-left:20px; margin-bottom:0;">
                <li><b>Atribut Demografis & Gaya Hidup:</b> Usia (<i>age</i>), Jenis Kelamin (<i>gender</i>), dan Riwayat Merokok (<i>smoking_history</i>).</li>
                <li><b>Riwayat Penyakit Penyerta:</b> Hipertensi (<i>hypertension</i>) dan Penyakit Jantung (<i>heart_disease</i>).</li>
                <li><b>Parameter Biometrik & Lab:</b> Indeks Massa Tubuh (<i>BMI</i>), Kadar Hemoglobin A1c (<i>HbA1c_level</i>), dan Glukosa Darah (<i>blood_glucose_level</i>).</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Baris 3: Teknologi yang Digunakan (Tech Stack) - Kartu Utuh + Baca Selengkapnya
    st.markdown("#### 🛠️ Teknologi & Framework yang Digunakan")

    tech_items = [
        ("🐍", "Bahasa Pemrograman", "Python 3", "Bahasa inti untuk pemrosesan data, pelatihan model, dan logika aplikasi."),
        ("🧠", "Deep Learning Framework", "TensorFlow & Keras", "Membangun dan melatih arsitektur Multi-Layer Perceptron (MLP) biner."),
        ("🔍", "Explainable AI (XAI)", "SHAP (KernelExplainer)", "Menghitung kontribusi matematis setiap fitur klinis secara lokal & global."),
        ("📊", "Data & Machine Learning", "Pandas, NumPy, Scikit-Learn & Imblearn", "Pembersihan data, standardisasi fitur, SMOTE, dan evaluasi kurva PR/ROC."),
        ("🌐", "Web & Visualisasi Interaktif", "Streamlit, Plotly & Matplotlib", "Antarmuka web klinis responsif, gauge meter interaktif, dan grafik SHAP.")
    ]

    # Baris atas: 3 kartu lebar agar teks tampil penuh tanpa terpotong
    row1_cols = st.columns(3, gap="medium")
    for idx in range(3):
        icon, category, title, desc = tech_items[idx]
        with row1_cols[idx]:
            st.markdown(f"""
            <div class="about-card" style="padding: 16px 18px; margin-bottom: 12px;">
                <div style="font-size: 0.82rem; color: #00D4B6; font-weight: 600; margin-bottom: 4px;">{icon} {category}</div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF; margin-bottom: 6px;">{title}</div>
                <div style="font-size: 0.88rem; color: #B8C7D9; line-height: 1.45;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # Baris bawah: 2 kartu lebar
    row2_cols = st.columns(2, gap="medium")
    for idx in range(3, 5):
        icon, category, title, desc = tech_items[idx]
        with row2_cols[idx - 3]:
            st.markdown(f"""
            <div class="about-card" style="padding: 16px 18px; margin-bottom: 8px;">
                <div style="font-size: 0.82rem; color: #00D4B6; font-weight: 600; margin-bottom: 4px;">{icon} {category}</div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF; margin-bottom: 6px;">{title}</div>
                <div style="font-size: 0.88rem; color: #B8C7D9; line-height: 1.45;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # Fitur Baca Selengkapnya untuk detail teknis arsitektur & pustaka
    with st.expander("📖 Baca Selengkapnya: Rincian Peran Setiap Teknologi dalam Pipeline Sistem"):
        st.markdown("""
        * **Python 3:** Lingkungan eksekusi utama yang menghubungkan *pipeline* prapemrosesan numerik, inferensi jaringan saraf, hingga antarmuka pengguna.
        * **TensorFlow & Keras:** Digunakan untuk menyusun arsitektur *Sequential* MLP (64-32-16 neuron), fungsi aktivasi *ReLU* dan *Sigmoid*, lapisan *Batch Normalization*, *Dropout*, serta fungsi kerugian *Binary Cross-Entropy*.
        * **SHAP (*SHapley Additive exPlanations*):** Menggunakan pendekatan teori permainan kooperatif (*KernelExplainer*) dengan 100 sampel latar belakang (*background dataset*) untuk menguraikan kontribusi masing-masing indikator klinis pasien.
        * **Scikit-Learn, Imbalanced-Learn, Pandas & NumPy:** Mengelola pembagian data terstratifikasi (*Stratified Split*), penyekalaan *Z-Score* (*StandardScaler*), eksperimen *SMOTE*, penghitungan kurva *Precision-Recall*, serta transformasi matriks cepat saat aplikasi dijalankan.
        * **Streamlit, Plotly & Matplotlib:** Menyajikan *dashboard* interaktif multi-tab, *gauge chart* probabilitas risiko dinamis, grafik perbandingan simulasi *What-If*, serta visualisasi *SHAP Waterfall Plot*.
        """)

    # Baris 4: Anggota Kelompok Pengembang
    st.markdown("#### 👥 Tim Pengembang Proyek")
    st.caption("Program Studi Teknik Informatika — Universitas Nusa Putra")

    m_col1, m_col2, m_col3 = st.columns(3, gap="large")
    with m_col1:
        st.markdown("""
        <div class="team-card">
            <div style="font-size:2.4rem;">👨‍💻</div>
            <div class="team-name">Ikhsanuril Al Fayadh</div>
            <div class="team-nim">NIM: 20240040070</div>
            <div class="team-role">Teknik Informatika — Universitas Nusa Putra</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col2:
        st.markdown("""
        <div class="team-card">
            <div style="font-size:2.4rem;">👩‍💻</div>
            <div class="team-name">Nazwa Azzahra</div>
            <div class="team-nim">NIM: 20240040230</div>
            <div class="team-role">Teknik Informatika — Universitas Nusa Putra</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col3:
        st.markdown("""
        <div class="team-card">
            <div style="font-size:2.4rem;">👨‍💻</div>
            <div class="team-name">Sukma Sumap Praja</div>
            <div class="team-nim">NIM: 20240040222</div>
            <div class="team-role">Teknik Informatika — Universitas Nusa Putra</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.warning(
        "**Disclaimer Medis:** Aplikasi ini dikembangkan sebagai proyek akademik Data Science dan berfungsi sebagai "
        "alat bantu skrining risiko awal. Hasil prediksi dari model kecerdasan buatan tidak menggantikan diagnosis, "
        "saran, maupun pemeriksaan klinis langsung oleh dokter atau tenaga medis profesional."
    )