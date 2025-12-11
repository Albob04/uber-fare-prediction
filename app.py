import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Uber Fare Prediction NYC", layout="wide")

# --- Load Model ---
@st.cache_resource
def load_model():
    return joblib.load('uber_model_complex.pkl')

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model tidak ditemukan. Jalankan train_model.py dulu.")
    st.stop()

# --- Fungsi Helper ---
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def check_airport(lat, lon):
    # Koordinat Bandara NYC
    airports = {
        'JFK': (40.6413, -73.7781),
        'LGA': (40.7769, -73.8740),
        'EWR': (40.6895, -74.1745)
    }
    is_jfk = 1 if haversine_distance(lat, lon, *airports['JFK']) < 1.5 else 0
    is_lga = 1 if haversine_distance(lat, lon, *airports['LGA']) < 1.5 else 0
    is_ewr = 1 if haversine_distance(lat, lon, *airports['EWR']) < 1.5 else 0
    return is_jfk, is_lga, is_ewr

def reset_map():
    st.session_state.pickup = None
    st.session_state.dropoff = None

# --- Inisialisasi State ---
if 'pickup' not in st.session_state:
    st.session_state.pickup = None
if 'dropoff' not in st.session_state:
    st.session_state.dropoff = None

# ==========================================
# JUDUL & PETA (BAGIAN ATAS)
# ==========================================
st.title("🚖 Prediksi Harga Uber NYC")
st.markdown("---")

# Instruksi Peta
col_instr, col_reset = st.columns([4, 1])
with col_instr:
    st.info("👇 **CARA PAKAI PETA:** Klik 1x untuk titik **JEMPUT (Hijau)**, lalu klik 1x lagi untuk titik **TUJUAN (Merah)**.")
with col_reset:
    # Tombol Reset Peta
    st.button("🔄 Reset Peta", on_click=reset_map, use_container_width=True)

# Tampilan Peta
m = folium.Map(location=[40.75, -73.98], zoom_start=12)

# Menampilkan Marker jika ada di session state
if st.session_state.pickup:
    folium.Marker(
        [st.session_state.pickup['lat'], st.session_state.pickup['lng']], 
        popup="Jemput", icon=folium.Icon(color="green", icon="play")
    ).add_to(m)

if st.session_state.dropoff:
    folium.Marker(
        [st.session_state.dropoff['lat'], st.session_state.dropoff['lng']], 
        popup="Tujuan", icon=folium.Icon(color="red", icon="stop")
    ).add_to(m)
    
    # Garis Rute
    folium.PolyLine([
        [st.session_state.pickup['lat'], st.session_state.pickup['lng']],
        [st.session_state.dropoff['lat'], st.session_state.dropoff['lng']]
    ], color="blue", weight=2.5, opacity=0.8).add_to(m)

# Render Peta
output = st_folium(m, height=450, use_container_width=True)

# Logika Klik Peta
if output and output['last_clicked']:
    if not st.session_state.pickup:
        st.session_state.pickup = output['last_clicked']
        st.rerun() # Refresh otomatis setelah klik
    elif not st.session_state.dropoff:
        st.session_state.dropoff = output['last_clicked']
        st.rerun()

# ==========================================
# INPUT USER (BAGIAN TENGAH - BAWAH PETA)
# ==========================================
st.markdown("### ⚙️ Pengaturan Perjalanan")

# Membuat 3 Kolom agar rapi
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**1. Penumpang & Waktu**")
    passenger_count = st.number_input("Jumlah Penumpang", min_value=1, max_value=6, value=1)
    hour = st.slider("Jam Berapa? (0-23)", 0, 23, 12, help="Jam mempengaruhi kemacetan dan harga.")

with c2:
    st.markdown("**2. Hari & Tanggal**")
    is_holiday_input = st.checkbox("Apakah Hari Libur Nasional?", value=False)
    # Kami set default bulan Juni 2015 agar model tetap jalan (data training tahun 2015)
    month = 6 
    year = 2015
    # Jika libur, asumsikan seperti hari Minggu (6), jika tidak Senin (0) untuk simplifikasi
    day_of_week = 6 if is_holiday_input else 0 
    is_holiday = 1 if is_holiday_input else 0

with c3:
    st.markdown("**3. Kondisi Cuaca**")
    weather_type = st.selectbox("Pilih Cuaca:", ["Normal / Cerah", "Hujan Sedang", "Hujan Deras / Badai", "Salju"])
    
    # Konversi input user ke angka (PRCP, SNOW, TMAX)
    if weather_type == "Hujan Sedang":
        input_prcp = 0.2; input_snow = 0.0; input_temp = 65
    elif weather_type == "Hujan Deras / Badai":
        input_prcp = 1.5; input_snow = 0.0; input_temp = 60
    elif weather_type == "Salju":
        input_prcp = 0.1; input_snow = 2.0; input_temp = 30
    else: # Normal
        input_prcp = 0.0; input_snow = 0.0; input_temp = 75

# ==========================================
# HASIL PREDIKSI (BAGIAN PALING BAWAH)
# ==========================================
st.markdown("---")
st.markdown("### 💰 Hasil Prediksi")

# Cek apakah koordinat sudah lengkap
if st.session_state.pickup and st.session_state.dropoff:
    # Ambil koordinat
    p_lat = st.session_state.pickup['lat']
    p_lon = st.session_state.pickup['lng']
    d_lat = st.session_state.dropoff['lat']
    d_lon = st.session_state.dropoff['lng']
    
    # Hitung Jarak
    dist_km = haversine_distance(p_lat, p_lon, d_lat, d_lon)
    
    # Cek Airport
    p_jfk, p_lga, p_ewr = check_airport(p_lat, p_lon)
    d_jfk, d_lga, d_ewr = check_airport(d_lat, d_lon)
    is_JFK = 1 if (p_jfk or d_jfk) else 0
    is_LGA = 1 if (p_lga or d_lga) else 0
    is_EWR = 1 if (p_ewr or d_ewr) else 0

    # Tampilkan Info Rute
    col_res1, col_res2 = st.columns([1, 3])
    with col_res1:
        st.metric("Jarak Tempuh", f"{dist_km:.2f} km")
        if is_JFK or is_LGA or is_EWR:
            st.warning("✈️ Rute Bandara Terdeteksi!")
    
    with col_res2:
        # TOMBOL PREDIKSI BESAR DI BAWAH
        if st.button("HITUNG HARGA SEKARANG", type="primary", use_container_width=True):
            
            # Susun Dataframe input sesuai urutan training
            input_data = pd.DataFrame([[
                dist_km, passenger_count, 
                hour, day_of_week, month, year, is_holiday,
                is_JFK, is_LGA, is_EWR,
                input_prcp, input_snow, input_temp
            ]], columns=[
                'distance_km', 'passenger_count', 'hour', 'day_of_week', 'month', 'year', 'is_holiday',
                'is_JFK', 'is_LGA', 'is_EWR', 'PRCP', 'SNOW', 'TMAX'
            ])
            
            # Prediksi
            prediction = model.predict(input_data)[0]
            
            # Tampilkan Hasil
            st.success(f"Estimasi Harga Perjalanan: **${prediction:.2f}**")

else:
    st.warning("⚠️ Harap pilih lokasi **Jemput** dan **Tujuan** di peta terlebih dahulu di bagian atas.")
