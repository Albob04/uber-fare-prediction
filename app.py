import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium
import datetime
import holidays

# --- Konfigurasi ---
st.set_page_config(page_title="Uber Fare Prediction NYC (Complete)", layout="wide")

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
    # Cek apakah koordinat dekat bandara (1.5km radius)
    airports = {
        'JFK': (40.6413, -73.7781),
        'LGA': (40.7769, -73.8740),
        'EWR': (40.6895, -74.1745)
    }
    is_jfk, is_lga, is_ewr = 0, 0, 0
    
    if haversine_distance(lat, lon, *airports['JFK']) < 1.5: is_jfk = 1
    if haversine_distance(lat, lon, *airports['LGA']) < 1.5: is_lga = 1
    if haversine_distance(lat, lon, *airports['EWR']) < 1.5: is_ewr = 1
    
    return is_jfk, is_lga, is_ewr

# --- UI Sidebar ---
st.title("🚖 NYC Uber Fare Estimator")
st.markdown("Prediksi harga Uber menggunakan Multi Linear Regression dengan fitur **Jarak, Waktu, Bandara, & Cuaca**.")

with st.sidebar:
    st.header("1. Detail Perjalanan")
    passenger_count = st.slider("Penumpang", 1, 6, 1)
    
    # Input Waktu (Penting untuk Hour/Holiday)
    pickup_date = st.date_input("Tanggal", datetime.date(2015, 6, 1))
    pickup_time = st.time_input("Jam", datetime.time(12, 0))
    
    st.header("2. Kondisi Cuaca")
    weather_type = st.radio("Cuaca saat itu:", ["Normal/Cerah", "Hujan", "Salju"])
    
    # Set default values berdasarkan kategori user
    if weather_type == "Hujan":
        input_prcp = 0.5; input_snow = 0.0; input_temp = 60
    elif weather_type == "Salju":
        input_prcp = 0.1; input_snow = 3.0; input_temp = 30
    else:
        input_prcp = 0.0; input_snow = 0.0; input_temp = 75

    if st.button("Reset Peta"):
        if 'pickup' in st.session_state: del st.session_state['pickup']
        if 'dropoff' in st.session_state: del st.session_state['dropoff']
        st.experimental_rerun()

# --- Peta ---
st.subheader("📍 Pilih Rute di Peta")
col1, col2 = st.columns([3, 1])

with col1:
    if 'pickup' not in st.session_state: st.session_state.pickup = None
    if 'dropoff' not in st.session_state: st.session_state.dropoff = None

    m = folium.Map(location=[40.73, -73.98], zoom_start=11)
    
    if st.session_state.pickup:
        folium.Marker([st.session_state.pickup['lat'], st.session_state.pickup['lng']], 
                      popup="Pickup", icon=folium.Icon(color="green", icon="play")).add_to(m)
    if st.session_state.dropoff:
        folium.Marker([st.session_state.dropoff['lat'], st.session_state.dropoff['lng']], 
                      popup="Dropoff", icon=folium.Icon(color="red", icon="stop")).add_to(m)

    output = st_folium(m, height=500, use_container_width=True)

    if output and output['last_clicked']:
        if not st.session_state.pickup:
            st.session_state.pickup = output['last_clicked']
            st.experimental_rerun()
        elif not st.session_state.dropoff:
            st.session_state.dropoff = output['last_clicked']
            st.experimental_rerun()

# --- Kalkulasi & Prediksi ---
with col2:
    st.info("Klik peta untuk set titik Jemput & Tujuan.")
    
    if st.session_state.pickup and st.session_state.dropoff:
        p_lat = st.session_state.pickup['lat']
        p_lon = st.session_state.pickup['lng']
        d_lat = st.session_state.dropoff['lat']
        d_lon = st.session_state.dropoff['lng']
        
        # 1. Hitung Jarak
        dist_km = haversine_distance(p_lat, p_lon, d_lat, d_lon)
        st.write(f"**Jarak:** {dist_km:.2f} km")
        
        # 2. Ekstrak Waktu
        p_datetime = datetime.datetime.combine(pickup_date, pickup_time)
        hour = p_datetime.hour
        day_of_week = p_datetime.weekday()
        month = p_datetime.month
        year = p_datetime.year
        
        # 3. Cek Holiday
        us_holidays = holidays.US(state='NY')
        is_holiday = 1 if p_datetime in us_holidays else 0
        if is_holiday: st.warning("🎉 Hari Libur Terdeteksi!")
        
        # 4. Cek Airport (Pickup OR Dropoff is Airport)
        p_jfk, p_lga, p_ewr = check_airport(p_lat, p_lon)
        d_jfk, d_lga, d_ewr = check_airport(d_lat, d_lon)
        
        is_JFK = 1 if (p_jfk or d_jfk) else 0
        is_LGA = 1 if (p_lga or d_lga) else 0
        is_EWR = 1 if (p_ewr or d_ewr) else 0
        
        if is_JFK: st.caption("✈️ Rute Bandara JFK")
        if is_LGA: st.caption("✈️ Rute Bandara LaGuardia")
        if is_EWR: st.caption("✈️ Rute Bandara Newark")
        
        # 5. Prediksi
        # Fitur: distance_km, passenger_count, hour, day_of_week, month, year, is_holiday, is_JFK, is_LGA, is_EWR, PRCP, SNOW, TMAX
        input_data = pd.DataFrame([[
            dist_km, passenger_count, 
            hour, day_of_week, month, year, is_holiday,
            is_JFK, is_LGA, is_EWR,
            input_prcp, input_snow, input_temp
        ]], columns=['distance_km', 'passenger_count', 'hour', 'day_of_week', 'month', 'year', 'is_holiday', 'is_JFK', 'is_LGA', 'is_EWR', 'PRCP', 'SNOW', 'TMAX'])
        
        if st.button("Hitung Harga"):
            prediction = model.predict(input_data)[0]
            st.success(f"### Estimasi: ${prediction:.2f}")