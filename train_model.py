import pandas as pd
import numpy as np
import holidays
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def add_airport_features(df):

    jfk_center = (40.6413, -73.7781)
    lga_center = (40.7769, -73.8740)
    ewr_center = (40.6895, -74.1745)
    

    radius = 1.5 
    
 
    df['pickup_dist_jfk'] = haversine_distance(df['pickup_latitude'], df['pickup_longitude'], jfk_center[0], jfk_center[1])
    df['dropoff_dist_jfk'] = haversine_distance(df['dropoff_latitude'], df['dropoff_longitude'], jfk_center[0], jfk_center[1])
    
    df['pickup_dist_lga'] = haversine_distance(df['pickup_latitude'], df['pickup_longitude'], lga_center[0], lga_center[1])
    df['dropoff_dist_lga'] = haversine_distance(df['dropoff_latitude'], df['dropoff_longitude'], lga_center[0], lga_center[1])
    
    df['pickup_dist_ewr'] = haversine_distance(df['pickup_latitude'], df['pickup_longitude'], ewr_center[0], ewr_center[1])
    df['dropoff_dist_ewr'] = haversine_distance(df['dropoff_latitude'], df['dropoff_longitude'], ewr_center[0], ewr_center[1])
    
    df['is_JFK'] = ((df['pickup_dist_jfk'] < radius) | (df['dropoff_dist_jfk'] < radius)).astype(int)
    df['is_LGA'] = ((df['pickup_dist_lga'] < radius) | (df['dropoff_dist_lga'] < radius)).astype(int)
    df['is_EWR'] = ((df['pickup_dist_ewr'] < radius) | (df['dropoff_dist_ewr'] < radius)).astype(int)
    
    return df

print("Loading Data...")
uber_df = pd.read_csv('uber.csv')
weather_df = pd.read_csv('weatherUber.csv')

print("Cleaning Outliers...")
uber_df.dropna(inplace=True)
uber_df = uber_df[(uber_df['fare_amount'] > 2.5) & (uber_df['fare_amount'] <= 200)]
uber_df = uber_df[(uber_df['passenger_count'] > 0) & (uber_df['passenger_count'] <= 6)]

uber_df = uber_df[
    (uber_df['pickup_latitude'].between(40.5, 41.5)) &
    (uber_df['pickup_longitude'].between(-74.5, -73.5)) &
    (uber_df['dropoff_latitude'].between(40.5, 41.5)) &
    (uber_df['dropoff_longitude'].between(-74.5, -73.5))
]

print("Feature Engineering...")
uber_df['pickup_datetime'] = pd.to_datetime(uber_df['pickup_datetime'].str.replace(' UTC', ''), errors='coerce')


uber_df['hour'] = uber_df['pickup_datetime'].dt.hour
uber_df['day_of_week'] = uber_df['pickup_datetime'].dt.dayofweek
uber_df['month'] = uber_df['pickup_datetime'].dt.month
uber_df['year'] = uber_df['pickup_datetime'].dt.year

us_holidays = holidays.US(state='NY')
uber_df['is_holiday'] = uber_df['pickup_datetime'].dt.normalize().apply(lambda x: 1 if x in us_holidays else 0)

uber_df['distance_km'] = haversine_distance(
    uber_df['pickup_latitude'], uber_df['pickup_longitude'],
    uber_df['dropoff_latitude'], uber_df['dropoff_longitude']
)

uber_df = uber_df[uber_df['distance_km'] > 0.1]


uber_df = add_airport_features(uber_df)


print("Merging Weather...")
weather_df['DATE'] = pd.to_datetime(weather_df['DATE'])
weather_df['PRCP'] = pd.to_numeric(weather_df['PRCP'], errors='coerce').fillna(0)
weather_df['SNOW'] = pd.to_numeric(weather_df['SNOW'], errors='coerce').fillna(0)
weather_df['TMAX'] = pd.to_numeric(weather_df['TMAX'], errors='coerce').fillna(60)

uber_df['DATE'] = uber_df['pickup_datetime'].dt.normalize()
data = pd.merge(uber_df, weather_df[['DATE', 'PRCP', 'SNOW', 'TMAX']], on='DATE', how='left')
data.dropna(inplace=True)


print("Training Multi Linear Regression...")
features = [
    'distance_km', 'passenger_count', 
    'hour', 'day_of_week', 'month', 'year', 'is_holiday',
    'is_JFK', 'is_LGA', 'is_EWR',
    'PRCP', 'SNOW', 'TMAX'
]

X = data[features]
y = data['fare_amount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")


joblib.dump(model, 'uber_model_complex.pkl')
print("Model saved as 'uber_model_complex.pkl'")