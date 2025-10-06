import pandas as pd
import numpy as np
import h5py

# Files
traffic_file = "data/pems-bay.h5"
weather_file = "data/weather.csv"
output_file  = "data/pems-bay-with-weather.h5"


# Load traffic
with h5py.File(traffic_file, 'r') as f:
    traffic_data = f['data'][:]
num_timesteps, num_nodes, num_features = traffic_data.shape

# Load weather
weather_df = pd.read_csv(weather_file, parse_dates=['date'])
weather_features = ['temp_avg', 'temp_max', 'temp_min', 'Precipitation', 'rain_flag', 'mist_flag', 'haze_flag']
weather_array = weather_df[weather_features].values
num_days, num_weather_features = weather_array.shape

# Upsample weather to 5-min steps
steps_per_day = num_timesteps // num_days
weather_expanded = np.repeat(weather_array, steps_per_day, axis=0)

# Repeat weather across all nodes
weather_input = np.repeat(weather_expanded[:, np.newaxis, :], num_nodes, axis=1)

# Concatenate traffic + weather
fused_input = np.concatenate([traffic_data, weather_input], axis=2)

# Save fused data
with h5py.File(output_file, 'w') as f:
    f.create_dataset('data', data=fused_input)
