import pandas as pd
import numpy as np
import h5py

# =========================
# File paths
# =========================
traffic_file = "data/pems-bay.h5"
weather_file = "data/weather.csv"
output_file  = "data/pems-bay-with-weather.h5"

# =========================
# Load traffic data
# =========================
with h5py.File(traffic_file, 'r') as f:
    # PeMS HDF5 has structure: speed/block0_values
    traffic_data = f['speed/block0_values'][:]  # shape: (T, N)

# Add feature dimension for traffic
traffic_data = traffic_data[:, :, np.newaxis]  # shape: (T, N, 1)
num_timesteps, num_nodes, _ = traffic_data.shape

# =========================
# Load weather data
# =========================
weather_df = pd.read_csv(weather_file, parse_dates=['date'])
weather_features = ['temp_avg', 'temp_max', 'temp_min', 'Precipitation', 'rain_flag', 'mist_flag', 'haze_flag']
weather_array = weather_df[weather_features].values  # shape: (D, W)
num_days, num_weather_features = weather_array.shape

# =========================
# Upsample weather to match traffic timesteps
# =========================
steps_per_day = num_timesteps // num_days
weather_expanded = np.repeat(weather_array, steps_per_day, axis=0)

# Pad weather if shorter than traffic
if weather_expanded.shape[0] < num_timesteps:
    pad_len = num_timesteps - weather_expanded.shape[0]
    last_row = np.tile(weather_expanded[-1, :], (pad_len, 1))
    weather_expanded = np.vstack([weather_expanded, last_row])

# Repeat weather across all nodes
weather_input = np.repeat(weather_expanded[:, np.newaxis, :], num_nodes, axis=1)  # shape: (T, N, W)

# =========================
# Concatenate traffic + weather along features axis
# =========================
fused_input = np.concatenate([traffic_data, weather_input], axis=2)  # shape: (T, N, 1+W)

# =========================
# Save fused data
# =========================
with h5py.File(output_file, 'w') as f:
    f.create_dataset('data', data=fused_input)

print(f"Saved fused traffic+weather data to {output_file}, shape={fused_input.shape}")
