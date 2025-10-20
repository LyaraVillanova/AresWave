import numpy as np
from obspy import UTCDateTime
from dsmpy import seismicmodel_Mars
from areswave.gridsearch import main_analysis
import os

# Event
event_id = 'mqs2019kxjd'
name = 'S0185a'
latitude = 41.59816
longitude = 90.13083
distance = 59.8  #degrees
baz = 322.7
magnitude = 3.1
depth = 24.1  #km
time_p = UTCDateTime("2019-06-05T02:13:40")
time_s = UTCDateTime("2019-06-05T02:19:42")
centroid_time = UTCDateTime("2019-06-05 02:06:37")

# Initial Moment Tensor
Mrr = -2.8e20
Mrt = -1.9e20
Mrp = -1.3e20
Mtt = -1.4e20
Mtp = -5.3e20
Mpp = 1.8e20

# Directories
sac_folder_path = '/home/lyara/areswave/dsmpy/SAC'
model_directory = '/home/lyara/areswave/dsmpy/models'
output_waveform_fig_path = '/home/lyara/areswave/dsmpy/figs/gridsearch_cross_correlation_S0185a.png'
output_grid_search_fig_path = '/home/lyara/areswave/dsmpy/figs/gridsearch_cross_correlation_depths_S0185a.png'
os.makedirs(os.path.dirname(output_waveform_fig_path), exist_ok=True)
os.makedirs(os.path.dirname(output_grid_search_fig_path), exist_ok=True)

# Seismic parameters
tlen = 1276.8
nspc = 256
sampling_hz = 20
seismic_model = seismicmodel_Mars.SeismicModel.test2()

# Grid search parameters
depth_range = np.arange(5, 100, 5)
strike_range = np.arange(0, 360, 10)
dip_range = np.arange(0, 90, 10)
rake_range = np.arange(-180, 180, 10)
frequency_range = (0.1, 1.0)
frequency_interval = 0.1

# Main
if __name__ == "__main__":
    print("Starting grid search analisys to event S0185a...\n")
    try:
        best_params, min_variation = main_analysis(
            event_id, latitude, longitude, distance, baz, magnitude, depth, time_p, time_s, centroid_time,Mrr, Mrt, Mrp, Mtt, Mtp, Mpp,
            sac_folder_path, tlen, nspc, sampling_hz, model_directory, seismic_model, depth_range, strike_range, dip_range, rake_range,
            frequency_range, frequency_interval, output_waveform_fig_path, output_grid_search_fig_path)
        print("\n===== Final result =====")
        print(f"Best depth: {best_params['depth']} km")
        print(f"Strike: {best_params['strike']}°, Dip: {best_params['dip']}°, Rake: {best_params['rake']}°")
        print(f"Minimum variation: {min_variation:.5f}")
    except Exception as e:
        print("\n[ERROR]")
        print(str(e))