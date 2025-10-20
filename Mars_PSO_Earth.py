import numpy as np
import os
from obspy import Stream, read, UTCDateTime
from dsmpy import seismicmodel_Mars
from dsmpy.event import Event, MomentTensor
from dsmpy.station import Station
from areswave.PSOearth import create_cost_function, run_pso_with_restarts, plot_results

# Configuration: Event and Station
event_id = 'IV_AIO'
latitude, longitude, distance, depth = 37.8215, 14.5907, 58.5969, 31
magnitude = 3.3
time_p = UTCDateTime("2020-11-18T02:06:37")
time_s = UTCDateTime("2020-11-18T02:06:46")
centroid_time = UTCDateTime((time_p.timestamp + time_s.timestamp) / 2)
initial_mt = MomentTensor(Mrr=-2.8e20, Mrt=-1.9e20, Mrp=-1.3e20, Mtt=-1.4e20, Mtp=-5.3e20, Mpp=1.8e20)

event = Event(event_id=event_id, latitude=latitude, longitude=longitude,
              depth=depth, mt=initial_mt, centroid_time=centroid_time, source_time_function=None)

stations = [Station(name='AIO', network='IV', latitude=37.9712, longitude=15.233)]
seismic_model = seismicmodel_Mars.SeismicModel.prem()
sampling_hz = 100
nspc = 256
tlen = 110

# ------------------ LOAD & TRIM REAL DATA ------------------
sac_folder_path = '/home/lyara/areswave/SAC'
sac_files = [
    os.path.join(sac_folder_path, 'IV_AIO__HHR.sac'),
    os.path.join(sac_folder_path, 'IV_AIO__HHT.sac'),
    os.path.join(sac_folder_path, 'IV_AIO__HHZ.sac')
]
if not all(os.path.exists(f) for f in sac_files):
    raise FileNotFoundError("Um ou mais arquivos SAC não foram encontrados. Verifique o caminho: " + sac_folder_path)

stream = Stream()
for sac_file in sac_files:
    tr = read(sac_file)[0]
    tr.detrend('linear')
    tr.taper(max_percentage=0.05)
    tr.resample(sampling_hz)
    tr.trim(starttime=time_p - 10, endtime=time_p + 100)
    stream += tr

Z_trace = stream.select(channel='HHZ')[0]
R_trace = stream.select(channel='HHR')[0]
T_trace = stream.select(channel='HHT')[0]
real_data_list = [Z_trace, R_trace, T_trace]

# Normalize real data
from areswave.synthetics_function import normalize
for i in range(len(real_data_list)):
    real_data_list[i].data = normalize(real_data_list[i].data)

# Configuration: PSO
bounds = (np.array([20, 0, 0, -180]), np.array([56, 360, 90, 180]))
options = {'c1': 1.5, 'c2': 1.5, 'w': 0.5}
n_particles = 50
n_iterations = 40
restart_interval = 30

# Running the PSO
cost_func, tested_params_tracker = create_cost_function(
    event, stations, seismic_model, tlen, nspc, sampling_hz, real_data_list,
    magnitude, distance, time_p, time_s
)

best_position, best_cost = run_pso_with_restarts(
    cost_func, bounds, n_particles, n_iterations, restart_interval, options
)

# Figures and save results
fig_dir = "/home/lyara/areswave/figs"
plot_results(
    best_position, event, stations, seismic_model, tlen, nspc, sampling_hz,
    real_data_list, time_p, time_s, magnitude, distance, tested_params_tracker, fig_dir, event_id, n_particles, restart_interval
)
print(f"Processo concluído. Resultados e plots salvos em: {fig_dir}")