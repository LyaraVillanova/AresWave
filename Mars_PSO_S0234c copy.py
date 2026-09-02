import numpy as np
import os
from obspy import Stream, read, UTCDateTime
from dsmpy import seismicmodel_Mars
from dsmpy.event_Mars import Event, MomentTensor
from dsmpy.station_Mars import Station
from areswave.PSO import create_cost_function, run_pso_with_restarts, plot_results

# Configuration: Event and Station
event_id = 'S0234c_mqs2019olnq'
latitude, longitude, distance, depth = 41.59816, 90.13083, 60.0, 30
magnitude = 2.9
time_p = UTCDateTime("2019-07-25T12:54:01")
time_s = UTCDateTime("2019-07-25T12:59:59")
centroid_time = UTCDateTime("2019-07-25T12:51:18")
initial_mt = MomentTensor(Mrr=-2.8e20, Mrt=-1.9e20, Mrp=-1.3e20, Mtt=-1.4e20, Mtp=-5.3e20, Mpp=1.8e20)

event = Event(event_id=event_id, latitude=latitude, longitude=longitude,
              depth=depth, mt=initial_mt, centroid_time=centroid_time, source_time_function=None)

stations = [Station(name='ELYSE', network='XB', latitude=4.502384, longitude=135.623447)]
seismic_model = seismicmodel_Mars.SeismicModel.test2()
sampling_hz = 20
nspc = 256
tlen = 400

# Upload and pre-processing real data
sac_folder_path = '/home/lyara/areswave/SAC'
sac_files = [
    os.path.join(sac_folder_path, 'S0234c_trlq_denois03.R.sac'),
    os.path.join(sac_folder_path, 'S0234c_trlq_denois04.T.sac'),
    os.path.join(sac_folder_path, 'S0234c_trlq_denois05.Z.sac')
]
if not all(os.path.exists(f) for f in sac_files):
    raise FileNotFoundError("Um ou mais arquivos SAC não foram encontrados. Verifique o caminho: " + sac_folder_path)

stream = Stream()
for sac_file in sac_files:
    tr = read(sac_file)[0]
    tr.detrend('linear')
    tr.taper(max_percentage=0.05)
    tr.resample(sampling_hz)
    tr.trim(starttime=time_p - 10, endtime=time_p + 380)
    stream += tr

Z_trace = stream.select(channel='BHZ')[0]
R_trace = stream.select(channel='BHR')[0]
T_trace = stream.select(channel='BHT')[0]
real_data_list = [Z_trace, R_trace, T_trace]

# Normalize real data
from areswave.synthetics_function import normalize
for i in range(len(real_data_list)):
    real_data_list[i].data = normalize(real_data_list[i].data)

# Configuration: PSO
bounds = (np.array([12, 0, 0, -180]), np.array([52, 360, 90, 180]))
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