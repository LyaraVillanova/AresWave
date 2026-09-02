![banner](https://github.com/user-attachments/assets/555ebae5-0d10-47dd-a00e-19ed2b748dd7)

# AresWave

[![Python](https://img.shields.io/badge/python-3.9--3.11-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

AresWave is a Python package for source parameter estimation of martian seismic events (marsquakes) through waveform fitting between real SEIS data and synthetic seismograms.  
It provides an end-to-end workflow that includes data preprocessing, synthetic generation, optimization (PSO), Bayesian depth inference, visualization, and result export.



## Functions

- Preprocess real waveforms (e.g., InSight/SEIS; SAC format)
- Generate synthetics (DSMpy engine embedded)
- Compare real vs. synthetic seismograms (P/S windows; correlation/MSE)
- Optimize focal mechanism & depth via PSO
- Infer depth uncertainty via Bayesian module (S–P)
- Visualize: waveforms, beachballs, and fault-plane diagrams
- Organize outputs: figures, CSV summaries, and synthetic traces



## Installation

AresWave is designed to run in a Linux environment. For Windows users, WSL (Windows Subsystem for Linux) with Ubuntu is recommended. Visual Studio Code with WSL support can be used for development, but it is not required.

The package metadata supports Python 3.9–3.11. The installation and compiled extensions have been validated with Python 3.10.18.

### 1) Install system dependencies

Inside Ubuntu or another Debian-based Linux environment:

```bash
sudo apt-get update
sudo apt-get install -y python3-venv gcc gfortran openmpi-bin libopenmpi-dev
```

You also need Python 3.9, 3.10, or 3.11, together with `pip` and `venv`.

### 2) Clone AresWave

```bash
git clone https://github.com/LyaraVillanova/AresWave.git
cd AresWave
```

### 3) Create a virtual environment

Using Python 3.10:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 4) Install AresWave

From the repository root:

```bash
python -m pip install .
```

This installs the Python dependencies and compiles the modified DSMpy Fortran extensions bundled with AresWave.

### 5) Verify the installation

```bash
python - <<'PYTEST'
import areswave
import dsmpy
from dsmpy.flib import tish, tipsv

print("AresWave:", areswave.__file__)
print("DSMpy:", dsmpy.__file__)
print("Fortran extensions: OK")
PYTEST
```

If the imports complete without an exception, the installation is working correctly.

## Requirements

Runtime dependencies are maintained in `requirements.txt`. They currently include:

```text
arviz
corner
matplotlib
numpy<2
obspy
pandas
pymc
pyswarms
scipy
tqdm
mpi4py
joblib
sqlalchemy
requests
geographiclib
```

NumPy is currently restricted to versions below 2 because the bundled DSMpy Fortran extensions use the legacy NumPy/F2PY build stack.

The modified DSMpy code used by AresWave is bundled with this repository and must not be installed separately.

AresWave builds on DSMpy, originally developed by Anselme Borgeaud and Kensuke Konishi. The bundled version contains modifications required for the Mars workflows used by AresWave.


## How to use
```python
import numpy as np
import matplotlib.pyplot as plt
from obspy.taup import TauPyModel
from obspy import Stream, read, UTCDateTime
from dsmpy import seismicmodel_Mars
from areswave.synthetics_function import generate_synthetics, apply_filter
from dsmpy.event_Mars import Event, MomentTensor
from dsmpy.station_Mars import Station
from areswave.denoising import polarization_filter
from scipy.signal import correlate
import os

# EVENT
event_id = 'mqs2019kxjd'
name = 'S0185a'
latitude = 41.6
longitude = 90.1
distance = 59.8
baz = 92.0
magnitude = 3.1
depth = 30.37
time_p = UTCDateTime("2019-06-05T02:13:48")
time_s = UTCDateTime("2019-06-05T02:19:47")
centroid_time = UTCDateTime((time_p.timestamp + time_s.timestamp) / 2)

Mrr = 3.04e20
Mrt = -2.16e19
Mrp = -3.26e20
Mtt = -1.21e20
Mtp = -1.23e20
Mpp = -7.31e19
mt = MomentTensor(Mrr, Mrt, Mrp, Mtt, Mtp, Mpp)

event = Event(
    event_id=event_id,
    latitude=latitude,
    longitude=longitude,
    depth=depth,
    mt=mt,
    centroid_time=centroid_time.timestamp,
    source_time_function=None
)

# STATION
stations = [
    Station(name='ELYSE', network='XB', latitude=4.502384, longitude=135.623447),
]

# SEISMIC MODEL
seismic_model = seismicmodel_Mars.SeismicModel.tayak()
tlen = 1276.8
nspc = 1256
sampling_hz = 20

# REAL DATA
sac_folder_path = 'SAC'
sac_files = [
    os.path.join(sac_folder_path, 'S0185a_trlq_denois03.Z.sac'),
    os.path.join(sac_folder_path, 'S0185a_trlq_denois04.T.sac'),
    os.path.join(sac_folder_path, 'S0185a_trlq_denois05.R.sac')
]

if not all(os.path.exists(f) for f in sac_files):
    raise FileNotFoundError("Um ou mais arquivos SAC não foram encontrados. Verifique o caminho: " + sac_folder_path)

real_data_dict = {}
stream = Stream()
for sac_file in sac_files:
    tr = read(sac_file)[0]
    tr.detrend('linear')
    tr.taper(max_percentage=0.05)
    tr.resample(sampling_hz)
    stream += tr
    if sac_file.endswith(".Z.sac"):
        real_data_dict['Z'] = tr
    elif sac_file.endswith(".R.sac"):
        real_data_dict['R'] = tr
    elif sac_file.endswith(".T.sac"):
        real_data_dict['T'] = tr

real_data_list = [real_data_dict['Z'], real_data_dict['R'], real_data_dict['T']]

# SYNTHETICS
output = generate_synthetics(event, stations, seismic_model, tlen, nspc, sampling_hz)
ts = output.ts
output.write(root_path='synthetics/', format='sac')

u_Z_ELYSE_XB = output['Z', 'ELYSE_XB']
u_R_ELYSE_XB = output['R', 'ELYSE_XB']
u_T_ELYSE_XB = output['T', 'ELYSE_XB']

max_time = min(1500, ts[-1])
max_idx = np.searchsorted(ts, max_time)
ts = ts[:max_idx]

u_Z_ELYSE_XB = u_Z_ELYSE_XB[:max_idx]
u_R_ELYSE_XB = u_R_ELYSE_XB[:max_idx]
u_T_ELYSE_XB = u_T_ELYSE_XB[:max_idx]

u_Z_ELYSE_XB_filtered = apply_filter(u_Z_ELYSE_XB, sampling_hz)
u_R_ELYSE_XB_filtered = apply_filter(u_R_ELYSE_XB, sampling_hz)
u_T_ELYSE_XB_filtered = apply_filter(u_T_ELYSE_XB, sampling_hz)

synthetic_data = [u_Z_ELYSE_XB_filtered, u_R_ELYSE_XB_filtered, u_T_ELYSE_XB_filtered]
synthetic_data = polarization_filter(synthetic_data, sampling_hz)
synthetics = [synthetic_data[0], synthetic_data[1], synthetic_data[2]]

# TEMPORAL ALIGNMENT
model = TauPyModel(model="models/TAYAK.npz")
arrivals = model.get_travel_times(source_depth_in_km=depth, distance_in_degree=distance, phase_list=['P', 'S'])
if not arrivals:
    raise ValueError('Não foi possível calcular o tempo de chegada da onda P.')
travel_time_p = arrivals[0].time
travel_time_s = arrivals[1].time
ts_adjusted = ts - travel_time_p
tss_adjusted = ts - travel_time_s
for tr in real_data_list:
    trace_start_time = tr.stats.starttime
    tr.data = tr.data[:max_idx]
    shift_real_p = (time_p - trace_start_time)
    shift_real_s = (time_s - trace_start_time)
    tr.times_shifted_p = tr.times() - shift_real_p
    tr.times_shifted_s = tr.times() - shift_real_s
    tr.data = apply_filter(tr.data, sampling_hz)
real_data_dict = {
    'R': real_data_list[0],
    'Z': real_data_list[1],
    'T': real_data_list[2]
}
real_data_list = [real_data_dict['Z'], real_data_dict['R'], real_data_dict['T']]
synthetic_dict = {
    'Z': synthetics[0],
    'T': synthetics[1],
    'R': synthetics[2]
}
synthetics = [synthetic_dict['Z'], synthetic_dict['R'], synthetic_dict['T']]
components = ['Z', 'R', 'T']

# PLOT
components = ['Z', 'R', 'T']
fig, axs = plt.subplots(3, 4, figsize=(28, 12))
for i, (comp, synthetic, real_data) in enumerate(zip(components, synthetics, real_data_list)):
    synthetic_norm = synthetic / np.max(np.abs(synthetic))
    real_data_norm = real_data.data / np.max(np.abs(real_data.data))
    cross_corr = correlate(synthetic_norm, real_data_norm, mode='full')
    cross_corr /= np.max(cross_corr)
    corr_coefficient = np.corrcoef(synthetic_norm, real_data_norm)[0, 1]

    axs[i, 0].plot(ts_adjusted, synthetic_norm, label=f'Synthetic {comp}', color='blue', alpha=0.7)
    axs[i, 0].axvline(x=travel_time_p, linestyle='--', color='black', label='P-wave')
    axs[i, 0].axvline(x=travel_time_s, linestyle='--', color='black', label='S-wave')
    axs[i, 0].set_xlim([-100, 700])
    axs[i, 0].set_title(f'Synthetic Component {comp}')
    axs[i, 0].legend(loc='lower right')
    axs[i, 0].text(0.05, 0.95, f'Correlation: {corr_coefficient:.2f}', transform=axs[i, 0].transAxes)
    axs[i, 1].plot(real_data.times_shifted_p, real_data_norm, label=f'Real {comp}', color='red', alpha=0.7)
    axs[i, 1].axvline(x=0, linestyle='--', color='black', label='P-wave')
    axs[i, 1].axvline(x=shift_real_s - shift_real_p, linestyle='--', color='black', label='S-wave')
    axs[i, 1].set_xlim([-100, 700])
    axs[i, 1].set_title(f'Real Component {comp}')
    axs[i, 1].legend(loc='lower right')

    # Zoom - P arrival
    xlim_min, xlim_max = -5, 5
    idx_min = max(0, int((xlim_min - ts_adjusted[0]) / (ts_adjusted[1] - ts_adjusted[0])))
    idx_max = min(len(ts_adjusted), int((xlim_max - ts_adjusted[0]) / (ts_adjusted[1] - ts_adjusted[0])))
    corr_zoom = np.corrcoef(synthetic_norm[idx_min:idx_max], real_data_norm[idx_min:idx_max])[0, 1]
    axs[i, 2].plot(ts_adjusted, synthetic_norm, color='blue', alpha=0.7)
    axs[i, 2].plot(real_data.times_shifted_p, real_data_norm, color='red', alpha=0.7)
    axs[i, 2].axvspan(xlim_min, xlim_max, color='gray', alpha=0.2)
    axs[i, 2].set_xlim([-10, 10])
    axs[i, 2].set_ylim(-0.1, 0.1)
    axs[i, 2].set_title(f'Cross Correlation ({comp})\nCorr: {corr_zoom:.2f}')
    axs[i, 2].legend(loc='lower right')

    # Zoom - S arrival
    xlim_min, xlim_max = 355, 370
    idx_min = max(0, int((xlim_min - ts_adjusted[0]) / (ts_adjusted[1] - ts_adjusted[0])))
    idx_max = min(len(ts_adjusted), int((xlim_max - ts_adjusted[0]) / (ts_adjusted[1] - ts_adjusted[0])))
    corr_zoom = np.corrcoef(synthetic_norm[idx_min:idx_max], real_data_norm[idx_min:idx_max])[0, 1]
    axs[i, 3].plot(ts_adjusted, synthetic_norm, color='blue', alpha=0.7)
    axs[i, 3].plot(real_data.times_shifted_p, real_data_norm, color='red', alpha=0.7)
    axs[i, 3].axvspan(xlim_min, xlim_max, color='gray', alpha=0.2)
    axs[i, 3].set_xlim([350, 380])
    axs[i, 3].set_ylim(-0.5, 0.5)
    axs[i, 3].set_title(f'Detail View ({comp})\nCorr: {corr_zoom:.2f}')
    axs[i, 3].legend(loc='lower right')

plt.subplots_adjust(hspace=0.4, wspace=0.3)
plt.tight_layout()
plt.savefig('figs/output_cross_correlation.png')
```



## Publications
Villanova, L. S., & Genda, H. (2026). AresWave: estimation of marsquake source parameters by waveform fitting with stochastic optimization. Academia Earth and Planetary Science, 1(2). https://doi.org/10.20935/AcadEPS8376

Villanova, L. S., & Genda, H. (submitted). Constraints on the depth and tectonic nature of Martian seismic sources.


## Contact
If you have any questions, find a bug, or would like to collaborate, feel free to reach out!

**Lyara S. Villanova**  
villanova@elsi.jp  
[GitHub Issues](https://github.com/LyaraVillanova/AresWave/issues) — open a ticket or suggest improvements  
