![banner](https://github.com/user-attachments/assets/555ebae5-0d10-47dd-a00e-19ed2b748dd7)

# AresWave

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

O AresWave é um pacote Python para a estimativa de profundidade e mecanismo focal de eventos sísmicos marcianos (marsquakes), combinando modelagem de forma de onda (via DSMpy) com otimização estocástica por Particle Swarm Optimization (PSO). O código também inclui ferramentas para ajuste de modelos 1D e estimativas bayesianas de profundidade com base em tempos S–P.


## 🪐 Funcionalidades

- Geração de formas de onda sintéticas com DSMpy
- Comparação de formas de onda reais e sintéticas (P e S)
- Otimização de parâmetros fonte com PSO
- Estimativa bayesiana de profundidade via PyMC
- Inversão iterativa de modelos de velocidade 1D
- Teste com o evento S0185a (SEIS/InSight dataset)


## 🪐 Instalação

It is recommended the use of Visual Studio Code (https://code.visualstudio.com/) since it simplifies package management and usage. Additionally, install Ubuntu for Windows users (https://ubuntu.com/desktop/wsl). Once it is installed, run the following commands (tested on Windows 10 and 11 systems, with Python 3.10.12):


1) Open a PowerShell (in admin mode)

2) In the PowerShell, type
```
wsl --install
```

3) From the Ubuntu terminal, install python, gcc and openmpi
```
sudo apt-get update && apt-get install -y python3 python3-pip
sudo apt install python-is-python3
sudo apt-get install gcc
sudo apt-get install -y openmpi-bin libopenmpi-dev
```

4) Create a directory for a new python project (rename new_project as your preference), and open it in Visual Studio Code
```
cd ~/git
mkdir new_project
code new_project
```

5) Clone the repository
```
git clone https://github.com/LyaraVillanova/AresWave.git
```

6) Install requirements
```
python3 -m pip install -r requirements.txt
```

7) Install [*build*](https://pypi.org/project/build/),
```
pip install build
```

8) From the root directory ```dsmpy``` run
```
python -m build .
```
9) Now it can be installed with
```
pip install dist/*.tar.gz
```


## 🪐 Requisitos

AresWave requer as seguintes bibliotecas:

```
arviz
concurrent.futures
dsmpy
glob
matplotlib
numpy
obspy
os
pandas
pymc
pyswarm
scipy
```

> Recomenda-se Python 3.8 ou superior. Certifique-se de que o DSMpy está corretamente instalado a partir do repositório oficial: [https://github.com/afeborgeaud/dsmpy]
> Imenso agradecimento ao Anselme Borgeaud @afeborgeaud, desenvolvedor do DSMpy.


## 🪐 Como usar

```python
import numpy as np
import matplotlib.pyplot as plt
from obspy.taup import TauPyModel
from obspy import Stream
from dsmpy import seismicmodel_Mars
from dsmpy.synthetics_function import generate_synthetics, apply_filter
from dsmpy.event_Mars import Event, MomentTensor
from dsmpy.station_Mars import Station
from dsmpy.denoising import polarization_filter
from obspy import read, UTCDateTime
from scipy.signal import correlate
import glob
import os

# Definindo valores do evento
event_id = 'mqs2019kxjd'
name = 'S0185a'
latitude = 41.59816
longitude = 90.13083
distance = 59.8
baz = 92.0 #322.7
magnitude = 3.1
depth = 30 #24.1
time_p = UTCDateTime("2019-06-05T02:13:48")
time_s = UTCDateTime("2019-06-05T02:19:47")
centroid_time = UTCDateTime((time_p.timestamp + time_s.timestamp) / 2)

# Tensor de Momento
Mrr = -2.8e12
Mrt = -1.9e13
Mrp = -1.3e13
Mtt = -1.4e13
Mtp = -5.3e12
Mpp = 1.8e13
mt = MomentTensor(Mrr, Mrt, Mrp, Mtt, Mtp, Mpp)

# Crie o objeto Event
event = Event(
    event_id=event_id,
    latitude=latitude,
    longitude=longitude,
    depth=depth,
    mt=mt,
    centroid_time=centroid_time.timestamp,
    source_time_function=None
)

# Defina a estação ELYSE
stations = [
    Station(name='ELYSE', network='XB', latitude=4.502384, longitude=135.623447),
]

# Carregue o modelo sísmico
seismic_model = seismicmodel_Mars.SeismicModel.test()
tlen = 1276.8  # duração dos sintéticos (s)
nspc = 1256  # número de pontos no domínio da frequência
sampling_hz = 20  # frequência de amostragem para os sintéticos

# Caminho para os arquivos .sac
sac_folder_path = '/home/lyara/my_project/dsmpy-1/SAC'
sac_files = glob.glob(os.path.join(sac_folder_path, '*.sac'))
if not sac_files:
    raise FileNotFoundError(f"Nenhum arquivo .sac encontrado na pasta: {sac_folder_path}")

# Lista para armazenar os dados reais
real_data_list = []
stream = Stream()
for sac_file in sac_files:
    real_data = read(sac_file)[0]
    real_data.detrend('linear')
    real_data.taper(max_percentage=0.05)
    real_data.resample(sampling_hz)
    stream += real_data  # Add the trace to the stream
    real_data_list.append(real_data)
Z_trace = stream.select(channel='BHZ')
R_trace = stream.select(channel='BHR')
T_trace = stream.select(channel='BHT')
Z_trace = Z_trace[0]
R_trace = R_trace[0]
T_trace = T_trace[0]

output = generate_synthetics(event, stations, seismic_model, tlen, nspc, sampling_hz)
us = output.us  # synthetics. us.shape = (3,nr,tlen)
ts = output.ts  # time points [0, tlen]
output.write(root_path='synthetics/.', format='sac')

# Ajuste nos dados sintéticos
model = TauPyModel(model="cd_model1")
arrivals = model.get_travel_times(source_depth_in_km=depth, distance_in_degree=distance, phase_list=['P', 'S'])
print(f"Depth: {depth} km - Arrivals: {arrivals}")
if arrivals:
    travel_time_p = arrivals[0].time
    travel_time_s = arrivals[1].time
else:
    raise ValueError('Não foi possível calcular o tempo de chegada da onda P.')
ts_adjusted = ts - travel_time_p
tss_adjusted = ts - travel_time_s

# Ajuste para os dados reais
for i in range(len(real_data_list)):
    trace_start_time = real_data_list[i].stats.starttime
    shift_real_p = (time_p - trace_start_time)
    shift_real_s = (time_s - trace_start_time)
    real_data_list[i].times_shifted_p = real_data_list[i].times() - shift_real_p
    real_data_list[i].times_shifted_s = real_data_list[i].times() - shift_real_s

# Limitar as séries temporais para o comprimento dos dados reais
max_time = min(1500, ts[-1])
max_idx = np.searchsorted(ts, max_time)

# Ajustar as formas de onda para o comprimento máximo
u_Z_ELYSE_XB = output['Z', 'ELYSE_XB'][:max_idx]
u_R_ELYSE_XB = output['R', 'ELYSE_XB'][:max_idx]
u_T_ELYSE_XB = output['T', 'ELYSE_XB'][:max_idx]
ts = ts[:max_idx]

# Garantir que os dados reais também estejam cortados corretamente
for i in range(len(real_data_list)):
    real_data_list[i].data = real_data_list[i].data[:max_idx]

# Aplicar filtros
u_Z_ELYSE_XB_filtered = apply_filter(u_Z_ELYSE_XB, sampling_hz)
u_R_ELYSE_XB_filtered = apply_filter(u_R_ELYSE_XB, sampling_hz)
u_T_ELYSE_XB_filtered = apply_filter(u_T_ELYSE_XB, sampling_hz)
for i in range(len(real_data_list)):
    real_data_list[i].data = apply_filter(real_data_list[i].data, sampling_hz)

#synthetic_data = [u_Z_ELYSE_XB, u_R_ELYSE_XB, u_T_ELYSE_XB]
synthetic_data = [u_Z_ELYSE_XB_filtered, u_R_ELYSE_XB_filtered, u_T_ELYSE_XB_filtered]
filtered_synthetic_data = polarization_filter(synthetic_data, sampling_hz)
u_Z_ELYSE_XB_filtered = filtered_synthetic_data[2]
u_R_ELYSE_XB_filtered = filtered_synthetic_data[0]
u_T_ELYSE_XB_filtered = filtered_synthetic_data[1]
#for i in range(len(real_data_list)):
#    real_data_list[i].data = polarization_filter(real_data_list[i].data, sampling_hz)

# Plotando dados sintéticos e reais para cada componente
synthetics = [u_Z_ELYSE_XB_filtered, u_R_ELYSE_XB_filtered, u_T_ELYSE_XB_filtered]
#synthetics = [normalize(output[comp, 'ELYSE_XB']) for comp in ['Z', 'R', 'T']]
#synthetics = [u_Z_ELYSE_XB, u_R_ELYSE_XB, u_T_ELYSE_XB]
components = ['Z', 'R', 'T']

fig, axs = plt.subplots(3, 4, figsize=(28, 12))
for i, (comp, synthetic, real_data) in enumerate(zip(components, synthetics, real_data_list)):
    synthetic_norm = synthetic / np.max(np.abs(synthetic))
    real_data_norm = real_data.data / np.max(np.abs(real_data.data))
    if np.isnan(synthetic_norm).any():
        raise ValueError(f"Os dados sintéticos normalizados contêm valores NaN na componente {comp}.")
    if np.isnan(real_data_norm).any():
        raise ValueError(f"Os dados reais normalizados contêm valores NaN na componente {comp}.")
    
    cross_corr = correlate(synthetic_norm, real_data_norm, mode='full')
    cross_corr /= np.max(cross_corr)  # Normalize cross-correlation
    lags = np.arange(-len(synthetic_norm) + 1, len(synthetic_norm))
    corr_coefficient = np.corrcoef(synthetic_norm, real_data_norm)[0, 1]

    # Plotar dados sintéticos com o tempo ajustado
    axs[i, 0].plot(ts_adjusted[:max_idx], synthetic_norm[:max_idx], label=f'Synthetic {comp}', color='silver', alpha=0.7)
    axs[i, 0].axvline(x=travel_time_p, linestyle='--', color='black', label='P-wave')
    axs[i, 0].axvline(x=travel_time_s, linestyle='--', color='magenta', label='S-wave')
    axs[i, 0].set_xlim([-100, 700])
    axs[i, 0].set_xlabel('Time (s)')
    axs[i, 0].set_ylabel('Normalized Amplitude')
    axs[i, 0].set_title(f'Synthetic Component {comp}')
    axs[i, 0].legend(loc='lower right')
    axs[i, 0].text(0.05, 0.95, f'Correlation: {corr_coefficient:2f}', transform=axs[i, 0].transAxes, fontsize=12, verticalalignment='top')
    
    # Plotar dados reais com o tempo ajustado
    axs[i, 1].plot(real_data.times_shifted_p[:max_idx], real_data_norm[:max_idx], label=f'Real {comp}', color='red', alpha=0.7)
    axs[i, 1].axvline(x=0, linestyle='--', color='black', label='P-wave')
    axs[i, 1].axvline(x=shift_real_s - shift_real_p, linestyle='--', color='magenta', label='S-wave')
    axs[i, 1].set_xlim([-100, 700])
    axs[i, 1].set_xlabel('Time (s)')
    axs[i, 1].set_ylabel('Normalized Amplitude')
    axs[i, 1].set_title(f'Real Component {comp}')
    axs[i, 1].legend(loc='lower right')
    axs[i, 1].text(0.05, 0.95, f'Correlation: {corr_coefficient:2f}', transform=axs[i, 1].transAxes, fontsize=12, verticalalignment='top')
    
    xlim_min, xlim_max = -5, 5
    idx_min = max(0, int((xlim_min - ts_adjusted[0]) / (ts_adjusted[1] - ts_adjusted[0])))
    idx_max = min(len(ts_adjusted), int((xlim_max - ts_adjusted[0]) / (ts_adjusted[1] - ts_adjusted[0])))
    synthetic_norm_limited = synthetic_norm[idx_min:idx_max]
    real_data_norm_limited = real_data_norm[idx_min:idx_max]
    cross_corr_limited = correlate(synthetic_norm_limited, real_data_norm_limited, mode='full')
    cross_corr_limited /= np.max(cross_corr_limited)  # Normalize cross-correlation
    corr_coefficient_limited = np.corrcoef(synthetic_norm_limited, real_data_norm_limited)[0, 1]

    axs[i, 2].plot(ts_adjusted[:max_idx], synthetic_norm[:max_idx], label=f'Synthetic {comp}', alpha=0.7, color='silver')
    axs[i, 2].plot(real_data.times_shifted_p[:max_idx], real_data_norm[:max_idx], label=f'Real {comp}', alpha=0.7, color='red')
    axs[i, 2].axvline(x=0, linestyle='--', color='black', label='P-wave')
    axs[i, 2].axvline(x=shift_real_s - shift_real_p, linestyle='--', color='magenta', label='S-wave')
    axs[i, 2].set_xlim([-10, 10])
    axs[i, 2].set_ylim(-0.1, 0.1)
    axs[i, 2].set_xlabel('Time (s)')
    axs[i, 2].set_ylabel('Normalized Amplitude')
    axs[i, 2].set_title(f'Cross Correlation ({comp} Component)')
    axs[i, 2].legend(loc='lower right')
    axs[i, 2].text(0.05, 0.95, f'Corr: {corr_coefficient_limited:.2f}', transform=axs[i, 2].transAxes)

    xlim_min, xlim_max = 355, 370
    idx_min = max(0, int((xlim_min - ts_adjusted[0]) / (ts_adjusted[1] - ts_adjusted[0])))
    idx_max = min(len(ts_adjusted), int((xlim_max - ts_adjusted[0]) / (ts_adjusted[1] - ts_adjusted[0])))
    synthetic_norm_limited = synthetic_norm[idx_min:idx_max]
    real_data_norm_limited = real_data_norm[idx_min:idx_max]
    cross_corr_limited = correlate(synthetic_norm_limited, real_data_norm_limited, mode='full')
    cross_corr_limited /= np.max(cross_corr_limited)  # Normalize cross-correlation
    corr_coefficient_limited = np.corrcoef(synthetic_norm_limited, real_data_norm_limited)[0, 1]

    axs[i, 3].plot(ts_adjusted[:max_idx], synthetic_norm[:max_idx], label=f'Synthetic {comp}', alpha=0.7, color='silver')
    axs[i, 3].plot(real_data.times_shifted_p[:max_idx], real_data_norm[:max_idx], label=f'Real {comp}', alpha=0.7, color='red')
    axs[i, 3].axvline(x=0, linestyle='--', color='black', label='P-wave')
    axs[i, 3].axvline(x=shift_real_s - shift_real_p, linestyle='--', color='magenta', label='S-wave')
    axs[i, 3].set_xlim([355, 375])  # Limites do eixo x entre 350 e 375
    axs[i, 3].set_ylim(-0.5, 0.5)
    axs[i, 3].set_xlabel('Time (s)')
    axs[i, 3].set_ylabel('Normalized Amplitude')
    axs[i, 3].set_title(f'Detail View ({comp} Component)')
    axs[i, 3].legend(loc='lower right')
    axs[i, 3].text(0.05, 0.95, f'Corr: {corr_coefficient_limited:.2f}', transform=axs[i, 3].transAxes)
plt.subplots_adjust(hspace=0.4, wspace=0.3)
plt.tight_layout()
plt.savefig('figs/output_cross_correlation.png')
```

Scripts de exemplo e notebooks estão disponíveis na pasta `examples/`.


## 🪐 Resultados

O método foi aplicado com sucesso ao evento S0185a, obtendo uma profundidade de ~39 km e mecanismo focal normal. Veja detalhes no artigo (link abaixo).


## 🪐 Publicação

Se usar este código, por favor cite:

> Villanova & Genda, 2025. *AresWave: Estimation of marsquake source parameters by waveform fitting with stochastic optimization*. [Link para o preprint ou DOI]


## 🪐 Contato

Para dúvidas ou colaborações, envie um e-mail para: [villanova@elsi.jp]


## 🪐 Licença

Este projeto está licenciado sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.
