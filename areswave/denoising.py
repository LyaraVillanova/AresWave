import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import obspy
from scipy import signal
from obspy import read
from obspy.core.trace import Trace, Stats
from obspy.core.stream import Stream
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
client = Client("IRIS")
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import AnchoredText
import matplotlib.patches
from matplotlib.pyplot import cm
from matplotlib import cycler
from matplotlib.gridspec import GridSpec

def waveforms(start, end, adjtime):
    st = client.get_waveforms("XB", "ELYSE", "02", "BH*", start-(adjtime/2), end+adjtime, attach_response=True)
    st_rawc = st.copy()
    return st_rawc

def waveformd(start, end, adjtime):
    st = client.get_waveforms("XB", "ELYSE", "02", "BH*", start-(adjtime/2), end+adjtime, attach_response=True)
    st_disp = st.copy()
    st_disp.remove_response(output='DISP')
    return st_disp

def waveform1(start, end, adjtime):
    st = client.get_waveforms("XB", "ELYSE", "02", "BH*", start-(adjtime/2), end+adjtime, attach_response=True)
    st_flt1 = st.copy()
    st_flt1.filter('bandpass', freqmin=0.3, freqmax=0.9, zerophase=True)
    return st_flt1

def waveform2(start, end, adjtime):
    st = client.get_waveforms("XB", "ELYSE", "02", "BH*", start-(adjtime/2), end+adjtime, attach_response=True)
    st_flt2 = st.copy()
    st_flt2.filter('bandpass', freqmin=0.5, freqmax=0.9, zerophase=True)
    return st_flt2

def rotate(c1,c2,a):
    #c1 c2 are the X and Y axes, of a Cartesian coordinate system
    #a is an angle in degrees, positive angle means a clockwise rotation of the coordinate system.
    #o1 o2 are the X and Y axes, of a rotated Cartesian coordinate system
    o1 = np.cos(np.radians(a))*c1 - np.sin(np.radians(a))*c2
    o2 = np.sin(np.radians(a))*c1 + np.cos(np.radians(a))*c2
    return o1, o2

def enz2rtql(st,angles):
    BAz = angles[0]
    Pincd = angles[1]
    Sincd = angles[2]
    
    for trace in st:
        head = trace.stats
        channel = head.channel
        if channel == 'BHE': E = trace.data
        elif channel == 'BHN': N = trace.data
        elif channel == 'BHZ': Z = trace.data
        else:
            print('Trace.channel is not BHU, BHV, or BHW')
            return st

    head.channel = 'BHE'; trE = Trace(data=E, header=head)
    head.channel = 'BHN'; trN = Trace(data=N, header=head)
    head.channel = 'BHZ'; trZ = Trace(data=Z, header=head)
    stENZ = Stream(traces=[trE,trN,trZ])
    
    hhe = stENZ[0].data
    hhn = stENZ[1].data
    hhz = stENZ[2].data
    
    hhT,hhR = rotate(trE,trN,BAz)
    head.channel = 'BHT'; trT = Trace(data=hhT, header=head)
    head.channel = 'BHR'; trR = Trace(data=hhR, header=head)
    
    phhQ,phhL = rotate(trR,trZ,Pincd)
    shhQ,shhL = rotate(trR,trZ,Sincd)
    head.channel = 'BHL'; trL = Trace(data=phhL, header=head)
    head.channel = 'BHQ'; trQ = Trace(data=shhQ, header=head)
    
    stALL = Stream(traces=[trE,trN,trZ,trT,trR,trL,trQ])
    
    return stALL

def compute_covariance_matrix(components):
    n_components = len(components)
    covariance_matrix = np.zeros((n_components, n_components))

    # Garantir que todos os vetores tenham o mesmo comprimento
    min_length = min(len(comp) for comp in components)
    truncated_components = [comp[:min_length] for comp in components]

    for i in range(n_components):
        for j in range(n_components):
            comp_i = truncated_components[i]
            comp_j = truncated_components[j]

            # Verificar se há valores inválidos
            if np.any(np.isnan(comp_i)) or np.any(np.isnan(comp_j)):
                return None

            if len(comp_i) < 2 or len(comp_j) < 2:
                return None  # Necessário pelo menos dois pontos para calcular a covariância

            covariance = np.cov(comp_i, comp_j)[0, 1]
            covariance_matrix[i, j] = covariance

    return covariance_matrix

def calculate_rectilinearity(lambda1, lambda2, n=0.5, J=1):
    return (1 - (lambda2 / lambda1) ** n) ** J

def calculate_direction_function(eigenvector, k=2):
    return eigenvector ** k

def apply_polarization_filter(components, sampling_rate, window_duration=5, n=0.5, J=1, k=2):
    if len(components) != 3:
        raise ValueError("É necessário fornecer exatamente 3 componentes: Z, R e T.")

    # Inicializar componentes filtradas
    Z_filt, R_filt, T_filt = [], [], []

    window_length = int(window_duration * sampling_rate)  # Duração da janela em amostras
    total_samples = len(components[0])  # Assumimos que todas as componentes têm o mesmo comprimento

    for t in range(0, total_samples, window_length):
        end_sample = min(t + window_length, total_samples)

        # Extrair os dados da janela
        window_data = [comp[t:end_sample] for comp in components]

        # Verificar se há dados suficientes na janela
        if any(len(w) < window_length and end_sample != total_samples for w in window_data):
            print(f"Janela em t={t} não tem dados suficientes.")
            continue

        # Calcular a matriz de covariância
        covariance_matrix = compute_covariance_matrix(window_data)
        if covariance_matrix is None:
            print(f"Matriz de covariância inválida em t={t}.")
            continue

        # Calcular autovalores e autovetores
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Ordenar autovalores e autovetores de forma decrescente
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        lambda1, lambda2 = eigenvalues[:2]

        # Garantir que os autovalores sejam válidos
        if lambda1 <= 0 or lambda2 <= 0:
            print(f"Autovalores não positivos em t={t}.")
            continue

        # Calcular retangularidade e função de direção
        rectilinearity = calculate_rectilinearity(lambda1, lambda2, n, J)
        direction_functions = [calculate_direction_function(e, k) for e in eigenvectors[:, 0]]

        # Aplicar o filtro aos dados da janela
        filtered_window = [
            w * rectilinearity * direction_functions[i] for i, w in enumerate(window_data)
        ]

        # Adicionar os dados filtrados às respectivas componentes
        Z_filt.extend(filtered_window[0])
        R_filt.extend(filtered_window[1])
        T_filt.extend(filtered_window[2])

    return np.array(Z_filt), np.array(R_filt), np.array(T_filt)

def polarization_filter(components, sampling_rate, window_duration=5, n=0.5, J=1, k=2):
    return apply_polarization_filter(components, sampling_rate, window_duration, n, J, k)

def smooth(x, window_len=11, window='hanning'):
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]
    if window == 'flat':  # média móvel
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[int(window_len/2-1):-int(window_len/2)]