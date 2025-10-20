import numpy as np
import dsmpy.dsm_Mars
from dsmpy.event_Mars import MomentTensor
from scipy.signal import butter, filtfilt, correlate

def generate_synthetics(event, stations, seismic_model, tlen, nspc, sampling_hz):
    input = dsmpy.dsm_Mars.PyDSMInput.input_from_arrays(event, stations, seismic_model, tlen, nspc, sampling_hz)
    output = dsmpy.dsm_Mars.compute(input)
    output.to_time_domain()  # perform inverse FFT
    #output.filter(freq=0.04)  # apply a 25 seconds low-pass filter
    us = output.us  # synthetics. us.shape = (3,nr,tlen)
    ts = output.ts  # time points [0, tlen]
    return output

def normalize(data):
    return data / (np.max(np.abs(data)) + 1e-8)

def apply_filter(data, sampling_rate, filter_type='bandpass'):
    nyquist = 0.5 * sampling_rate
    low = 0.5 / nyquist
    high = 0.9 / nyquist
    b, a = butter(4, [low, high], btype=filter_type, analog=False)
    return filtfilt(b, a, data)

def apply_filter3(data, sampling_rate, filter_type='bandpass'):
    nyquist = 0.5 * sampling_rate
    low = 0.3 / nyquist
    high = 0.9 / nyquist
    b, a = butter(4, [low, high], btype=filter_type, analog=False)
    return filtfilt(b, a, data)

def apply_filter_earth(data, sampling_rate, filter_type='bandpass'):
    nyquist = 0.5 * sampling_rate
    low = 0.7 / nyquist
    high = 3.0 / nyquist
    b, a = butter(4, [low, high], btype=filter_type, analog=False)
    return filtfilt(b, a, data)

def calculate_variation(
    real_PZ, syn_PZ,
    real_PR, syn_PR,
    real_SZ, syn_SZ,
    real_ST, syn_ST,
    time_array,
    magnitude, sampling_rate,
    p_idx, s_idx):

    def extract_window(data, center_idx, before, after, sampling_rate=20):
        start = int(max(0, center_idx - before * sampling_rate))
        end = int(min(len(data), center_idx + after * sampling_rate))
        return data[start:end]

    def window_misfit(real, syn):
        if len(real) == 0 or len(syn) == 0:
            return 0.0
        min_len = min(len(real), len(syn))
        return np.mean((real[:min_len] - syn[:min_len]) ** 2)

    r_PZ_win = extract_window(real_PZ, p_idx, 5, 5)
    s_PZ_win = extract_window(syn_PZ, p_idx, 5, 5)
    r_PR_win = extract_window(real_PR, p_idx, 5, 5)
    s_PR_win = extract_window(syn_PR, p_idx, 5, 5)

    r_SZ_win = extract_window(real_SZ, s_idx, 5, 10)
    s_SZ_win = extract_window(syn_SZ, s_idx, 5, 10)
    r_ST_win = extract_window(real_ST, s_idx, 5, 10)
    s_ST_win = extract_window(syn_ST, s_idx, 5, 10)

    var_P = window_misfit(r_PZ_win, s_PZ_win) + window_misfit(r_PR_win, s_PR_win)
    var_S = window_misfit(r_SZ_win, s_SZ_win) + window_misfit(r_ST_win, s_ST_win)
    total = float(var_P + var_S)

    print(f"Variation PZ: {window_misfit(r_PZ_win, s_PZ_win):.5f}")
    print(f"Variation PR: {window_misfit(r_PR_win, s_PR_win):.5f}")
    print(f"Variation SZ: {window_misfit(r_SZ_win, s_SZ_win):.5f}")
    print(f"Variation ST: {window_misfit(r_ST_win, s_ST_win):.5f}")
    return total

def calculate_moment_tensor(magnitude, strike, dip, rake, depth, distance, frequency_range=(0.1, 1.0), interval=0.1):
    # Cálculo do momento sísmico
    Mo = 10 ** ((magnitude + 10.7) * 3 / 2)
    strike_rad, dip_rad, rake_rad = np.radians([strike, dip, rake])

    Mrr = Mo * np.sin(2 * dip_rad) * np.sin(rake_rad)
    Mrt = -Mo * (np.cos(dip_rad) * np.cos(rake_rad) * np.cos(strike_rad) + np.cos(2 * dip_rad) * np.sin(rake_rad) * np.sin(strike_rad))
    Mrp = -Mo * (np.cos(dip_rad) * np.cos(rake_rad) * np.sin(strike_rad) - np.cos(2 * dip_rad) * np.sin(rake_rad) * np.cos(strike_rad))
    Mtt = Mo * (np.sin(dip_rad) * np.cos(rake_rad) * np.sin(2 * strike_rad) + np.sin(2 * dip_rad) * np.sin(rake_rad) * np.sin(strike_rad) ** 2)
    Mtp = Mo * (np.sin(dip_rad) * np.cos(rake_rad) * np.cos(2 * strike_rad) + 0.5 * np.sin(2 * dip_rad) * np.sin(rake_rad) * np.sin(2 * strike_rad) ** 2)
    Mpp = Mo * (np.sin(dip_rad) * np.cos(rake_rad) * np.sin(2 * strike_rad) - np.sin(2 * dip_rad) * np.sin(rake_rad) * np.cos(strike_rad) ** 2)

    if 0 <= depth <= 24:
        vp = 2.3
        vs = 1.55
    elif 24.01 <= depth <= 49.5:
        vp = 2.6
        vs = 2.54
    elif depth >= 49.5:
        vp = 6.1
        vs = 3.59
    else:
        raise ValueError("Profundidade fora do intervalo esperado.")

    Qp = 400
    Qs = 150
    frequencies = np.arange(frequency_range[0], frequency_range[1] + interval, interval)
    moment_tensors = []
    for frequency in frequencies:
        attenuation_p = np.exp(-distance / (Qp * vp * frequency))
        attenuation_s = np.exp(-distance / (Qs * vs * frequency))

        Mrr_att = Mrr * attenuation_p
        Mtt_att = Mtt * attenuation_p
        Mpp_att = Mpp * attenuation_p
        Mrt_att = Mrt * attenuation_s
        Mrp_att = Mrp * attenuation_s
        Mtp_att = Mtp * attenuation_s

        # Create DSMpy MomentTensor; DSMpy may expect units of 1e25 dyne-cm (warning is printed elsewhere)
        mt_obj = MomentTensor(Mrr_att, Mrt_att, Mrp_att, Mtt_att, Mtp_att, Mpp_att)

        moment_tensors.append({
            'frequency': frequency,
            'moment_tensor': mt_obj
        })
    return moment_tensors

def align_by_correlation(real, synthetic, max_shift):
    corr = correlate(real, synthetic, mode='full')
    lag = np.argmax(corr) - len(synthetic) + 1
    lag = np.clip(lag, -max_shift, max_shift)
    if lag > 0:
        aligned = np.pad(synthetic, (lag, 0), mode='constant')[:len(synthetic)]
    elif lag < 0:
        aligned = np.pad(synthetic, (0, -lag), mode='constant')[-lag:]
        aligned = np.pad(aligned, (0, len(synthetic) - len(aligned)), mode='constant')
    else:
        aligned = synthetic
    return aligned