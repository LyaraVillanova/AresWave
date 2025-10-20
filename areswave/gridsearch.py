import numpy as np
import matplotlib.pyplot as plt
from obspy import Stream, read, UTCDateTime
from dsmpy import seismicmodel_Mars
from areswave.synthetics_function import generate_synthetics, apply_filter, calculate_variation, calculate_moment_tensor
from dsmpy.event_Mars import Event, MomentTensor
from dsmpy.station_Mars import Station
from scipy.signal import correlate
import glob
import os

def load_and_process_sac_data(sac_folder_path, sampling_hz, time_p, time_s):
    sac_files = glob.glob(os.path.join(sac_folder_path, '*.sac'))
    if not sac_files:
        raise FileNotFoundError(f"Nenhum arquivo .sac encontrado na pasta: {sac_folder_path}")
    real_data_list = []
    stream = Stream()
    for sac_file in sac_files:
        real_data = read(sac_file)[0]
        real_data.detrend('linear')
        real_data.taper(max_percentage=0.05)
        real_data.resample(sampling_hz)
        stream += real_data
        real_data_list.append(real_data)
    Z_trace = stream.select(channel='BHZ')[0]
    R_trace = stream.select(channel='BHR')[0]
    T_trace = stream.select(channel='BHT')[0]

    for trace in [Z_trace, R_trace, T_trace]:
        trace.data = apply_filter(trace.data, sampling_hz)

    for i in range(len(real_data_list)):
        trace_start_time = real_data_list[i].stats.starttime
        shift_real_p = (time_p - trace_start_time)
        shift_real_s = (time_s - trace_start_time)
        real_data_list[i].times_shifted_p = real_data_list[i].times() - shift_real_p
        real_data_list[i].times_shifted_s = real_data_list[i].times() - shift_real_s
    return real_data_list, Z_trace, R_trace, T_trace

def run_grid_search(model, event_lat, event_lon, origin_time, depth_values, strike_values, dip_values, rake_values,
                    Z_trace, R_trace, T_trace, station_lat, station_lon, station_name, filter_freqs, sampling_hz):

    ts = Z_trace.times()
    max_time = min(1500, ts[-1])
    max_idx = np.searchsorted(ts, max_time)
    ts = ts[:max_idx]  # corta o vetor de tempo

    # Real data cortado
    real_PZ_full = Z_trace.data[:max_idx]
    real_PR_full = R_trace.data[:max_idx]
    real_SZ_full = Z_trace.data[:max_idx]
    real_ST_full = T_trace.data[:max_idx]

    misfit_values = []
    tested_params = []

    for depth in depth_values:
        for strike in strike_values:
            for dip in dip_values:
                for rake in rake_values:

                    mt = compute_moment_tensor(strike, dip, rake)
                    origin = obspy.UTCDateTime(origin_time)

                    output = generate_synthetics(model=model,
                                                 origin_time=origin,
                                                 event_lat=event_lat,
                                                 event_lon=event_lon,
                                                 event_depth_km=depth,
                                                 mt=mt,
                                                 station_lat=station_lat,
                                                 station_lon=station_lon,
                                                 station_name=station_name,
                                                 components=["Z", "R", "T"],
                                                 duration=1500,
                                                 sampling_hz=sampling_hz)

                    print("compute PSV")
                    output.compute("PSV")
                    print("compute SH")
                    output.compute("SH")

                    print("generate_and_adjust_synthetics - ts length:", len(ts))
                    synthetic_PZ = apply_filter(output['Z', station_name], sampling_hz)[:max_idx]
                    synthetic_PR = apply_filter(output['R', station_name], sampling_hz)[:max_idx]
                    synthetic_SZ = apply_filter(output['Z', station_name], sampling_hz)[:max_idx]
                    synthetic_ST = apply_filter(output['T', station_name], sampling_hz)[:max_idx]

                    print("generate_and_adjust_synthetics - synthetic Z length:", len(synthetic_PZ))
                    print("generate_and_adjust_synthetics - synthetic R length:", len(synthetic_PR))
                    print("generate_and_adjust_synthetics - synthetic T length:", len(synthetic_ST))

                    # Garante que todos os arrays têm o mesmo tamanho
                    assert len(real_PZ_full) == len(synthetic_PZ)
                    assert len(real_PR_full) == len(synthetic_PR)
                    assert len(real_SZ_full) == len(synthetic_SZ)
                    assert len(real_ST_full) == len(synthetic_ST)

                    # Calcula misfit (exemplo com L2 norm)
                    misfit_PZ = np.linalg.norm(real_PZ_full - synthetic_PZ)
                    misfit_PR = np.linalg.norm(real_PR_full - synthetic_PR)
                    misfit_SZ = np.linalg.norm(real_SZ_full - synthetic_SZ)
                    misfit_ST = np.linalg.norm(real_ST_full - synthetic_ST)

                    total_misfit = misfit_PZ + misfit_PR + misfit_SZ + misfit_ST

                    misfit_values.append(total_misfit)
                    tested_params.append((depth, strike, dip, rake))

    best_idx = np.argmin(misfit_values)
    best_params = tested_params[best_idx]
    best_misfit = misfit_values[best_idx]

    print(f"Melhor combinação: depth={best_params[0]}, strike={best_params[1]}, dip={best_params[2]}, rake={best_params[3]}")
    print(f"Misfit mínimo: {best_misfit:.4f}")

    return best_params, best_misfit, misfit_values, tested_params


def generate_and_adjust_synthetics(event, stations, seismic_model, tlen, nspc, sampling_hz, depth, distance, time_p, time_s):
    output = generate_synthetics(event, stations, seismic_model, tlen, nspc, sampling_hz)
    ts = output.ts

    print(f"generate_and_adjust_synthetics - ts length: {len(ts)}")
    print(f"generate_and_adjust_synthetics - synthetic Z length: {len(output['Z', 'ELYSE_XB'])}")
    print(f"generate_and_adjust_synthetics - synthetic R length: {len(output['R', 'ELYSE_XB'])}")
    print(f"generate_and_adjust_synthetics - synthetic T length: {len(output['T', 'ELYSE_XB'])}")
    
    shift_real_p = (time_p - time_p)
    shift_real_s = (time_s - time_p)
    ts_adjusted = ts - shift_real_p
    tss_adjusted = ts - shift_real_s
    travel_time_p = 0
    travel_time_s = shift_real_s
    return output, ts, ts_adjusted, tss_adjusted, travel_time_p, travel_time_s

def plot_waveforms(synthetics, real_data_list, ts_adjusted, shift_real_s, shift_real_p, max_idx, travel_time_p, travel_time_s, fig_path):
    components = ['Z', 'R', 'T']
    fig, axs = plt.subplots(3, 3, figsize=(24, 12))
    for i, (comp, synthetic, real_data) in enumerate(zip(components, synthetics, real_data_list)):
        synthetic_norm = synthetic / np.max(np.abs(synthetic))
        real_data_norm = real_data.data / np.max(np.abs(real_data.data))
        if np.isnan(synthetic_norm).any():
            raise ValueError(f"Os dados sintéticos normalizados contêm valores NaN na componente {comp}.")
        if np.isnan(real_data_norm).any():
            raise ValueError(f"Os dados reais normalizados contêm valores NaN na componente {comp}.")
        
        axs[i, 0].plot(ts_adjusted[:max_idx], synthetic_norm[:max_idx], label=f'Synthetic {comp}', color='silver', alpha=0.7)
        axs[i, 0].axvline(x=0, linestyle='--', color='black', label='P-wave')
        axs[i, 0].axvline(x=shift_real_s - shift_real_p, linestyle='--', color='magenta', label='S-wave')
        axs[i, 0].set_xlim([-100, 700])
        axs[i, 0].set_xlabel('Time (s)')
        axs[i, 0].set_ylabel('Normalized Amplitude')
        axs[i, 0].set_title(f'Synthetic Component {comp}')
        axs[i, 0].legend(loc='lower right')
        
        axs[i, 1].plot(real_data.times_shifted_p[:max_idx], real_data_norm[:max_idx], label=f'Real {comp}', color='red', alpha=0.7)
        axs[i, 1].axvline(x=0, linestyle='--', color='black', label='P-wave')
        axs[i, 1].axvline(x=shift_real_s - shift_real_p, linestyle='--', color='magenta', label='S-wave')
        axs[i, 1].set_xlim([-100, 700])
        axs[i, 1].set_xlabel('Time (s)')
        axs[i, 1].set_ylabel('Normalized Amplitude')
        axs[i, 1].set_title(f'Real Component {comp}')
        axs[i, 1].legend(loc='lower right')
        
        cross_corr = correlate(synthetic_norm, real_data_norm, mode='full')
        cross_corr /= np.max(cross_corr)  # Normalize cross-correlation
        lags = np.arange(-len(synthetic_norm) + 1, len(synthetic_norm))
        max_corr_idx = cross_corr.argmax()
        max_corr = cross_corr[max_corr_idx]
        corr_coefficient = np.corrcoef(synthetic_norm, real_data_norm)[0, 1]
        
        axs[i, 2].plot(ts_adjusted[:max_idx], synthetic_norm[:max_idx], label=f'Synthetic {comp}', alpha=0.7, color='silver')
        axs[i, 2].plot(real_data.times_shifted_p[:max_idx], real_data_norm[:max_idx], label=f'Real {comp}', alpha=0.7, color='red')
        axs[i, 2].axvline(x=0, linestyle='--', color='black', label='P-wave')
        axs[i, 2].axvline(x=shift_real_s - shift_real_p, linestyle='--', color='magenta', label='S-wave')
        axs[i, 2].axvline(x=travel_time_p - shift_real_p, linestyle='--', color='black', label='P-wave')
        axs[i, 2].axvline(x=travel_time_s - (shift_real_s - shift_real_p), linestyle='--', color='magenta', label='S-wave')
        axs[i, 2].set_xlim([-75, 75])
        axs[i, 2].set_xlabel('Time (s)')
        axs[i, 2].set_ylabel('Normalized Amplitude')
        axs[i, 2].set_title(f'Cross Correlation ({comp} Component)')
        axs[i, 2].legend(loc='lower right')
        axs[i, 2].text(0.05, 0.95, f'Correlation: {corr_coefficient:10f}', transform=axs[i, 2].transAxes, fontsize=12, verticalalignment='top')
    plt.tight_layout()
    plt.savefig(fig_path)


def main_analysis(
    event_id, latitude, longitude, distance, baz, magnitude, depth, time_p, time_s,
    centroid_time, Mrr, Mrt, Mrp, Mtt, Mtp, Mpp,
    sac_folder_path, tlen, nspc, sampling_hz,
    model_directory, seismic_model,
    depth_range, strike_range, dip_range, rake_range,
    frequency_range, frequency_interval,
    output_waveform_fig_path, output_grid_search_fig_path
):
    # Monta o evento com tensor de momento
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

    # Define estação única (ELYSE)
    stations = [
        Station(name='ELYSE', network='XB', latitude=4.502384, longitude=135.623447),
    ]

    # Carrega os dados reais (Z, R, T)
    real_data_list, Z_trace, R_trace, T_trace = load_and_process_sac_data(
        sac_folder_path, sampling_hz, time_p, time_s
    )

    # Filtra dados reais
    for i in range(len(real_data_list)):
        real_data_list[i].data = apply_filter(real_data_list[i].data, sampling_hz)

    # Gera sintéticos para plot (não faz parte da gridsearch em si)
    output, ts, ts_adjusted, tss_adjusted, travel_time_p, travel_time_s = generate_and_adjust_synthetics(
        event, stations, seismic_model, tlen, nspc, sampling_hz, depth, distance, time_p, time_s
    )

    max_time = min(1500, ts[-1])
    max_idx = np.searchsorted(ts, max_time)

    u_Z_ELYSE_XB = apply_filter(output['Z', 'ELYSE_XB'][:max_idx], sampling_hz)
    u_R_ELYSE_XB = apply_filter(output['R', 'ELYSE_XB'][:max_idx], sampling_hz)
    u_T_ELYSE_XB = apply_filter(output['T', 'ELYSE_XB'][:max_idx], sampling_hz)

    synthetics = [u_Z_ELYSE_XB, u_R_ELYSE_XB, u_T_ELYSE_XB]

    # Plota formas de onda reais e sintéticas
    plot_waveforms(
        synthetics, real_data_list, ts_adjusted,
        time_s.timestamp - time_p.timestamp, 0, max_idx,
        travel_time_p, travel_time_s, output_waveform_fig_path
    )

    # Executa a grid search com a nova função compatível
    best_params, min_variation = run_grid_search(
        event, stations, seismic_model, tlen, nspc, sampling_hz,
        real_data_list, Z_trace, R_trace, T_trace,
        time_p, time_s, magnitude, distance,
        depth_range, strike_range, dip_range, rake_range,
        frequency_range, frequency_interval, output_grid_search_fig_path
    )

    return best_params, min_variation
