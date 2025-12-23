import numpy as np
import matplotlib.pyplot as plt
from obspy import Stream, read, UTCDateTime
from dsmpy import seismicmodel_Mars
from areswave.synthetics_function import generate_synthetics, apply_filter, calculate_variation, calculate_moment_tensor, normalize, align_by_correlation
from areswave.denoising import polarization_filter
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

def reorder_traces(traces):
    trace_dict = {}
    for tr in traces:
        if tr.stats.channel.endswith("Z"):
            trace_dict["Z"] = tr
        elif tr.stats.channel.endswith("R"):
            trace_dict["R"] = tr
        elif tr.stats.channel.endswith("T"):
            trace_dict["T"] = tr
    return [trace_dict[c] for c in ["Z", "R", "T"]]


def run_grid_search(
    event, stations, seismic_model, tlen, nspc, sampling_hz,
    real_data_list, Z_trace, R_trace, T_trace,
    time_p, time_s, magnitude, distance,
    depth_range, strike_range, dip_range, rake_range,
    frequency_range, frequency_interval,
    output_grid_search_fig_path=None
):

    real_data_list = reorder_traces(real_data_list)

    tested_depths = []
    tested_strikes = []
    tested_dips = []
    tested_rakes = []
    tested_costs = []

    max_shift_samples = int(2.0 * sampling_hz)
    n_samples = len(real_data_list[0].data)

    start_time = real_data_list[0].stats.starttime
    p_idx = int((time_p - start_time) * sampling_hz)
    s_idx = int((time_s - start_time) * sampling_hz)

    real_Z = normalize(real_data_list[0].data[:n_samples])
    real_R = normalize(real_data_list[1].data[:n_samples])
    real_T = normalize(real_data_list[2].data[:n_samples])

    syn_times = np.arange(n_samples) / sampling_hz

    best_cost = np.inf
    best_params = None

    for dpt in depth_range:
        for stke in strike_range:
            for dp in dip_range:
                for rk in rake_range:
                    tested_depths.append(float(dpt))
                    tested_strikes.append(float(stke))
                    tested_dips.append(float(dp))
                    tested_rakes.append(float(rk))

                    if dpt < 10 or dpt > 560:
                        tested_costs.append(1e6)
                        continue

                    try:
                        mts = calculate_moment_tensor(
                            magnitude, stke, dp, rk, dpt, distance,
                            frequency_range=frequency_range, interval=frequency_interval
                        )
                        if mts is None or len(mts) == 0:
                            raise ValueError("calculate_moment_tensor retornou vazio/None.")

                        Mrr = np.mean([m["moment_tensor"].Mrr for m in mts])
                        Mtt = np.mean([m["moment_tensor"].Mtt for m in mts])
                        Mpp = np.mean([m["moment_tensor"].Mpp for m in mts])
                        Mrt = np.mean([m["moment_tensor"].Mrt for m in mts])
                        Mrp = np.mean([m["moment_tensor"].Mrp for m in mts])
                        Mtp = np.mean([m["moment_tensor"].Mtp for m in mts])
                        event.mt = MomentTensor(Mrr, Mrt, Mrp, Mtt, Mtp, Mpp)
                        event.depth = float(dpt)
                    except Exception as e:
                        print(f"Error to calculate the moment tensor (grid): {e}")
                        tested_costs.append(1e6)
                        continue

                    try:
                        output = generate_synthetics(event, stations, seismic_model, tlen, nspc, sampling_hz)
                    except Exception as e:
                        print(f"Error to generate the synthetics (grid): {e}")
                        tested_costs.append(1e6)
                        continue

                    ts = output.ts
                    max_idx = np.searchsorted(ts, tlen)

                    u_Z = output["Z", "ELYSE_XB"][:max_idx]
                    u_R = output["R", "ELYSE_XB"][:max_idx]
                    u_T = output["T", "ELYSE_XB"][:max_idx]

                    if u_Z.size == 0 or u_R.size == 0 or u_T.size == 0:
                        tested_costs.append(1e6)
                        continue

                    u_Z_f = apply_filter(u_Z, sampling_hz)
                    u_R_f = apply_filter(u_R, sampling_hz)
                    u_T_f = apply_filter(u_T, sampling_hz)
                    filtered = polarization_filter([u_Z_f, u_R_f, u_T_f], sampling_hz)

                    syn_Z_raw = normalize(filtered[0][:n_samples])
                    syn_R_raw = normalize(filtered[1][:n_samples])
                    syn_T_raw = normalize(filtered[2][:n_samples])

                    syn_Z = align_by_correlation(real_Z, syn_Z_raw, max_shift_samples)
                    syn_R = align_by_correlation(real_R, syn_R_raw, max_shift_samples)
                    syn_T = align_by_correlation(real_T, syn_T_raw, max_shift_samples)

                    try:
                        cost = calculate_variation(
                            real_Z, syn_Z,
                            real_R, syn_R,
                            real_Z, syn_Z,
                            real_T, syn_T,
                            syn_times, magnitude, sampling_hz,
                            p_idx, s_idx
                        )
                    except Exception as e:
                        print(f"Erro na calculate_variation (grid): {e}")
                        cost = 1e6

                    tested_costs.append(float(cost))

                    if cost < best_cost:
                        best_cost = float(cost)
                        best_params = (float(dpt), float(stke), float(dp), float(rk))
                        print(f"New best: depth={best_params[0]:.2f}, strike={best_params[1]:.2f}, "
                              f"dip={best_params[2]:.2f}, rake={best_params[3]:.2f} | cost={best_cost:.3e}")

    if best_params is None:
        raise RuntimeError("Grid search falhou: nenhuma combinação válida produziu sintéticos/custo.")

    print(f"\nMelhor combinação (grid): depth={best_params[0]}, strike={best_params[1]}, dip={best_params[2]}, rake={best_params[3]}")
    print(f"Misfit mínimo (grid): {best_cost:.4e}")

    if output_grid_search_fig_path:
        try:
            import pandas as pd
            out_csv = os.path.splitext(output_grid_search_fig_path)[0] + ".csv"
            df = pd.DataFrame({
                "Depth (km)": tested_depths,
                "Strike (°)": tested_strikes,
                "Dip (°)": tested_dips,
                "Rake (°)": tested_rakes,
                "Cost": tested_costs
            })
            df.to_csv(out_csv, index=False)

            plt.figure(figsize=(10, 5))
            x = np.arange(len(tested_costs))
            plt.scatter(x, tested_costs, s=10, alpha=0.5)
            plt.xlabel("Evaluation (grid)")
            plt.ylabel("Cost")
            plt.title("Grid search costs")
            plt.tight_layout()
            plt.savefig(output_grid_search_fig_path)
            plt.close()
        except Exception as e:
            print(f"Warning: falha ao salvar CSV/fig da grid search: {e}")

    return best_params, best_cost

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

    stations = [
        Station(name='ELYSE', network='XB', latitude=4.502384, longitude=135.623447),
    ]

    real_data_list, Z_trace, R_trace, T_trace = load_and_process_sac_data(
        sac_folder_path, sampling_hz, time_p, time_s
    )

    for i in range(len(real_data_list)):
        real_data_list[i].data = apply_filter(real_data_list[i].data, sampling_hz)

    output, ts, ts_adjusted, tss_adjusted, travel_time_p, travel_time_s = generate_and_adjust_synthetics(
        event, stations, seismic_model, tlen, nspc, sampling_hz, depth, distance, time_p, time_s
    )

    max_time = min(1500, ts[-1])
    max_idx = np.searchsorted(ts, max_time)

    u_Z_ELYSE_XB = apply_filter(output['Z', 'ELYSE_XB'][:max_idx], sampling_hz)
    u_R_ELYSE_XB = apply_filter(output['R', 'ELYSE_XB'][:max_idx], sampling_hz)
    u_T_ELYSE_XB = apply_filter(output['T', 'ELYSE_XB'][:max_idx], sampling_hz)

    synthetics = [u_Z_ELYSE_XB, u_R_ELYSE_XB, u_T_ELYSE_XB]

    plot_waveforms(
        synthetics, real_data_list, ts_adjusted,
        time_s.timestamp - time_p.timestamp, 0, max_idx,
        travel_time_p, travel_time_s, output_waveform_fig_path
    )

    best_params, min_variation = run_grid_search(
        event, stations, seismic_model, tlen, nspc, sampling_hz,
        real_data_list, Z_trace, R_trace, T_trace,
        time_p, time_s, magnitude, distance,
        depth_range, strike_range, dip_range, rake_range,
        frequency_range, frequency_interval, output_grid_search_fig_path
    )

    return best_params, min_variation