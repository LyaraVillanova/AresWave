import numpy as np
import os
import time
import corner
import pandas as pd
import pyswarms as ps
import matplotlib.pyplot as plt
from obspy import Trace
from obspy.imaging.beachball import beachball
from areswave.synthetics_function import generate_synthetics, calculate_moment_tensor, calculate_variation, apply_filter_earth, normalize, align_by_correlation
from areswave.denoising import polarization_filter
from dsmpy.event import Event, MomentTensor
from scipy.signal import correlate


def reorder_traces(traces):
    trace_dict = {}
    for tr in traces:
        if tr.stats.channel.endswith("Z"):
            trace_dict["Z"] = tr
        elif tr.stats.channel.endswith("R") or tr.stats.channel.endswith("R"):
            trace_dict["R"] = tr   # aceita R ou N como radial
        elif tr.stats.channel.endswith("T") or tr.stats.channel.endswith("T"):
            trace_dict["T"] = tr   # aceita T ou E como transversal
    return [trace_dict[c] for c in ["Z", "R", "T"] if c in trace_dict]

def create_cost_function(event, stations, seismic_model, tlen, nspc, sampling_hz, real_data_list, magnitude, distance, time_p, time_s):
    real_data_list = reorder_traces(real_data_list)
    tested_depths = []
    tested_strikes = []
    tested_dips = []
    tested_rakes = []
    tested_costs = []

    def cost_func(x):
        costs = []
        max_shift_samples = int(2.0 * sampling_hz)

        for i, (dpt, stke, dp, rk) in enumerate(x):
            t0 = time.time()
            print(f"\nParticle {i+1}/{len(x)}")
            print(f"Testing parameters: depth={dpt:.2f}, strike={stke:.2f}, dip={dp:.2f}, rake={rk:.2f}")

            tested_depths.append(dpt)
            tested_strikes.append(stke)
            tested_dips.append(dp)
            tested_rakes.append(rk)

            if dpt < 10 or dpt > 560:
                costs.append(1e6)
                tested_costs.append(1e6)
                continue

            try:
                mt0 = calculate_moment_tensor(magnitude, stke, dp, rk, dpt, distance,
                                              frequency_range=(0.1, 1.0), interval=0.1)[0]["moment_tensor"]
            except Exception as e:
                print(f"Error to calculate the moment tensor: {e}")
                costs.append(1e6)
                tested_costs.append(1e6)
                continue

            mts = calculate_moment_tensor(magnitude, stke, dp, rk, dpt, distance,
                              frequency_range=(0.1, 1.0), interval=0.1)
            Mrr = np.mean([m["moment_tensor"].Mrr for m in mts])
            Mtt = np.mean([m["moment_tensor"].Mtt for m in mts])
            Mpp = np.mean([m["moment_tensor"].Mpp for m in mts])
            Mrt = np.mean([m["moment_tensor"].Mrt for m in mts])
            Mrp = np.mean([m["moment_tensor"].Mrp for m in mts])
            Mtp = np.mean([m["moment_tensor"].Mtp for m in mts])
            mt0 = MomentTensor(Mrr, Mrt, Mrp, Mtt, Mtp, Mpp)
            event.mt = mt0
            event.depth = dpt

            try:
                output = generate_synthetics(event, stations, seismic_model, tlen, nspc, sampling_hz)
            except Exception as e:
                print(f"Error to generate the synthetics: {e}")
                costs.append(1e6)
                tested_costs.append(1e6)
                continue

            ts = output.ts
            max_idx = np.searchsorted(ts, tlen)
            u_Z = output["Z", "AIO_IV"][:max_idx]
            u_R = output["R", "AIO_IV"][:max_idx]
            u_T = output["T", "AIO_IV"][:max_idx]
            if u_Z.size == 0 or u_R.size == 0 or u_T.size == 0:
                costs.append(1e6)
                tested_costs.append(1e6)
                continue

            u_Z_f = apply_filter_earth(u_Z, sampling_hz)
            u_R_f = apply_filter_earth(u_R, sampling_hz)
            u_T_f = apply_filter_earth(u_T, sampling_hz)
            #filtered = polarization_filter([u_Z_f, u_R_f, u_T_f], sampling_hz)

            #n_samples = len(real_data_list[0].data)
            #syn_Z_raw = normalize(filtered[0][:n_samples])
            #syn_R_raw = normalize(filtered[1][:n_samples])
            #syn_T_raw = normalize(filtered[2][:n_samples])

            n_samples = len(real_data_list[0].data)
            syn_Z_raw = normalize(u_Z_f[:n_samples])
            syn_R_raw = normalize(u_R_f[:n_samples])
            syn_T_raw = normalize(u_T_f[:n_samples])            

            real_Z = normalize(real_data_list[0].data[:n_samples])
            real_R = normalize(real_data_list[1].data[:n_samples])
            real_T = normalize(real_data_list[2].data[:n_samples])

            syn_Z = align_by_correlation(real_Z, syn_Z_raw, max_shift_samples)
            syn_R = align_by_correlation(real_R, syn_R_raw, max_shift_samples)
            syn_T = align_by_correlation(real_T, syn_T_raw, max_shift_samples)

            syn_times = np.arange(n_samples) / sampling_hz

            try:
                start_time = real_data_list[0].stats.starttime
                p_idx = int((time_p - start_time) * sampling_hz)
                s_idx = int((time_s - start_time) * sampling_hz)

                var = calculate_variation(
                    real_Z, syn_Z,
                    real_R, syn_R,
                    real_Z, syn_Z,
                    real_T, syn_T,
                    syn_times, magnitude, sampling_hz,
                    p_idx, s_idx
                )
            except Exception as e:
                print(f"Erro na calculate_variation: {e}")
                var = 1e6

            costs.append(var)
            tested_costs.append(var)
            iter_time = time.time() - t0
            print(f"Iteration {i+1}/{len(x)}: Time={iter_time:.2f}s, Cost={var:.3e}")

        return np.ravel(costs)

    return cost_func, (tested_depths, tested_strikes, tested_dips, tested_rakes, tested_costs)


def run_pso_with_restarts(cost_func, bounds, n_particles, n_iterations, restart_interval, options):
    optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=4, options=options, bounds=bounds)
    print("Starting PSO with reinitiation in each", restart_interval, "iterations...")
    t0_total = time.time()
    global_best_cost = np.inf
    global_best_position = None

    for i in range(0, n_iterations, restart_interval):
        remaining_iters = min(restart_interval, n_iterations - i)
        print(f"\n--- Iterations {i+1} to {i+remaining_iters} ---")

        if i == 0:
            best_cost, best_pos = optimizer.optimize(cost_func, iters=remaining_iters)
            global_best_cost = best_cost
            global_best_position = best_pos
        else:
            spread = np.array([2.0, 20.0, 5.0, 30.0])
            new_particles = global_best_position + np.random.uniform(-spread, spread, size=(n_particles, 4))
            new_particles = np.clip(new_particles, bounds[0], bounds[1])

            optimizer.swarm.position = new_particles
            optimizer.swarm.velocity = np.random.uniform(-1, 1, size=(n_particles, 4))
            optimizer.swarm.best_pos = new_particles.copy()
            optimizer.swarm.pbest_cost = np.full(n_particles, np.inf)

            best_cost, best_pos = optimizer.optimize(cost_func, iters=remaining_iters, verbose=False)
            if best_cost < global_best_cost:
                global_best_cost = best_cost
                global_best_position = best_pos

    print(f"\nTotal PSO time: {(time.time() - t0_total)/60:.2f} min")
    print(f"Better global solution: {global_best_position} with cost {global_best_cost:.2e}")
    return global_best_position, global_best_cost


def plot_results(best_pos, event, stations, seismic_model, tlen, nspc, sampling_hz,
                 real_data_list, time_p, time_s, magnitude, distance, tested_params,
                 fig_dir, event_id, n_particles, restart_interval):
    real_data_list = reorder_traces(real_data_list)

    tested_depths, tested_strikes, tested_dips, tested_rakes, tested_costs = tested_params
    dpt, stke, dp, rk = best_pos
    print("\nCalculating the moment tensor to the best solution...")
    try:
        mts = calculate_moment_tensor(magnitude, stke, dp, rk, dpt, distance,
                                      frequency_range=(0.1, 1.0), interval=0.1)

        if mts is None or len(mts) == 0:
            raise ValueError("Invalid result of calculate_moment_tensor (empty or None).")

        Mrr = np.mean([m["moment_tensor"].Mrr for m in mts])
        Mtt = np.mean([m["moment_tensor"].Mtt for m in mts])
        Mpp = np.mean([m["moment_tensor"].Mpp for m in mts])
        Mrt = np.mean([m["moment_tensor"].Mrt for m in mts])
        Mrp = np.mean([m["moment_tensor"].Mrp for m in mts])
        Mtp = np.mean([m["moment_tensor"].Mtp for m in mts])
        mt = MomentTensor(Mrr, Mrt, Mrp, Mtt, Mtp, Mpp)
        event.mt = mt

        print(f"Mrr: {event.mt.Mrr:.3e}")
        print(f"Mtt: {event.mt.Mtt:.3e}")
        print(f"Mpp: {event.mt.Mpp:.3e}")
        print(f"Mrt: {event.mt.Mrt:.3e}")
        print(f"Mrp: {event.mt.Mrp:.3e}")
        print(f"Mtp: {event.mt.Mtp:.3e}")

        mt_vec = [event.mt.Mrr, event.mt.Mtt, event.mt.Mpp,
                  event.mt.Mrt, event.mt.Mrp, event.mt.Mtp]

        max_abs = max(abs(x) for x in mt_vec)
        if max_abs > 0:
            mt_vec = [x / max_abs for x in mt_vec]

        fig = beachball(mt_vec, size=200, linewidth=1, facecolor="b")
        fig.savefig(os.path.join(fig_dir, f"beachball_{event_id}.png"))
        plt.close(fig)

    except Exception as e:
        print(f"Error to calculate the moment tensor or generate the beachball: {e}")

    print("Generating final synthetics...")
    output = generate_synthetics(event, stations, seismic_model, tlen, nspc, sampling_hz)
    ts = output.ts
    max_idx = np.searchsorted(ts, tlen)
    u_Z = output["Z", "AIO_IV"][:max_idx]
    u_R = output["R", "AIO_IV"][:max_idx]
    u_T = output["T", "AIO_IV"][:max_idx]
    #filtered = polarization_filter([u_Z, u_R, u_T], sampling_hz)
    filtered = [u_Z, u_R, u_T]

    n_samples = len(real_data_list[0].data)
    max_shift_samples = int(2.0 * sampling_hz)
    final_traces = []

    for i, arr in enumerate(filtered):
        syn_raw = arr[:n_samples]
        real = normalize(real_data_list[i].data[:n_samples])
        corr = correlate(real, normalize(syn_raw), mode="full")
        lag = np.argmax(corr) - len(syn_raw) + 1
        lag = np.clip(lag, -max_shift_samples, max_shift_samples)
        aligned = align_by_correlation(real, normalize(syn_raw), max_shift_samples)
        syn = normalize(aligned)

        tr = Trace(data=np.asarray(syn))
        tr.stats.starttime = real_data_list[i].stats.starttime + lag / sampling_hz
        tr.stats.sampling_rate = sampling_hz
        final_traces.append(tr)

    synthetics_save_dir = os.path.join(fig_dir, f"final_synthetics_{event_id}")
    os.makedirs(synthetics_save_dir, exist_ok=True)
    component_labels = ["Z", "R", "T"]

    station = real_data_list[0].stats.station
    network = real_data_list[0].stats.network
    location = real_data_list[0].stats.location or ""

    for i, tr in enumerate(final_traces):
        tr.stats.station = station
        tr.stats.network = network
        tr.stats.channel = f"BH{component_labels[i]}"
        tr.stats.location = location
        tr.stats.npts = len(tr.data)
        sac_path = os.path.join(synthetics_save_dir, f"{event_id}.{station}.{component_labels[i]}.sac")
        tr.write(sac_path, format="SAC")

    # Plots
    fig, axs = plt.subplots(3, 2, figsize=(14, 8))
    axs = np.atleast_2d(axs)

    t_p = (time_p - real_data_list[0].stats.starttime - 10)
    t_s = (time_s - real_data_list[0].stats.starttime - 10)
    window_size = 4

    for i, comp in enumerate(("Z", "R", "T")):
        real_trace = real_data_list[i]
        syn_trace = final_traces[i]
        min_len = min(len(real_trace.data), len(syn_trace.data))
        real_data = real_trace.data[:min_len]
        syn_data = syn_trace.data[:min_len]
        times = np.arange(min_len) / sampling_hz - 10

        idx_p_min = max(0, int((t_p - window_size / 2 - times[0]) / (times[1] - times[0])))
        idx_p_max = min(len(times), int((t_p + window_size / 2 - times[0]) / (times[1] - times[0])))
        corr_p = np.corrcoef(real_data[idx_p_min:idx_p_max], syn_data[idx_p_min:idx_p_max])[0, 1]

        axs[i, 0].plot(times, real_data, label="Real", color="red", lw=1)
        axs[i, 0].plot(times, syn_data, label="Synthetics", color="blue", lw=1, linestyle="-")
        axs[i, 0].axvline(t_p, color="black", linestyle=":", lw=1.5, label="P-wave")
        axs[i, 0].axvspan(t_p - window_size / 2, t_p + window_size / 2, color="lightgray", alpha=0.2)
        axs[i, 0].set_xlim([-4, 4])
        axs[i, 0].set_ylim([-1, 1])
        axs[i, 0].set_title(f"{comp} Component - P wave")
        axs[i, 0].set_xlabel("Time (s)")
        axs[i, 0].set_ylabel("Amplitude")
        axs[i, 0].legend()
        axs[i, 0].text(0.05, 0.95, f"CC (janela): {corr_p:.3f}", transform=axs[i, 0].transAxes, fontsize=10, verticalalignment="top")

        idx_s_min = max(0, int((t_s - window_size / 2 - times[0]) / (times[1] - times[0])))
        idx_s_max = min(len(times), int((t_s + window_size - times[0]) / (times[1] - times[0])))
        corr_s = np.corrcoef(real_data[idx_s_min:idx_s_max], syn_data[idx_s_min:idx_s_max])[0, 1]

        axs[i, 1].plot(times, real_data, label="Real", color="red", lw=1)
        axs[i, 1].plot(times, syn_data, label="Synthetics", color="blue", lw=1, linestyle="-")
        axs[i, 1].axvline(t_s, color="black", linestyle=":", lw=1.5, label="S-wave")
        axs[i, 1].axvspan(t_s - window_size / 2, t_s + window_size / 2, color="lightgray", alpha=0.2)
        axs[i, 1].set_xlim([5, 13])
        axs[i, 1].set_ylim([-1, 1])
        axs[i, 1].set_title(f"{comp} Component - S wave")
        axs[i, 1].set_xlabel("Time (s)")
        axs[i, 1].set_ylabel("Amplitude")
        axs[i, 1].legend()
        axs[i, 1].text(0.05, 0.95, f"CC (janela): {corr_s:.3f}", transform=axs[i, 1].transAxes, fontsize=10, verticalalignment="top")

    plt.suptitle("Comparação entre traços reais e sintéticos", fontsize=14)
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, f"final_comparison_{event_id}.png"))

    # Corner plot
    params = np.vstack([tested_depths, tested_strikes, tested_dips, tested_rakes]).T
    figure = corner.corner(params, 
                           labels=["Depth (km)", "Strike (°)", "Dip (°)", "Rake (°)"],
                           show_titles=True,
                           title_fmt=".2f",
                           quantiles=[0.16, 0.5, 0.84],
                           title_kwargs={"fontsize": 16})
    figure.savefig(os.path.join(fig_dir, f"cornerplot_{event_id}.png"))

    # Save all tests
    results_df = pd.DataFrame({
        "Depth (km)": tested_depths,
        "Strike (°)": tested_strikes,
        "Dip (°)": tested_dips,
        "Rake (°)": tested_rakes,
        "Cost": tested_costs
    })
    results_csv_path = os.path.join(fig_dir, f"all_tested_parameters_{event_id}.csv")
    results_df.to_csv(results_csv_path, index=False)

    # Calculus of statistics per restart
    restart_stats = []
    n_total = len(tested_costs)
    costs_array = np.array(tested_costs)
    restart_steps = n_particles * restart_interval

    for i in range(0, n_total, restart_steps):
        block_costs = costs_array[i:i + restart_steps]
        if len(block_costs) == 0:
            continue
        restart_stats.append({
            "Restart": i // restart_steps + 1,
            "Mean Cost": np.mean(block_costs),
            "Std Cost": np.std(block_costs),
            "Min Cost": np.min(block_costs)
        })

    restart_df = pd.DataFrame(restart_stats)
    restart_csv_path = os.path.join(fig_dir, f"restart_cost_stats_{event_id}.csv")
    restart_df.to_csv(restart_csv_path, index=False)

    # Plot cost iterations
    plt.figure(figsize=(10, 5))
    x_vals = np.arange(len(tested_costs))
    plt.scatter(x_vals, tested_costs, label="Tested Costs", color="gray", s=10, alpha=0.5)

    best_costs = []
    iters_x = []
    for i in range(0, len(tested_costs), n_particles):
        chunk = tested_costs[i:i+n_particles]
        if chunk:
            best = np.min(chunk)
            best_costs.append(best)
            iters_x.append(i + n_particles // 2)
    plt.plot(iters_x, best_costs, label="Best per Iteration", color="blue", lw=2)

    restart_steps = n_particles * restart_interval
    centers = []
    means = []
    stds = []

    for i in range(0, len(tested_costs), restart_steps):
        block = tested_costs[i:i + restart_steps]
        if len(block) == 0:
            continue
        mu = np.mean(block)
        sigma = np.std(block)
        center = i + restart_steps // 2
        centers.append(center)
        means.append(mu)
        stds.append(sigma)

    centers = np.array(centers)
    means = np.array(means)
    stds = np.array(stds)

    plt.plot(centers, means, label="Mean per Restart", color="tomato", lw=2)
    plt.fill_between(centers, means - stds, means + stds, color="tomato", alpha=0.2, label="±1 Std Dev")
    plt.xlabel("Evaluation (particles)")
    plt.ylabel("Cost")
    plt.title("Evolution of tested costs and statistics per restart")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"tested_costs_over_iterations_{event_id}.png"))

    # Scatter of all particles
    n_iterations_run = len(tested_costs) // n_particles

    costs_array = np.array(tested_costs).reshape((n_iterations_run, n_particles))
    strikes_array = np.array(tested_strikes).reshape((n_iterations_run, n_particles))
    dips_array = np.array(tested_dips).reshape((n_iterations_run, n_particles))
    rakes_array = np.array(tested_rakes).reshape((n_iterations_run, n_particles))

    best_indices = np.argmin(costs_array, axis=1)
    iters = np.arange(n_iterations_run)

    fig1, axs1 = plt.subplots(2, 2, figsize=(14, 10))
    fig1.suptitle("All parameters and costs per particles x iteration", fontsize=16)
    param_arrays = [costs_array, strikes_array, dips_array, rakes_array]
    titles = ["Cost", "Strike", "Dip", "Rake"]
    colors = ["tab:red", "tab:blue", "tab:orange", "tab:green"]

    for ax, param, title, color in zip(axs1.flat, param_arrays, titles, colors):
        for i in range(n_particles):
            ax.scatter(iters, param[:, i], s=10, color=color, alpha=0.3)
        ax.scatter(iters, param[np.arange(n_iterations_run), best_indices], s=30, color="black", label="Best", zorder=10)
        ax.set_title(f"{title} vs Iteration")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(title)
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(fig_dir, f"PSO_particles_all_scatter_{event_id}.png"))

    # Best only
    fig2, axs2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle("Best parameters and cost per iteration", fontsize=16)
    for ax, param, title, color in zip(axs2.flat, param_arrays, titles, colors):
        best_series = param[np.arange(n_iterations_run), best_indices]
        ax.plot(iters, best_series, lw=2, color=color)
        ax.set_title(f"Best {title} over Iterations")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(title)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(fig_dir, f"PSO_best_only_lines_{event_id}.png"))