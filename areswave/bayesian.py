import numpy as np
import matplotlib.pyplot as plt
from obspy.taup import TauPyModel
from dsmpy.event_Mars import Event, MomentTensor
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import os
import pymc as pm
import arviz as az
from scipy.stats import gaussian_kde
import matplotlib.collections as mcoll


def process_model(model_name, depth, distance, model_directory):
    try:
        tau_model = TauPyModel(model=os.path.join(model_directory, model_name.lower()))
        arrivals = tau_model.get_travel_times(source_depth_in_km=depth, distance_in_degree=distance, phase_list=["P", "S"])
        return [(model_name, arr.name, arr.time) for arr in arrivals]
    except Exception as e:
        print(f"Erro no modelo {model_name}: {e}")
        return []

def calculate_depth(sp, vp, vs, sig_vp, sig_vs):
    if sp > 0:
        d = sp / 100 / (1 / vs - 1 / vp)
        e = d * np.sqrt((sig_vs / vs)**2 + (sig_vp / vp)**2)
        return d, e
    return np.nan, np.nan

def colorline(x, y, cmap='plasma', linewidth=2):
    z = y / y.max()
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = mcoll.LineCollection(segments, cmap=cmap, norm=plt.Normalize(0, 1))
    lc.set_array(z)
    lc.set_linewidth(linewidth)
    return lc

def plot_sp_times_and_depths(df, ordered_models, model_colors, fig_path):
    mean_sp = df["SP_Times"].mean()
    mean_depth = df["Depth (km)"].mean()
    mean_error = df["Error (km)"].mean()
    df["Model"] = pd.Categorical(df["Model"], categories=ordered_models + ["Real Data"], ordered=True)
    df = df.sort_values(by="Model")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    colors = [model_colors.get(m, "black") for m in df["Model"].astype(str)]
    bars = axes[0].bar(df["Model"].astype(str), df["SP_Times"], color=colors, alpha=0.6)
    axes[0].bar("Average", mean_sp, color="teal", alpha=0.6)
    axes[0].set_ylabel("S-P time (s)")
    axes[0].set_title("S-P times by model")
    axes[0].grid(False) #True, linestyle="--", alpha=0.3)
    axes[0].set_ylim(0, 700)
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=90)
    for bar in bars:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2, yval + 10, round(yval, 1), ha='center', va='bottom', rotation=90, fontsize=8)
    axes[0].text(len(df["Model"]) - 0.5, mean_sp + 10, round(mean_sp, 1), ha='center', va='bottom', rotation=90, fontsize=8, color='teal')
    depths = df["Depth (km)"]
    errors = df["Error (km)"]
    models = df["Model"].astype(str)
    axes[1].errorbar(models, depths, yerr=errors, fmt="o", color="black", ecolor=colors, alpha=0.6)
    axes[1].errorbar("Average", mean_depth, yerr=mean_error, fmt="o", color="teal", ecolor="turquoise", alpha=0.8)
    for i, (x_label, y, e) in enumerate(zip(models, depths, errors)):
        if not np.isnan(y):
            label = f"{y:.1f}±{e:.1f}"
            axes[1].text(i - 0.6, y + 0.5, label, ha='left', va='bottom', rotation=90, fontsize=8, color='black')
    average_index = len(models)
    label_avg = f"{mean_depth:.1f}±{mean_error:.1f}"
    axes[1].text(average_index + 0.15, mean_depth + 0.5, label_avg, ha='left', va='bottom', rotation=90, fontsize=8, color='teal')
    axes[1].set_ylim(15, 40)
    axes[1].set_ylabel("Depth (km)")
    axes[1].set_title("Depth estimates with uncertainties")
    axes[1].grid(False) #True, linestyle="--", alpha=0.3)
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=90)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)

def run_bayesian_inference(sp_time_obs, sp_sigma, vp_mu, vp_sigma, vs_mu, vs_sigma, depth_lower, depth_upper):
    with pm.Model() as model:
        vp = pm.Normal("vp", mu=vp_mu, sigma=vp_sigma)
        vs = pm.Normal("vs", mu=vs_mu, sigma=vs_sigma)
        depth = pm.Uniform("depth", lower=depth_lower, upper=depth_upper)
        t_sp = 100 * depth * (1 / vs - 1 / vp)
        pm.Normal("t_sp_obs", mu=t_sp, sigma=sp_sigma, observed=sp_time_obs)
        trace = pm.sample(10000, tune=5000, cores=2, return_inferencedata=True)
    return trace

def plot_posterior_distribution(trace, fig_path):
    depth_samples = trace.posterior["depth"].values.flatten()

    # KDE
    kde = gaussian_kde(depth_samples)
    x = np.linspace(5, 60, 500)
    y = kde(x)

    # Statistics
    mean_val = depth_samples.mean()
    median_val = np.median(depth_samples)
    mode_val = x[np.argmax(y)]
    hdi_bounds = az.hdi(depth_samples, hdi_prob=0.95)
    hdi_low, hdi_high = hdi_bounds[0], hdi_bounds[1]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    lc = colorline(x, y, cmap='plasma')
    ax.add_collection(lc)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0, y.max() * 1.2)
    ax.set_xlabel("Depth (km)")
    ax.set_ylabel("Density")
    ax.set_title("Posterior distribution of depth")
    ax.grid(False) #True, linestyle='--', alpha=0.3)
    ax.hlines(0, hdi_low, hdi_high, color="black", linewidth=3)
    ax.text((hdi_low + hdi_high)/2, y.max() * 0.03, f"95% HDI: {hdi_low:.1f}–{hdi_high:.1f}", ha="center", va="bottom", fontsize=10)
    stats = [
        (mean_val, "Mean", 'teal', 1.05),
        (median_val, "Median", 'darkorange', 1.10),
        (mode_val, "Mode", 'purple', 1.15)]
    for val, label, color, y_factor in stats:
        ax.axvline(val, color=color, linestyle="--", alpha=0.8, linewidth=1)
        ax.text(val, y.max() * y_factor, f"{label} = {val:.1f}", ha="center", fontsize=9, color=color)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)

def summarize_trace(trace):
    summary = az.summary(trace, var_names=["depth", "vp", "vs"], round_to=2)
    print(summary)
    return summary

def run_full_analysis(
    event_id, latitude, longitude, distance, depth, magnitude, time_p, time_s,
    centroid_time, Mrr, Mrt, Mrp, Mtt, Mtp, Mpp, stations, model_directory,
    model_colors, ordered_models, vp, vs, sigma_vp, sigma_vs, sp_sigma,
    vp_mu, vp_sigma, vs_mu, vs_sigma, depth_lower, depth_upper,
    output_fig1_path, output_fig2_path):
    mt = MomentTensor(Mrr, Mrt, Mrp, Mtt, Mtp, Mpp)
    event = Event(
        event_id=event_id, latitude=latitude, longitude=longitude,
        depth=depth, mt=mt, centroid_time=centroid_time.timestamp, source_time_function=None)
    with ThreadPoolExecutor() as executor:
        results = sum(executor.map(lambda model_name: process_model(model_name, depth, distance, model_directory), ordered_models), [])
    df = pd.DataFrame(results, columns=["Model", "Phase", "Travel_Time"])
    df_real = pd.DataFrame({
        "Model": ["Real Data", "Real Data"],
        "Phase": ["P", "S"],
        "Travel_Time": [time_p.timestamp, time_s.timestamp]})
    df = pd.concat([df, df_real], ignore_index=True)
    df = df.drop_duplicates(subset=["Model", "Phase"])
    df = df.pivot(index="Model", columns="Phase", values="Travel_Time").reset_index()
    df["SP_Times"] = df["S"] - df["P"]
    df[["Depth (km)", "Error (km)"]] = df["SP_Times"].apply(
        lambda x: pd.Series(calculate_depth(x, vp, vs, sigma_vp, sigma_vs)))

    plot_sp_times_and_depths(df, ordered_models, model_colors, output_fig1_path)

    sp_time_obs = df.loc[df["Model"] == "Real Data", "SP_Times"].values[0]
    trace = run_bayesian_inference(sp_time_obs, sp_sigma, vp_mu, vp_sigma, vs_mu, vs_sigma, depth_lower, depth_upper)

    plot_posterior_distribution(trace, output_fig2_path)

    summary = summarize_trace(trace)
    return summary