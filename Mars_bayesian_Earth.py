import numpy as np
from obspy import UTCDateTime
from obspy.geodetics import gps2dist_azimuth, kilometers2degrees
from dsmpy.station import Station
import areswave.bayesian as bayes
from areswave.bayesian import run_full_analysis
import pymc as pm

# Patches Earth
GLOBAL_DELTA_KM = None
def calculate_depth_fixed(sp, vp, vs, sig_vp, sig_vs, delta_km=None, sp_sigma=0.0):
    global GLOBAL_DELTA_KM
    if delta_km is None:
        delta_km = GLOBAL_DELTA_KM
    if not (np.isfinite(sp) and sp > 0):
        return np.nan, np.nan
    denom = (1.0/vs - 1.0/vp)
    R = sp / denom
    h2 = R*R - delta_km*delta_km
    if h2 <= 0:
        return 0.0, 0.0
    h = np.sqrt(h2)
    rel = (sig_vs/vs)**2 + (sig_vp/vp)**2
    if sp_sigma > 0 and sp > 0:
        rel += (sp_sigma/sp)**2
    dR = R * np.sqrt(rel)
    dh = (R/h)*dR
    return float(h), float(dh)
bayes.calculate_depth = calculate_depth_fixed

def run_bayesian_inference_fixed(sp_time_obs, sp_sigma,
                                 vp_mu, vp_sigma, vs_mu, vs_sigma,
                                 depth_lower, depth_upper):
    global GLOBAL_DELTA_KM
    delta_km = GLOBAL_DELTA_KM
    with pm.Model() as model:
        vp = pm.Normal("vp", mu=vp_mu, sigma=vp_sigma)
        vs = pm.Normal("vs", mu=vs_mu, sigma=vs_sigma)
        depth = pm.Uniform("depth", lower=depth_lower, upper=depth_upper)
        path_km = pm.math.sqrt(delta_km**2 + depth**2)
        t_sp = path_km * (1.0/vs - 1.0/vp)
        pm.Normal("t_sp_obs", mu=t_sp, sigma=sp_sigma, observed=sp_time_obs)
        trace = pm.sample(5000, tune=2000, cores=2,
                          return_inferencedata=True, progressbar=False)
    return trace

bayes.run_bayesian_inference = run_bayesian_inference_fixed

# Event
event_id = 'IV_AIO'
latitude, longitude, depth = 37.8215, 14.5907, 31
magnitude = 3.3
time_p = UTCDateTime("2020-11-18T02:06:37")
time_s = UTCDateTime("2020-11-18T02:06:46")
centroid_time = UTCDateTime((time_p.timestamp + time_s.timestamp) / 2)
Mrr, Mrt, Mrp, Mtt, Mtp, Mpp = -2.8e20, -1.9e20, -1.3e20, -1.4e20, -5.3e20, 1.8e20

stations = [Station(name='AIO', network='IV', latitude=37.9712, longitude=15.233)]

# Distance
dist_m, _, _ = gps2dist_azimuth(latitude, longitude,
                                stations[0].latitude, stations[0].longitude)
distance_km = dist_m/1000.0
distance_deg = kilometers2degrees(distance_km)
GLOBAL_DELTA_KM = distance_km
print(f"Δ = {distance_km:.1f} km ({distance_deg:.3f}°)")
print(f"S–P observado = {(time_s - time_p):.2f} s")

# Models
model_directory = "/home/lyara/areswave/models/"
model_colors = {
    **dict.fromkeys(["ak135"], "plum"),
    **dict.fromkeys(["prem"], "slateblue")}
ordered_models = list(model_colors.keys())

# Parameters
vp, vs = 6.3, 3.5
sigma_vp, sigma_vs = 0.1, 0.1
sp_sigma = 0.05
vp_mu, vp_sigma = 6.3, 0.1
vs_mu, vs_sigma = 3.5, 0.1
depth_lower, depth_upper = 10, 60

# Figures
output_fig1_path = "/home/lyara/areswave/figs/bayesian_depth_with_error_Earth.png"
output_fig2_path = "/home/lyara/areswave/figs/bayesian_depth_gradient_Earth.png"

# Main
if __name__ == "__main__":
    summary = run_full_analysis(
        event_id, latitude, longitude, distance_deg, depth, magnitude, time_p, time_s,
        centroid_time, Mrr, Mrt, Mrp, Mtt, Mtp, Mpp, stations, model_directory,
        model_colors, ordered_models, vp, vs, sigma_vp, sigma_vs, sp_sigma,
        vp_mu, vp_sigma, vs_mu, vs_sigma, depth_lower, depth_upper,
        output_fig1_path, output_fig2_path)
    print(summary)