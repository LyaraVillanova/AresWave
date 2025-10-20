import numpy as np
from obspy import UTCDateTime
from dsmpy.station_Mars import Station
from areswave.bayesian import run_full_analysis


# Event
event_id = 'mqs2022dulj'
latitude, longitude, distance, depth = 4.5024, 153.6234, 84.8, 30
magnitude = 3.0
time_p = UTCDateTime("2022-02-23T21:09:50")
time_s = UTCDateTime("2022-02-23T21:17:40")
centroid_time = UTCDateTime("2022-02-23T21:00:32")
Mrr, Mrt, Mrp, Mtt, Mtp, Mpp = -2.8e20, -1.9e20, -1.3e20, -1.4e20, -5.3e20, 1.8e20
stations = [Station(name='ELYSE', network='XB', latitude=4.502384, longitude=135.623447)]

# Models
model_directory = "/home/lyara/areswave/models/"
model_colors = {
    **dict.fromkeys(["dwak", "dwthot", "dwthotcrust1", "dwthotcrust1b", "eh45tcold", "eh45tcoldcrust1", "eh45tcoldcrust1b", "lfak", "sanak", 
                     "tayak", "maak", "gudkova"], "plum"),
    **dict.fromkeys(["MD_model1", "MD_model50", "MD_model100"], "mediumorchid"),
    **dict.fromkeys(["CD_model1", "CD_model50", "CD_model100"], "rebeccapurple"),
    **dict.fromkeys(["AK_model_2", "AK_model_50", "AK_model_100", "ceylan", "KKS21GP_blue_model", "KKS21GP_red_model", 
                     "Geophysical_model1", "Geophysical_model100", "Geophysical_model200", "Geophysical_model300", 
                     "Geophysical_model400", "Geophysical_model500", "Geophysical_model600", "Geophysical_model700", 
                     "Geophysical_model800", "Geophysical_model900", "Geophysical_model1000"], "slateblue")}
ordered_models = list(model_colors.keys())

# Parameters to calculate the depth
vp, vs = 4.3, 2.5 #5.6, 2.9
sigma_vp, sigma_vs = 0.6, 0.3

# Parameters to Bayesian
sp_sigma = 2.0
vp_mu, vp_sigma = 4.3, 0.6
vs_mu, vs_sigma = 2.5, 0.3
depth_lower, depth_upper = 0, 100

# Figures
output_fig1_path = "/home/lyara/areswave/figs/bayesian_depth_with_error_S1153a.png"
output_fig2_path = "/home/lyara/areswave/figs/bayesian_depth_gradient_S1153a.png"

# Main
if __name__ == "__main__":
    summary = run_full_analysis(
        event_id, latitude, longitude, distance, depth, magnitude, time_p, time_s,
        centroid_time, Mrr, Mrt, Mrp, Mtt, Mtp, Mpp, stations, model_directory,
        model_colors, ordered_models, vp, vs, sigma_vp, sigma_vs, sp_sigma,
        vp_mu, vp_sigma, vs_mu, vs_sigma, depth_lower, depth_upper,
        output_fig1_path, output_fig2_path)
    print(summary)