import os
from areswave.inversion import read_metadata, read_nd_model, misfit, save_nd_model, compute_gradient, update_model

# Directories
SYNTHETICS_FOLDER = '/home/lyara/areswave/dsmpy/synthetics'
OBSERVED_FOLDER = '/home/lyara/areswave/dsmpy/SAC'
SAVE_MODELS_DIR = '/home/lyara/areswave/dsmpy/models'
PLOTS_DIR = '/home/lyara/areswave/dsmpy/figs'
METADATA_CSV = '/home/lyara/areswave/dsmpy/metadata.csv'

# Configurations
MISFIT_TYPE = 'l2'
N_ITER = 5
ZOOM_DEPTH = (800, 1600)

# Main
if __name__ == "__main__":
    os.makedirs(SAVE_MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    metadata = read_metadata(METADATA_CSV)
    initial_models = sorted([f for f in os.listdir(SAVE_MODELS_DIR) if f.endswith(".nd")])

    for model_file in initial_models:
        model_path = os.path.join(SAVE_MODELS_DIR, model_file)
        model = read_nd_model(model_path)
        depth = model['depth']
        model_history = [model]
        error_history = []

        for i in range(N_ITER):
            print(f"Iteração {i+1} | Modelo {model_file}")
            erro_total = misfit(metadata, MISFIT_TYPE)
            error_history.append(erro_total)
            print(f"Misfit total: {erro_total:.4f}")

            save_nd_model(f"{SAVE_MODELS_DIR}/{model_file[:-3]}_iter_{i}.nd", depth, model['rho'], model['vp'], model['vs'])
            grad = compute_gradient(None, None, model)
            model = update_model(model, grad)
            model_history.append(model)

        print(f"Inversão concluída para: {model_file}")
