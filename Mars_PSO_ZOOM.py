import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === 1. CARREGAR OS DADOS ===
df = pd.read_csv("figs/all_tested_parameters_S0185a_mqs2019kxjd.csv")

# Ajuste: defina o número de partículas por iteração (use o valor correto do seu PSO)
n_particles = 40
n_iterations = len(df) // n_particles

# Converter em arrays 2D (iterações x partículas)
depths_array = df["Depth (km)"].values.reshape(n_iterations, n_particles)
costs_array = df["Cost"].values.reshape(n_iterations, n_particles)

# Índices da melhor partícula por iteração
best_indices = np.argmin(costs_array, axis=1)
iters = np.arange(n_iterations)

# === 2. CRIAR FIGURA ===
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# --- (A) COST vs ITERATION (zoom) ---
for i in range(n_particles):
    axes[0].scatter(iters, costs_array[:, i],
                    s=15, c="tab:red", alpha=0.3, marker='o')
axes[0].scatter(iters, costs_array[np.arange(n_iterations), best_indices],
                s=40, c="black", label="Best", zorder=3)

axes[0].set_title("Cost vs Iteration (zoomed)")
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("Cost")
axes[0].set_ylim(0.050, 0.5)   # ESCALA LINEAR
axes[0].legend()

# --- (B) DEPTH vs ITERATION ---
for i in range(n_particles):
    axes[1].scatter(iters, depths_array[:, i],
                    s=15, c="tab:purple", alpha=0.3, marker='o')
axes[1].scatter(iters, depths_array[np.arange(n_iterations), best_indices],
                s=40, c="black", label="Best", zorder=3)

axes[1].set_title("Depth vs Iteration")
axes[1].set_xlabel("Iteration")
axes[1].set_ylabel("Depth (km)")
axes[1].legend()

plt.tight_layout()
plt.savefig("figs/PSO_cost_depth_zoom_S0185a_mqs2019kxjd.png", dpi=300)