import os
import math
import traceback
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def strike_dip_rake_to_vectors(strike_deg, dip_deg, rake_deg):
    strike = math.radians(strike_deg)
    dip = math.radians(dip_deg)
    rake = math.radians(rake_deg)

    # s: strike unit vector (East, North, Up)
    s = np.array([math.sin(strike), math.cos(strike), 0.0])

    # dip azimuth = strike + 90°
    dip_az = strike + math.pi/2.0

    # d: down-dip unit vector (points into Earth => negative Up)
    d = np.array([
        math.sin(dip_az) * math.cos(dip),
        math.cos(dip_az) * math.cos(dip),
        -math.sin(dip)
    ])
    d /= (np.linalg.norm(d) + 1e-12)

    # normal n = s x d
    n = np.cross(s, d)
    n /= (np.linalg.norm(n) + 1e-12)

    # slip vector (measured from strike towards down-dip)
    slip = math.cos(rake) * s + math.sin(rake) * d
    slip /= (np.linalg.norm(slip) + 1e-12)

    return s, d, n, slip

def plane_mesh_from_basis(n, strike_deg, size=1.2, res=30):
    strike_rad = math.radians(strike_deg)
    s_vec = np.array([math.sin(strike_rad), math.cos(strike_rad), 0.0])
    p_vec = np.cross(n, s_vec)
    norm_p = np.linalg.norm(p_vec)
    if norm_p < 1e-8:
        arbitrary = np.array([1.0, 0.0, 0.0])
        p_vec = np.cross(n, arbitrary)
        norm_p = np.linalg.norm(p_vec) + 1e-12
    p_vec = p_vec / norm_p
    s_vec = s_vec / (np.linalg.norm(s_vec) + 1e-12)

    u = np.linspace(-size, size, res)
    v = np.linspace(-size, size, res)
    U, V = np.meshgrid(u, v)
    X = U * s_vec[0] + V * p_vec[0]
    Y = U * s_vec[1] + V * p_vec[1]
    Z = U * s_vec[2] + V * p_vec[2]
    return X.astype(float), Y.astype(float), Z.astype(float)

def draw_cube_edges(ax, L=1.2, color='gray'):
    box = np.array([
        [-L, -L, -L], [ L, -L, -L],
        [ L,  L, -L], [-L,  L, -L],
        [-L, -L,  L], [ L, -L,  L],
        [ L,  L,  L], [-L,  L,  L],
    ])
    edges = [(0,1),(1,2),(2,3),(3,0),
             (4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    for i,j in edges:
        xs = [box[i,0], box[j,0]]
        ys = [box[i,1], box[j,1]]
        zs = [box[i,2], box[j,2]]
        ax.plot(xs, ys, zs, color=color, linewidth=0.8, alpha=0.9)

def plot_and_save_fault(strike, dip, rake, depth, cost=None,
                       model_name="Model", output_dir="./figs"):
    s, d, n, slip = strike_dip_rake_to_vectors(strike, dip, rake)

    fig = plt.figure(figsize=(12,6))

    # -------- Map view (left) --------
    ax1 = fig.add_subplot(1,2,1)
    ax1.set_title("Map view")
    ax1.set_xlabel("E")
    ax1.set_ylabel("N")
    ax1.set_xlim(-1.1, 1.1)
    ax1.set_ylim(-1.1, 1.1)
    ax1.grid(True, linewidth=0.3)

    # Fault trace (line along strike)
    L = 0.9
    ax1.plot([-s[0]*L, s[0]*L], [-s[1]*L, s[1]*L], color="goldenrod", linewidth=4)

    # slip horizontal projection
    slip_h = np.array([slip[0], slip[1]])
    slip_hn = slip_h / (np.linalg.norm(slip_h) + 1e-12)
    ax1.arrow(0, 0, slip_hn[0]*0.5, slip_hn[1]*0.5, head_width=0.04, color='skyblue', length_includes_head=True)
    ax1.text(slip_hn[0]*0.55, slip_hn[1]*0.55, "slip (horiz)", color='skyblue')

    # down-dip projection
    dip_h = np.array([d[0], d[1]])
    ddn = dip_h / (np.linalg.norm(dip_h) + 1e-12)
    ax1.arrow(0, 0, ddn[0]*0.5, ddn[1]*0.5, head_width=0.04, color='k', length_includes_head=True)
    ax1.text(ddn[0]*0.55, ddn[1]*0.55, "down-dip", color='k')

    ax1.scatter([0], [0], s=30)
    ax1.set_aspect("equal")

    info = f"Strike={strike:.2f}°, Dip={dip:.2f}°, Rake={rake:.2f}°\nDepth={depth:.2f} km"
    if cost is not None:
        info += f", Cost={cost:.4f}"
    ax1.text(-1.05, 1.05, info, fontsize=10, va='top')

    # -------- 3D view (right) --------
    ax2 = fig.add_subplot(1,2,2, projection='3d')
    ax2.set_title("3D view (block)")
    ax2.set_xlabel("E")
    ax2.set_ylabel("N")
    ax2.set_zlabel("Down (relative)")

    Lblock = 1.2
    draw_cube_edges(ax2, L=Lblock, color='gray')

    try:
        Xp, Yp, Zp = plane_mesh_from_basis(n, strike_deg=strike, size=1.2, res=32)
        if np.isfinite(Xp).all() and np.isfinite(Yp).all() and np.isfinite(Zp).all():
            ax2.plot_surface(Xp, Yp, Zp, color='orange', alpha=0.5, linewidth=0, rstride=1, cstride=1)
        else:
            print("[WARN] Plano principal contém valores não-finitos — pulando plot_surface principal.")
    except Exception as e:
        print("[WARN] Falha ao construir/plottar plano principal:", e)
        traceback.print_exc()

    try:
        n_aux = np.cross(slip, n)
        n_aux /= (np.linalg.norm(n_aux) + 1e-12)
        Xa, Ya, Za = plane_mesh_from_basis(n_aux, strike_deg=strike, size=1.2, res=32)
        if np.isfinite(Xa).all() and np.isfinite(Ya).all() and np.isfinite(Za).all():
            ax2.plot_surface(Xa, Ya, Za, color='turquoise', alpha=0.5, linewidth=0, rstride=1, cstride=1)
        else:
            print("[WARN] Plano auxiliar contém valores não-finitos — pulando plot_surface auxiliar.")
    except Exception as e:
        print("[WARN] Falha ao construir/plottar plano auxiliar:", e)
        traceback.print_exc()

    sv = slip * 0.8
    ax2.quiver(0, 0, 0, sv[0], sv[1], sv[2], length=1.0, normalize=False, color='skyblue')
    ax2.text(sv[0]*1.05, sv[1]*1.05, sv[2]*1.05, "slip", color='skyblue')

    nn = n * 0.6
    ax2.quiver(0, 0, 0, nn[0], nn[1], nn[2], length=1.0, normalize=False, color='k')
    ax2.text(nn[0]*1.05, nn[1]*1.05, nn[2]*1.05, "n (fault normal)", color='k')

    ax2.set_xlim(-Lblock, Lblock)
    ax2.set_ylim(-Lblock, Lblock)
    ax2.set_zlim(-Lblock, Lblock)
    try:
        ax2.view_init(elev=18, azim=-65)
        ax2.set_box_aspect((1,1,0.6))
    except Exception:
        # some older matplotlib versions don't have set_box_aspect
        pass

    fig.suptitle(f"{model_name} - Fault geometry", fontsize=14)

    # --- save ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = f"fault_{model_name}.png"
    filepath = os.path.join(output_dir, filename)

    try:
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {filepath}")
        return filepath
    except Exception as e:
        print("[ERROR] Full figure save failed:", e)
        traceback.print_exc()
        try:
            map_fname = os.path.join(output_dir, f"fault_{model_name}_maponly.png")
            fig_map, ax_map = plt.subplots(figsize=(6,6))
            ax_map.set_xlim(-1.1,1.1); ax_map.set_ylim(-1.1,1.1)
            ax_map.plot([-s[0]*0.9, s[0]*0.9], [-s[1]*0.9, s[1]*0.9], color="goldenrod", linewidth=4)
            ax_map.arrow(0,0, slip[0]*0.5, slip[1]*0.5, head_width=0.04, color='skyblue', length_includes_head=True)
            ax_map.text(-1.05, 1.05, info if 'info' in locals() else '', fontsize=10, va='top')
            ax_map.set_xlabel("E"); ax_map.set_ylabel("N"); ax_map.grid(True)
            fig_map.suptitle(f"{model_name} - map-only (fallback)", fontsize=12)
            fig_map.savefig(map_fname, dpi=300, bbox_inches='tight')
            plt.close(fig_map)
            print(f"Map-only fallback saved: {map_fname}")
            return map_fname
        except Exception as e2:
            print("[FATAL] Fallback map-only save failed:", e2)
            traceback.print_exc()
            raise