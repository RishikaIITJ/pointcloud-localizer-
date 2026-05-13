
# evaluate.py — Quantitative evaluation and visualisation of ICP results.

import numpy as np
import matplotlib
matplotlib.use("Agg")          
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def rotation_error(T_est, T_gt):
   
    R_est = T_est[:3, :3]
    R_gt = T_gt[:3, :3]
    R_err = R_est.T @ R_gt
   
    cos_angle = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def translation_error(T_est, T_gt):
    t_est = T_est[:3, 3]
    t_gt = T_gt[:3, 3]
    return float(np.linalg.norm(t_est - t_gt))


def evaluate_registration(T_est, T_gt, rmse_history, n_iterations):
    
    rot_err = rotation_error(T_est, T_gt)
    trans_err = translation_error(T_est, T_gt)
    final_rmse = rmse_history[-1] if rmse_history else np.nan

    print("\n" + "=" * 55)
    print("  EVALUATION RESULTS")
    print("=" * 55)
    print(f"  Rotation error    : {rot_err:.4f} °")
    print(f"  Translation error : {trans_err:.6f} m")
    print(f"  Final RMSE        : {final_rmse:.6f} m")
    print(f"  ICP iterations    : {n_iterations}")
    print("=" * 55 + "\n")

    return {
        "rotation_error_deg": rot_err,
        "translation_error_m": trans_err,
        "final_rmse": final_rmse,
        "n_iterations": n_iterations,
        "rmse_history": rmse_history,
    }

def _subsample_for_plot(pts, max_pts= 3000):
    if len(pts) <= max_pts:
        return pts
    idx = np.random.choice(len(pts), max_pts, replace=False)
    return pts[idx]


def plot_before_after(source, target, aligned_source, output_path= "output/before_after.png"):
   
    s = _subsample_for_plot(source)
    t = _subsample_for_plot(target)
    a = _subsample_for_plot(aligned_source)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Point Cloud Registration — Before & After", fontsize=14, fontweight="bold")

    projections = [(0, 1, "X", "Y"), (0, 2, "X", "Z"), (1, 2, "Y", "Z")]

    for col, (i, j, xl, yl) in enumerate(projections):
      
        ax = axes[0, col]
        ax.scatter(t[:, i], t[:, j], s=1, c="steelblue", alpha=0.4, label="Target")
        ax.scatter(s[:, i], s[:, j], s=1, c="tomato", alpha=0.4, label="Source (before)")
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_title(f"Before — {xl}{yl} projection")
        ax.legend(markerscale=5, fontsize=7)
        ax.set_aspect("equal")

    
        ax = axes[1, col]
        ax.scatter(t[:, i], t[:, j], s=1, c="steelblue", alpha=0.4, label="Target")
        ax.scatter(a[:, i], a[:, j], s=1, c="seagreen", alpha=0.4, label="Aligned source")
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_title(f"After — {xl}{yl} projection")
        ax.legend(markerscale=5, fontsize=7)
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
   


def plot_rmse_curve(rmse_history,output_path: str = "output/rmse_curve.png"):
  
    fig, ax = plt.subplots(figsize=(8, 4))
    iters = np.arange(1, len(rmse_history) + 1)
    ax.plot(iters, rmse_history, "o-", color="royalblue", markersize=4, linewidth=1.5, label="RMSE")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("RMSE (m)")
    ax.set_title("ICP Convergence — RMSE per Iteration")
    ax.grid(True, alpha=0.4)
    ax.legend()

    
    ax.axhline(rmse_history[-1], color="tomato", linestyle="--", linewidth=1, label=f"Final={rmse_history[-1]:.5f} m")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    

def run_robustness_sweep(output_dir= "output"):

    from pointcloud_localizer.synthetic import generate_synthetic_pair
    from pointcloud_localizer.preprocess import voxel
    from pointcloud_localizer.icp import icp


    noise_levels = [0.0, 0.005, 0.02]        
    misalign_angles = [5.0, 15.0, 30.0]        
  
    rot_errors = np.zeros((len(noise_levels), len(misalign_angles)))
    trans_errors = np.zeros_like(rot_errors)
    rmse_final = np.zeros_like(rot_errors)
    n_iters_grid = np.zeros_like(rot_errors)

    print("\n[sweep] Running robustness sweep …")
    for i, sigma in enumerate(noise_levels):
        for j, mis_deg in enumerate(misalign_angles):
            mis_rad = np.radians(mis_deg)
            src, tgt, T_gt = generate_synthetic_pair(
                n_points=2000, noise_sigma=sigma,
                rx=0.1, ry=0.15, rz=0.05,
                tx=0.1, ty=0.05, tz=0.02,
                seed=42
            )
            src_ds = voxel(src, 0.05)
            tgt_ds = voxel(tgt, 0.05)

            from pointcloud_localizer.synthetic import make_transform, apply_transform
            T_mis = make_transform(mis_rad, mis_rad * 0.5, 0, 0.05, 0.03, 0.01)
            src_mis = apply_transform(src_ds, T_mis)

            result = icp(src_mis, tgt_ds, max_iterations=150,
                         outlier_ratio=0.1, verbose=False)

            T_est = result["transform"]
           
            T_combined = T_est @ T_mis

            rot_err = rotation_error(T_combined, T_gt)
            trans_err = translation_error(T_combined, T_gt)

            rot_errors[i, j] = rot_err
            trans_errors[i, j] = trans_err
            rmse_final[i, j] = result["final_rmse"]
            n_iters_grid[i, j] = result["n_iterations"]

            print(f"  σ={sigma:.3f} m | mis={mis_deg:5.1f}° → "
                  f"rot_err={rot_err:.2f}°, trans_err={trans_err:.4f} m, "
                  f"RMSE={result['final_rmse']:.5f} m, "
                  f"iters={result['n_iterations']}")

   
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Robustness Sweep: ICP Performance vs Noise & Misalignment",
                 fontsize=13, fontweight="bold")

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    noise_labels = [f"σ={s}" for s in noise_levels]
    mis_labels = [f"{m}°" for m in misalign_angles]
    cmap = "YlOrRd"

    def _heat(ax, data, title, fmt=".2f", unit=""):
        im = ax.imshow(data, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(mis_labels))); ax.set_xticklabels(mis_labels)
        ax.set_yticks(range(len(noise_labels))); ax.set_yticklabels(noise_labels)
        ax.set_xlabel("Initial misalignment")
        ax.set_ylabel("Noise level (m)")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.8)
        for r in range(data.shape[0]):
            for c in range(data.shape[1]):
                ax.text(c, r, f"{data[r,c]:{fmt}}{unit}",
                        ha="center", va="center", fontsize=9,
                        color="black" if data[r, c] < data.max() * 0.6 else "white")

    _heat(fig.add_subplot(gs[0, 0]), rot_errors,   "Rotation Error (°)",     ".2f", "°")
    _heat(fig.add_subplot(gs[0, 1]), trans_errors,  "Translation Error (m)",  ".4f", "m")
    _heat(fig.add_subplot(gs[1, 0]), rmse_final,    "Final RMSE (m)",         ".5f", "m")
    _heat(fig.add_subplot(gs[1, 1]), n_iters_grid,  "ICP Iterations",         ".0f", "")

    out = str(output_dir/ "robustness_sweep.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[sweep] Saved robustness sweep plot → {out}")

    print("\n{:<10} {:<12} {:<18} {:<20} {:<16} {:<10}".format(
        "Noise(m)", "Misalign(°)", "Rot error(°)", "Trans error(m)",
        "Final RMSE(m)", "Iters"))
    print("-" * 90)
    for i, sigma in enumerate(noise_levels):
        for j, mis_deg in enumerate(misalign_angles):
            print(f"{sigma:<10.3f} {mis_deg:<12.1f} "
                  f"{rot_errors[i,j]:<18.3f} {trans_errors[i,j]:<20.5f} "
                  f"{rmse_final[i,j]:<16.5f} {int(n_iters_grid[i,j]):<10}")
