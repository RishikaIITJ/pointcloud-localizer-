import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


def rotation_error(T_est, T_gt):
    R_err = T_est[:3, :3].T @ T_gt[:3, :3]
    cos_theta = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def translation_error(T_est, T_gt):
    return float(np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3]))


def evaluate_registration(T_est, T_gt, rmse_history, n_iterations):
    rot_err   = rotation_error(T_est, T_gt)
    trans_err = translation_error(T_est, T_gt)
    final_rmse = rmse_history[-1] if rmse_history else np.nan

    print(f"  Rotation error    : {rot_err:.4f} deg")
    print(f"  Translation error : {trans_err:.6f} m")
    print(f"  Final RMSE        : {final_rmse:.6f} m")
    print(f"  Iterations        : {n_iterations}")
   
    return {
        "rotation_error":  rot_err,
        "translation_error": trans_err,
        "final_rmse":          final_rmse,
        "n_iterations":        n_iterations
    }


def before_after(source,target,aligned_source,output_path="output/before_after.png"):

    fig,axes = plt.subplots(1,2,figsize=(14, 5))
    fig.patch.set_facecolor("#1e1e1e")
    
    plots = [(axes[0],source,target,"red","blue","Source","Target","Before ICP"),
             (axes[1],aligned_source,target,"green","blue","Aligned source","Target","After ICP")]

    for ax, pts1, pts2, c1, c2, l1, l2, title in plots:
        ax.set_facecolor("#2b2b2b")
        ax.scatter(pts2[:, 0], pts2[:, 1],s=2,c=c2,alpha=0.5,label=l2)
        ax.scatter(pts1[:, 0], pts1[:, 1],s=2,c=c1,alpha=0.5,label=l1)
        ax.set_title(title, color="white")
        ax.set_xlabel("X", color="white")
        ax.set_ylabel("Y", color="white")
        ax.tick_params(colors="white")
        ax.legend(markerscale=4,fontsize=8,facecolor="#3a3a3a", labelcolor="white")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def rmse_curve(rmse_history,output_path="output/rmse_curve.png"):

    fig, ax = plt.subplots(figsize=(8,4))
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#2b2b2b")
    ax.plot(rmse_history,marker="o",color="royalblue",markersize=4,linewidth=1.5)
    ax.axhline(rmse_history[-1],color="tomato",linestyle="--",linewidth=1,label=f"Final_RMSE={rmse_history[-1]:.5f} m")
    ax.set_title("RMSE Convergence Curve", color="white")
    ax.set_xlabel("Iterations", color="white")
    ax.set_ylabel("RMSE (m)", color="white")
    ax.tick_params(colors="white")
    ax.grid()
    ax.legend(facecolor="#3a3a3a", labelcolor="white")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
   
def sweep(rot_errors, trans_errors, rmse_final, n_iters,noise_levels, misalignments,output_path="output/robustness.png"):

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Robustness Sweep Results",fontsize=16,fontweight="bold")
    gs = gridspec.GridSpec(2,2,figure=fig,hspace=0.35,wspace=0.25)

    def heat(ax,data,title):
        im = ax.imshow(data,cmap="YlOrRd")
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Initial Misalignment (deg)")
        ax.set_ylabel("Noise Level")
        ax.set_xticks(range(len(misalignments)))
        ax.set_xticklabels([f"{m}°" for m in misalignments])
        ax.set_yticks(range(len(noise_levels)))
        ax.set_yticklabels([f"{n}" for n in noise_levels])
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                ax.text(j,i,f"{val:.3f}",ha="center",va="center",color="black")
        plt.colorbar(im, ax=ax)

    heat(fig.add_subplot(gs[0, 0]),rot_errors,  "Rotation Error (deg)")
    heat(fig.add_subplot(gs[0, 1]),trans_errors, "Translation Error (m)")
    heat(fig.add_subplot(gs[1, 0]),rmse_final,   "Final RMSE (m)")
    heat(fig.add_subplot(gs[1, 1]),n_iters,      "ICP Iterations")

    plt.savefig(output_path)
    
   