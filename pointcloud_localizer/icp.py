# icp.py — Hand-rolled Iterative Closest Point (ICP) registration.

from ast import List

from typing import List

import numpy as np
from scipy.spatial import KDTree


def svd(source_corr, target_corr):
    
    mu_s = source_corr.mean(axis=0)
    mu_t = target_corr.mean(axis=0)

    S = source_corr - mu_s  
    T = target_corr - mu_t   

    H = S.T @ T              
    U, _, Vt = np.linalg.svd(H)

    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])

    R = Vt.T @ D @ U.T
    t = mu_t - R @ mu_s
    return R, t


def Rt(points, R, t):
    return (R @ points.T).T + t


def rmse(source_t, target_corr):
    
    diff = source_t - target_corr
    return float(np.sqrt((diff ** 2).sum(axis=1).mean()))




def icp(source, target, max_iterations=100, tolerance= 1e-6, max_correspondence_dist= None, outlier_ratio= 0.0,verbose = True):
    
    T_total = np.eye(4)
    R_total = T_total[:3, :3].copy()
    t_total = T_total[:3, 3].copy()

    src = Rt(source.copy(), R_total, t_total)

    tree = KDTree(target)

    if max_correspondence_dist is None:
        sample_idx = np.random.choice(len(src), min(500, len(src)), replace=False)
        dists, _ = tree.query(src[sample_idx], k=1)
        max_correspondence_dist = 3.0 * float(np.median(dists))

    if verbose:
        print(f"source={len(source)} pts, target={len(target)} pts")
        print(f"max_correspondence_dist={max_correspondence_dist:.4f} m, "
              f"max_iter={max_iterations}, tol={tolerance:.2e}")

    rmse_history: List[float] = []
    inlier_counts: List[int] = []
    prev_rmse = np.inf

    for iteration in range(max_iterations):

        dists, indices = tree.query(src, k=1)
        dists = dists.ravel()
        indices = indices.ravel()

        valid_mask = dists < max_correspondence_dist

        if outlier_ratio > 0.0 and valid_mask.sum() > 10:
            valid_dists = dists[valid_mask]
            threshold = np.percentile(valid_dists, 100*(1.0 - outlier_ratio))
            valid_mask &= (dists < threshold)

        n_inliers = int(valid_mask.sum())
        if n_inliers < 6:
            if verbose:
                print(f"iter {iteration:3d}: too few inliers ({n_inliers}), stopping")
            break

        src_corr = src[valid_mask]
        tgt_corr = target[indices[valid_mask]]

        rmse = rmse(src_corr, tgt_corr)
        rmse_history.append(rmse)
        inlier_counts.append(n_inliers)

        if verbose:
            print(f"iter {iteration:3d}: RMSE={rmse:.6f} m, "
                  f"inliers={n_inliers}/{len(src)}")

        if abs(prev_rmse - rmse) < tolerance:
            if verbose:
                print(f"Converged at iteration {iteration} "
                      f"(ΔRMSE={abs(prev_rmse-rmse):.2e} < {tolerance:.2e})")
            R_step, t_step = svd(src_corr, tgt_corr)
            src = Rt(src, R_step, t_step)
            R_total = R_step @ R_total
            t_total = R_step @ t_total + t_step
            break
        prev_rmse = rmse

        R_step, t_step = svd(src_corr, tgt_corr)

        src = Rt(src, R_step, t_step)

        R_total = R_step @ R_total
        t_total = R_step @ t_total + t_step

    else:
        if verbose:
            print(f"Reached max iterations ({max_iterations})")

    T_est = np.eye(4)
    T_est[:3, :3] = R_total
    T_est[:3, 3] = t_total

    final_rmse = rmse_history[-1] if rmse_history else np.inf
    converged = len(rmse_history) > 0 and (
        abs(rmse_history[-1] - rmse_history[-2]) < tolerance
        if len(rmse_history) >= 2 else False
    )

    if verbose:
        print(f"\n Done — {len(rmse_history)} iterations, "
              f"final RMSE={final_rmse:.6f} m")
        print(f"Estimated transform:\n{T_est}")

    return {
        "transform": T_est,
        "rmse_history": rmse_history,
        "n_iterations": len(rmse_history),
        "converged": converged,
        "final_rmse": final_rmse,
        "inlier_counts": inlier_counts,
        "aligned_source": src,
    }
