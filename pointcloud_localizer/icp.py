import numpy as np
from scipy.spatial import KDTree


def transform(points, R, t):
    return (R @ points.T).T + t

def compute_rmse(source_corr, target_corr):
    diff = source_corr - target_corr
    return float(np.sqrt((diff**2).sum(axis=1).mean()))

def svd_rotation(source_corr, target_corr):
    mu_s = source_corr.mean(axis=0)
    mu_t = target_corr.mean(axis=0)
    S = source_corr - mu_s
    T = target_corr - mu_t
    H = S.T @ T
    U, _, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = mu_t - R @ mu_s

    return R, t

def icp(source,target,max_iters,tol,outlier_ratio,verbose=True):

    T = np.eye(4)
    R = T[:3, :3].copy()
    t = T[:3, 3].copy()

    source_transformed = transform(source.copy(), R, t)
    tree = KDTree(target)

    sample_idx = np.random.choice(len(source_transformed), min(500, len(source_transformed)), replace=False)
    nearest_dists, _ = tree.query(source_transformed[sample_idx],k=1)
    dist_threshold = 3.0*float(np.median(nearest_dists))

    rmse_history = []
    inlier_history = []
    prev_rmse = np.inf

    for iter in range(max_iters):
        dists,indices = tree.query(source_transformed, k=1)
        dists = dists.ravel()
        indices = indices.ravel()

        inlier_mask = dists < dist_threshold

        if inlier_mask.sum() > 10:
            valid_dists = dists[inlier_mask]
            threshold = np.percentile(valid_dists, 100*(1.0 - outlier_ratio))
            inlier_mask &= (dists<threshold)

        n_inliers = int(inlier_mask.sum())
        if inlier_mask.sum() < 6:
            if verbose:
                print(f"Iter{iter}: too few inliers ({n_inliers}), stopping")
            break

        src_corr = source_transformed[inlier_mask]
        tgt_corr = target[indices[inlier_mask]]
    
        rmse = compute_rmse(src_corr, tgt_corr)
        rmse_history.append(rmse)
        inlier_history.append(n_inliers)

        if verbose:
            print(f"Iter{iter:3d}: RMSE={rmse:.6f},"f"inliers={n_inliers}/{len(source_transformed)}")

        if abs(prev_rmse - rmse) < tol:
            R_iter, t_iter = svd_rotation(src_corr, tgt_corr)
            source_transformed = transform(source_transformed, R_iter, t_iter)
            R = R_iter @ R
            t = R_iter @ t + t_iter
            break

        prev_rmse = rmse
        R_iter, t_iter = svd_rotation(src_corr, tgt_corr)
        source_transformed = transform(source_transformed, R_iter, t_iter)
        R = R_iter @ R
        t = R_iter @ t + t_iter

    else:
        if verbose:
            print(f"Reached max iterations.")

    T_est = np.eye(4)
    T_est[:3, :3] = R
    T_est[:3, 3] = t

    final_rmse = rmse_history[-1] if rmse_history else np.inf
    converged = (len(rmse_history) >= 2 and abs(rmse_history[-1] - rmse_history[-2]) < tol)

    if verbose:
        print(f"Final RMSE={final_rmse:.6f}")
        print(f"Estimated transformation:\n{T_est}")

    return {
        "transformation":      T_est,
        "rmse_history":   rmse_history,
        "n_iterations":   len(rmse_history),
        "converged":      converged,
        "final_rmse":     final_rmse,
        "inlier_counts":  inlier_history,
        "aligned_source": source_transformed,
    }