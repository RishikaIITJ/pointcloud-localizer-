import numpy as np
import os
import sys

from pointcloud_localizer.synthetic import create_pair, transformation_matrix, transform
from pointcloud_localizer.preprocess import voxel_downsample
from pointcloud_localizer.icp import icp
from pointcloud_localizer.loader import save_pc
from pointcloud_localizer.evaluate import evaluate_registration, rotation_error, translation_error, before_after, rmse_curve, sweep

def run_register(mesh_path, n_pts, sigma, rx, ry, rz,tx, ty, tz,voxel_size, max_iters,tol, outlier_ratio, output_dir):

    src, tgt, T_gt = create_pair(mesh_path, n_pts, sigma,rx, ry, rz, tx, ty, tz)

    src_ds = voxel_downsample(src, voxel_size)
    tgt_ds = voxel_downsample(tgt, voxel_size)

    result = icp(src_ds, tgt_ds, max_iters=max_iters,tol=tol, outlier_ratio=outlier_ratio,verbose=True)

    evaluate_registration(result["transformation"], T_gt, result["rmse_history"], result["n_iterations"])

    os.makedirs(output_dir, exist_ok=True)

    before_after(src_ds, tgt_ds, result["aligned_source"],output_path=os.path.join(output_dir, "before_after.png"))

    rmse_curve(result["rmse_history"],output_path=os.path.join(output_dir, "rmse_curve.png"))
   
    save_pc(result["aligned_source"],os.path.join(output_dir,"aligned_source.ply"))


def run_sweep(mesh_path, n_pts, rx, ry, rz, tx, ty, tz, voxel_size, max_iters, tol, outlier_ratio, output_dir):

   
    sigmas = [0.0, 0.005, 0.02]
    misalignments = [5, 15, 30]

    rot_errors   = np.zeros((3, 3))
    trans_errors = np.zeros((3, 3))
    rmse_final   = np.zeros((3, 3))
    n_iters      = np.zeros((3, 3))

    for i, sigma in enumerate(sigmas):
        for j, mis in enumerate(misalignments):

            src, tgt, T_gt = create_pair(mesh_path, n_pts, sigma, rx, ry, rz,tx, ty, tz)

            src_ds = voxel_downsample(src, voxel_size)
            tgt_ds = voxel_downsample(tgt, voxel_size)

            mis_rad = np.radians(mis)
            T_mis = transformation_matrix(mis_rad, mis_rad*0.5, mis_rad*0.3,tx+0.02,ty+0.01,tz+0.01)
            src_mis = transform(src_ds, T_mis)

            result = icp(src_mis,tgt_ds,max_iters=max_iters,tol=tol, outlier_ratio=outlier_ratio,verbose=False)

            T_combined = result["transformation"] @ T_mis

            rot_errors[i, j]   = rotation_error(T_combined, T_gt)
            trans_errors[i, j] = translation_error(T_combined, T_gt)
            rmse_final[i, j]   = result["final_rmse"]
            n_iters[i, j]      = result["n_iterations"]

    sweep(rot_errors,trans_errors,rmse_final,n_iters,sigmas,misalignments,output_path=os.path.join(output_dir, "robustness.png"))

if __name__ == "__main__":

    mode = "sweep"  # "register" or "sweep"

    mesh = "reconstruction/bun_zipper_res2.ply"
    n_pts = 5000
    sigma = 0.001
    rx, ry, rz = 0.1, 0.1, 0.05
    tx, ty, tz = 0.05, 0.03, 0.02
    voxel_size = 0.003
    max_iters = 150
    tol = 1e-6
    outlier_ratio = 0.1
    output_dir = "output"

    if mode == "register":
        run_register(mesh, n_pts, sigma,rx, ry, rz, tx, ty, tz,voxel_size, max_iters, tol,outlier_ratio, output_dir)

    elif mode == "sweep":
        run_sweep(mesh, n_pts, rx, ry, rz, tx, ty, tz, voxel_size, max_iters, tol, outlier_ratio, output_dir)

    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)