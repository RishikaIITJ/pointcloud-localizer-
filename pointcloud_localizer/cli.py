
# cli.py — Command-line interface for pointcloud-localizer.

import argparse
import numpy as np
import sys


def _parse_args():
    p = argparse.ArgumentParser(
        description="pointcloud-localizer: ICP-based 3-D scan registration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=["synthetic", "file", "sweep"],
                   default="synthetic",
                   help="Run mode: synthetic demo | real files | robustness sweep")

    # Synthetic options
    p.add_argument("--n-points", type=int, default=5000,
                   help="Number of source points (synthetic mode)")
    p.add_argument("--noise", type=float, default=0.005,
                   help="Gaussian noise σ (m) added to clouds (synthetic)")
    p.add_argument("--shape", choices=["composite", "sphere", "box", "cylinder"],
                   default="composite", help="Surface shape to sample")
    p.add_argument("--rx", type=float, default=0.1,
                   help="Ground-truth rotation about X (radians)")
    p.add_argument("--ry", type=float, default=0.15,
                   help="Ground-truth rotation about Y (radians)")
    p.add_argument("--rz", type=float, default=0.05,
                   help="Ground-truth rotation about Z (radians)")
    p.add_argument("--tx", type=float, default=0.1,
                   help="Ground-truth translation X (m)")
    p.add_argument("--ty", type=float, default=0.05,
                   help="Ground-truth translation Y (m)")
    p.add_argument("--tz", type=float, default=0.02,
                   help="Ground-truth translation Z (m)")

    # File options
    p.add_argument("--source", type=str, default=None,
                   help="Path to source .ply/.pcd (file mode)")
    p.add_argument("--target", type=str, default=None,
                   help="Path to target .ply/.pcd (file mode)")

    # ICP options
    p.add_argument("--voxel-size", type=float, default=0.05,
                   help="Voxel downsampling size (m). Set 0 to skip.")
    p.add_argument("--max-iter", type=int, default=150,
                   help="Max ICP iterations")
    p.add_argument("--tolerance", type=float, default=1e-6,
                   help="ICP convergence tolerance on ΔRMSE")
    p.add_argument("--outlier-ratio", type=float, default=0.1,
                   help="Fraction of worst correspondences to reject each iter")
    p.add_argument("--max-dist", type=float, default=None,
                   help="Max correspondence distance (m). Auto if not set.")

    # Output
    p.add_argument("--output-dir", type=str, default="output",
                   help="Directory for output plots")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-iteration ICP output")

    return p.parse_args()


def run_synthetic(args):
    from pointcloud_localizer.synthetic import generate_synthetic_pair
    from pointcloud_localizer.preprocess import voxel_downsample
    from pointcloud_localizer.icp import icp
    from pointcloud_localizer.evaluate import (
        evaluate_registration, plot_before_after, plot_rmse_curve
    )

    print("\n=== Synthetic mode ===")
    src, tgt, T_gt = generate_synthetic_pair(
        n_points=args.n_points,
        noise_sigma=args.noise,
        rx=args.rx, ry=args.ry, rz=args.rz,
        tx=args.tx, ty=args.ty, tz=args.tz,
        shape=args.shape,
        seed=42,
    )

    src_ds = voxel_downsample(src, args.voxel_size) if args.voxel_size > 0 else src
    tgt_ds = voxel_downsample(tgt, args.voxel_size) if args.voxel_size > 0 else tgt

    result = icp(
        src_ds, tgt_ds,
        max_iterations=args.max_iter,
        tolerance=args.tolerance,
        max_correspondence_dist=args.max_dist,
        outlier_ratio=args.outlier_ratio,
        verbose=not args.quiet,
    )

    metrics = evaluate_registration(
        result["transform"], T_gt,
        result["rmse_history"], result["n_iterations"]
    )

    odir = args.output_dir
    plot_before_after(src_ds, tgt_ds, result["aligned_source"],
                      output_path=f"{odir}/before_after.png")
    plot_rmse_curve(result["rmse_history"],
                    output_path=f"{odir}/rmse_curve.png")

    return metrics


def run_file(args):
    from pointcloud_localizer.loader import load_pc
    from pointcloud_localizer.preprocess import voxel
    from pointcloud_localizer.icp import icp
    from pointcloud_localizer.evaluate import plot_before_after, plot_rmse_curve

    if not args.source or not args.target:
        print("ERROR: --source and --target are required in file mode", file=sys.stderr)
        sys.exit(1)

    print("\n=== File mode ===")
    src = load_pc(args.source)
    tgt = load_pc(args.target)

    src_ds = voxel(src, args.voxel_size) if args.voxel_size > 0 else src
    tgt_ds = voxel(tgt, args.voxel_size) if args.voxel_size > 0 else tgt

    result = icp(
        src_ds, tgt_ds,
        max_iterations=args.max_iter,
        tolerance=args.tolerance,
        max_correspondence_dist=args.max_dist,
        outlier_ratio=args.outlier_ratio,
        verbose=not args.quiet,
    )

    odir = args.output_dir
    plot_before_after(src_ds, tgt_ds, result["aligned_source"],
                      output_path=f"{odir}/before_after.png")
    plot_rmse_curve(result["rmse_history"],
                    output_path=f"{odir}/rmse_curve.png")

    print(f"\nEstimated transform (source → target):\n{result['transform']}")
    print(f"Final RMSE: {result['final_rmse']:.6f} m")


def run_sweep(args):
    from pointcloud_localizer.evaluate import run_robustness_sweep
    print("\n=== Robustness sweep ===")
    run_robustness_sweep(output_dir=args.output_dir)


def main():
    args = _parse_args()
    
    if args.mode == "synthetic":
        run_synthetic(args)
    elif args.mode == "file":
        run_file(args)
    elif args.mode == "sweep":
        run_sweep(args)


if __name__ == "__main__":
    main()
