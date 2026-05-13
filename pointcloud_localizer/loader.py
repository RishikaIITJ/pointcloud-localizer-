#loader.py — Load point clouds from .ply or .pcd files.

import numpy as np
import open3d as o3d

def load_pc(filepath):
   
    pcd = o3d.io.read_point_cloud(filepath)
    points = np.asarray(pcd.points, dtype=np.float64)
    print(f"Loaded {points.shape[0]} points from '{filepath}'")
    return points

def save_pc(points,filepath):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    o3d.io.write_point_cloud(filepath, pcd)
    print(f"Saved {points.shape[0]} points to '{filepath}'")