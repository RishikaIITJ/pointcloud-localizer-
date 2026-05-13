# preprocess.py — Point cloud preprocessing.

import numpy as np
import open3d as o3d
  


def voxel(points, voxel_size) -> np.ndarray:
   
    voxel_indices = np.floor(points/voxel_size).astype(np.int64)

    mins = voxel_indices.min(axis=0)
    shifted = voxel_indices-mins                       
    dims = shifted.max(axis=0) + 1
    keys = (shifted[:, 0]*dims[1]*dims[2] + shifted[:, 1]*dims[2] + shifted[:, 2])

    order = np.argsort(keys)
    sorted_keys = keys[order]
    sorted_pts = points[order]

    boundaries = np.concatenate([[0], np.where(np.diff(sorted_keys))[0] + 1, [len(sorted_keys)]])

    centroids = np.empty((len(boundaries)-1, 3), dtype=np.float64)
    for i in range(len(boundaries) - 1):
        centroids[i] = sorted_pts[boundaries[i]:boundaries[i + 1]].mean(axis=0)

    print(f"Voxel downsample: {len(points)} → {len(centroids)} pts "
          f"(voxel_size={voxel_size:.4f} m)")
    return centroids


def estimate_normals(points):
   
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
   
    pcd.orient_normals_towards_camera_location(np.array([0.0, 0.0, 0.0]))
    normals = np.asarray(pcd.normals, dtype=np.float64)
    print(f"Estimated normals for {len(normals)} points")
    return normals

def preprocess(points, voxel_size= 0.05, compute_normals= False,radius= 0.1):
   
    ds = voxel(points, voxel_size)
    normals = None
    if compute_normals:
        normals = estimate_normals(ds)
    return {"points": ds, "normals": normals}
