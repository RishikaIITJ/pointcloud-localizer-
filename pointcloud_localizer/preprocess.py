import numpy as np
import open3d as o3d
    
def voxel_downsample(points, voxel_size):
    
    voxel_indices = np.floor(points/voxel_size).astype(np.int64)
    min_voxel_coords = voxel_indices.min(axis=0)
    shifted_voxel_indices = voxel_indices - min_voxel_coords

    dims = shifted_voxel_indices.max(axis=0) + 1
    keys = shifted_voxel_indices[:, 0]*dims[1]*dims[2] + shifted_voxel_indices[:, 1]*dims[2] + shifted_voxel_indices[:, 2]

    order = np.argsort(keys)
    sorted_keys = keys[order]
    sorted_pts = points[order]

    boundaries = np.concatenate([[0],np.where(np.diff(sorted_keys))[0] + 1,[len(sorted_keys)]])

    centroids = np.empty((len(boundaries) - 1, 3), dtype=np.float64)
    for i in range(len(boundaries) - 1):
        centroids[i] = sorted_pts[boundaries[i]:boundaries[i+1]].mean(axis=0)

    print(f"Downsampled {len(points)} to {len(centroids)} pts")
    return centroids