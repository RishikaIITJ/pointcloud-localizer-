import numpy as np

from pointcloud_localizer.loader import load_pc


def rotation_matrix(rx, ry, rz):
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    return Rz @ Ry @ Rx

def transformation_matrix(rx, ry, rz, tx, ty, tz):
    T = np.eye(4)
    T[:3, :3] = rotation_matrix(rx, ry, rz)
    T[:3, 3] = [tx, ty, tz]
    return T

def transform(points, T):
    N = points.shape[0]
    homo_pts = np.hstack([points, np.ones((N, 1))])
    return (T @ homo_pts.T).T[:, :3]

def create_pair(filepath,n_pts,sigma,rx,ry,rz,tx,ty,tz,seed=42):

    if seed is not None:
        np.random.seed(seed)

    all_pts = load_pc(filepath)

    if len(all_pts) > n_pts:
        idx = np.random.choice(len(all_pts), n_pts, replace=False)
        base_pts = all_pts[idx]
    else:
        base_pts = all_pts

    gndTruth = transformation_matrix(rx, ry, rz, tx, ty, tz)
    
   
    target_cloud = transform(base_pts, gndTruth)
    source_cloud = base_pts + np.random.normal(0,sigma,base_pts.shape)
    noisy_target = target_cloud + np.random.normal(0,sigma,base_pts.shape)

    print(f"Noise={sigma}")
    print(f"GT rotation = ({np.degrees(rx):.2f}, " f"{np.degrees(ry):.2f}, {np.degrees(rz):.2f}) deg")
    print(f"GT translation = ({tx}, {ty}, {tz}) m")

    return source_cloud, noisy_target, gndTruth