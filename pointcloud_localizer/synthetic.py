# synthetic.py — Synthetic point cloud generation for testing and benchmarking.

from typing import Optional, Tuple

import numpy as np

def sphere(n, radius=1.0):

    phi = np.random.uniform(0, 2*np.pi, n)
    costheta = np.random.uniform(-1, 1, n)
    theta = np.arccos(costheta)
    x = radius* np.sin(theta)* np.cos(phi)
    y = radius* np.sin(theta)* np.sin(phi)
    z = radius* np.cos(theta)
    return np.stack([x, y, z], axis=1)


def box(n, size=(2.0,1.0,1.5)):
   
    lx, ly, lz = size
   
    area = np.array([ly*lz, ly*lz, lx*lz, lx*lz, lx*ly, lx*ly])
    prob = area/area.sum()
    face_counts = np.random.multinomial(n, prob)
    pts = []
  
    for sign in [+1, -1]:
        k = face_counts[0 if sign == 1 else 1]
        u = np.random.uniform(-ly/2, ly/2, k)
        v = np.random.uniform(-lz/2, lz/2, k)
        x = np.full(k, sign*lx/2)
        pts.append(np.stack([x, u, v], axis=1))
   
   
    for sign in [+1, -1]:
        k = face_counts[2 if sign == 1 else 3]
        u = np.random.uniform(-lx/2, lx/2, k)
        v = np.random.uniform(-lz/2, lz/2, k)
        y = np.full(k, sign*ly/2)
        pts.append(np.stack([u, y, v], axis=1))
    
    for sign in [+1, -1]:
        k = face_counts[4 if sign == 1 else 5]
        u = np.random.uniform(-lx/2, lx/2, k)
        v = np.random.uniform(-ly/2, ly/2, k)
        z = np.full(k, sign*lz/2)
        pts.append(np.stack([u, v, z], axis=1))
    return np.vstack(pts)


def cylinder(n, radius=0.5, height=2.0):
    
    curved_area = 2* np.pi *radius *height
    plate_area = np.pi* radius **2
    total = curved_area + 2*plate_area
    n_side = int(n*curved_area /total)
    n_cap = (n - n_side)//2
    n_cap2 = n - n_side - n_cap

    theta = np.random.uniform(0, 2*np.pi, n_side)
    z_s = np.random.uniform(-height/2, height/2, n_side)
    side = np.stack([radius*np.cos(theta), radius*np.sin(theta), z_s], axis=1)

    def cap(k, z_val):
        r = radius*np.sqrt(np.random.uniform(0, 1, k))
        t = np.random.uniform(0, 2*np.pi, k)
        return np.stack([r*np.cos(t), r*np.sin(t), np.full(k, z_val)], axis=1)

    return np.vstack([side, cap(n_cap, height/2), cap(n_cap2, -height/2)])


def composite(n):
   
    n3 = n // 3
    sphere = sphere(n3, radius=0.4)
    sphere[:, 0] += 1.5                         

    box =box(n-2*n3, size=(1.0, 0.6, 0.8))

    cyl = cylinder(n3, radius=0.25, height=1.2)
    cyl[:, 0] -= 1.5                           

    return np.vstack([sphere, box, cyl])


def rotation_matrix_from_euler(rx,ry,rz):
   
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([
        [1, 0, 0],
        [0, cx, -sx],
        [0, sx, cx]
    ])

    Ry = np.array([
        [cy, 0, sy],
        [0, 1, 0],
        [-sy, 0, cy]
    ])

    Rz = np.array([
        [cz, -sz, 0],
        [sz, cz, 0],
        [0, 0, 1]
    ])

    return Rz @ Ry @ Rx


def make_transform(rx, ry, rz, tx, ty, tz):
  
    T = np.eye(4)
    T[:3, :3] = rotation_matrix_from_euler(rx, ry, rz)
    T[:3, 3] = [tx, ty, tz]
    return T


def apply_transform(points, T):

    N = points.shape[0]
    pts_h = np.hstack([points, np.ones((N, 1))]) 
    return (T @ pts_h.T).T[:, :3]


def generate_synthetic_pair(
    n_points = 5000,
    noise_sigma= 0.005,
    rx = 0.1,
    ry = 0.15,
    rz = 0.05,
    tx = 0.1,
    ty = 0.05,
    tz = 0.02,
    shape= "composite",
    seed= 42 ):
    
    if seed is not None:
        np.random.seed(seed)

    samplers = {
        "sphere": lambda:sphere(n_points),
        "box": lambda: box(n_points),
        "cylinder": lambda: cylinder(n_points),
        "composite": lambda: composite(n_points),
    }
    
    base = samplers[shape]()
    T_gt = make_transform(rx, ry, rz, tx, ty, tz)

    source = base + np.random.normal(0, noise_sigma, base.shape)
    target_clean = apply_transform(base, T_gt)
    target = target_clean + np.random.normal(0, noise_sigma, target_clean.shape)

    print(f"Generated '{shape}' cloud: {n_points} pts, "
          f"noise σ={noise_sigma:.4f} m")
    print(f"Ground-truth rotation (rx,ry,rz) = "
          f"({np.degrees(rx):.2f}°, {np.degrees(ry):.2f}°, {np.degrees(rz):.2f}°)")
    print(f"Ground-truth translation = "
          f"({tx:.4f}, {ty:.4f}, {tz:.4f}) m")

    return source, target, T_gt
