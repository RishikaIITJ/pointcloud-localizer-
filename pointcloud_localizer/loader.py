import numpy as np
import open3d as o3d

def load_pc(filepath):
   cloud = o3d.io.read_point_cloud(filepath)
   points = np.asarray(cloud.points, dtype=np.float64)
   return points 

def save_pc(points,filepath):
    cloud_obj = o3d.geometry.PointCloud()
    cloud_obj.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    o3d.io.write_point_cloud(filepath,cloud_obj)