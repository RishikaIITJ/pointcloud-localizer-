import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from pointcloud_localizer.synthetic import create_pair
from pointcloud_localizer.preprocess import voxel_downsample
from pointcloud_localizer.icp import icp
from pointcloud_localizer.evaluate import rotation_error, translation_error


mesh = "reconstruction/bun_zipper_res2.ply"


def run_icp(rx, ry, rz, tx, ty, tz,seed=42):

    src, tgt, T_gt = create_pair(mesh,n_pts=6000,sigma=0.0,rx=rx, ry=ry, rz=rz,tx=tx, ty=ty, tz=tz,seed=seed)

    src_ds = voxel_downsample(src, 0.003)
    tgt_ds = voxel_downsample(tgt, 0.003)

    result = icp(src_ds, tgt_ds,max_iters=150,tol=1e-7,outlier_ratio=0.05,verbose=False)

    rot_err   = rotation_error(result["transformation"], T_gt)
    trans_err = translation_error(result["transformation"], T_gt)

    return rot_err, trans_err

def test_rotation_error():
    rot_err, _ = run_icp(rx=0.1,ry=0.1, rz=0.05,tx=0.05, ty=0.03, tz=0.02)
    assert rot_err < 1.0


def test_translation_error():
    _, trans_err = run_icp(rx=0.1, ry=0.1, rz=0.05,tx=0.05, ty=0.03, tz=0.02)
    assert trans_err < 0.01

if __name__ == "__main__":
    pytest.main([__file__, "-v"])