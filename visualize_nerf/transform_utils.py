import open3d as o3d
import numpy as np
import json
import numpy as np
import sys
sys.path.append('/home/zubairirshad/nerf_pl')
from visualize_nerf.utils import repeat_interleave


def get_3d_bbox(size, shift=0):
    """
    Args:
        size: [3] or scalar
        shift: [3] or scalar
    Returns:
        bbox_3d: [3, N]

    """
    bbox_3d = np.array([[+size[0] / 2, +size[1] / 2, +size[2] / 2],
                    [+size[0] / 2, +size[1] / 2, -size[2] / 2],
                    [-size[0] / 2, +size[1] / 2, +size[2] / 2],
                    [-size[0] / 2, +size[1] / 2, -size[2] / 2],
                    [+size[0] / 2, -size[1] / 2, +size[2] / 2],
                    [+size[0] / 2, -size[1] / 2, -size[2] / 2],
                    [-size[0] / 2, -size[1] / 2, +size[2] / 2],
                    [-size[0] / 2, -size[1] / 2, -size[2] / 2]]) + shift
    return bbox_3d

def convert_points_to_homopoints(points):
  """Project 3d points (3xN) to 4d homogenous points (4xN)"""
  assert len(points.shape) == 2
  assert points.shape[0] == 3
  points_4d = np.concatenate([
      points,
      np.ones((1, points.shape[1])),
  ], axis=0)
  assert points_4d.shape[1] == points.shape[1]
  assert points_4d.shape[0] == 4
  return points_4d

def project(K, p_3d):
    projections_2d = np.zeros((2, p_3d.shape[1]), dtype='float32')
    p_2d = np.dot(K, p_3d)
    projections_2d[0, :] = p_2d[0, :]/p_2d[2, :]
    projections_2d[1, :] = p_2d[1, :]/p_2d[2, :]
    return projections_2d

def projection(c_xyz, focal, c, NV):
    """Converts [x,y,z] in camera coordinates to image coordinates 
        for the given focal length focal and image center c.
    :param c_xyz: points in camera coordinates (SB*NV, NP, 3)
    :param focal: focal length (SB, 2)
    :c: image center (SB, 2)
    :output uv: pixel coordinates (SB, NV, NP, 2)
    """
    uv = -c_xyz[..., :2] / (c_xyz[..., 2:] + 1e-9)  # (SB*NV, NC, 2); NC: number of grid cells 
    # uv = c_xyz[..., :2] / (c_xyz[..., 2:] + 1e-9)  # (SB*NV, NC, 2); NC: number of grid cells 
    uv *= repeat_interleave(
                focal.unsqueeze(1), NV if focal.shape[0] > 1 else 1
            )
    uv += repeat_interleave(
                c.unsqueeze(1), NV if c.shape[0] > 1 else 1
            )
    return uv

def convert_homopoints_to_points(points_4d):
  """Project 4d homogenous points (4xN) to 3d points (3xN)"""
  assert len(points_4d.shape) == 2
  assert points_4d.shape[0] == 4
  points_3d = points_4d[:3, :] / points_4d[3:4, :]
  assert points_3d.shape[1] == points_3d.shape[1]
  assert points_3d.shape[0] == 3
  return points_3d