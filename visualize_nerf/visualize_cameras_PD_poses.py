from curses import resize_term
import open3d as o3d
import json
import numpy as np
import json

import torch
from kornia import create_meshgrid
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.visualizer as pv
# from datasets.google_scanned_utils import *
from PIL import Image
# from datasets.ray_utils import homogenise_np
# from datasets.ray_utils import bbox_intersection_batch
import torchvision.transforms as T
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Rotation as R

def get_world_grid(side_lengths, grid_size):
    """ Returns a 3D grid of points in world coordinates.
    :param side_lengths: (min, max) for each axis (3, 2)
    :param grid_size: number of points along each dimension () or (3)
    :output grid: (1, grid_size**3, 3)
    """
    if len(grid_size) == 1:
        grid_size = [grid_size[0] for _ in range(3)]
        
    w_x = torch.linspace(side_lengths[0][0], side_lengths[0][1], grid_size[0])
    w_y = torch.linspace(side_lengths[1][0], side_lengths[1][1], grid_size[1])
    w_z = torch.linspace(side_lengths[2][0], side_lengths[2][1], grid_size[2])
    # Z, Y, X = torch.meshgrid(w_x, w_y, w_z)
    X, Y, Z = torch.meshgrid(w_x, w_y, w_z)
    w_xyz = torch.stack([X, Y, Z], axis=-1) # (gs, gs, gs, 3), gs = grid_size
#     w_xyz = torch.stack(torch.meshgrid(w_x, w_y, w_z), axis=-1) # (gs, gs, gs, 3), gs = grid_size
    print(w_xyz.shape)
    w_xyz = w_xyz.reshape(-1, 3).unsqueeze(0) # (1, grid_size**3, 3)
    return w_xyz

def repeat_interleave(input, repeats, dim=0):
    """
    Repeat interleave along axis 0
    torch.repeat_interleave is currently very slow
    https://github.com/pytorch/pytorch/issues/31980
    """
    output = input.unsqueeze(1).expand(-1, repeats, *input.shape[1:])
    return output.reshape(-1, *input.shape[1:])

def world2camera(w_xyz, cam2world, NS=None):
    """Converts the points in world coordinates to camera view.
    :param xyz: points in world coordinates (SB*NV, NC, 3)
    :param poses: camera matrix (SB*NV, 4, 4)
    :output points in camera coordinates (SB*NV, NC, 3)
    : SB batch size
    : NV number of views in each scene
    : NC number of coordinate points
    """
    #print(w_xyz.shape, cam2world.shape)
    if NS is not None:
        w_xyz = repeat_interleave(w_xyz, NS)  # (SB*NS, B, 3)
    rot = cam2world[:, :3, :3].transpose(1, 2)  # (B, 3, 3)
    trans = -torch.bmm(rot, cam2world[:, :3, 3:])  # (B, 3, 1)
    #print(rot.shape, w_xyz.shape)
    cam_rot = torch.matmul(rot[:, None, :3, :3], w_xyz.unsqueeze(-1))[..., 0]
    cam_xyz = cam_rot + trans[:, None, :, 0]
    # cam_xyz = cam_xyz.reshape(-1, 3)  # (SB*B, 3)
    return cam_xyz

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


def convert_homopoints_to_points(points_4d):
  """Project 4d homogenous points (4xN) to 3d points (3xN)"""
  assert len(points_4d.shape) == 2
  assert points_4d.shape[0] == 4
  points_3d = points_4d[:3, :] / points_4d[3:4, :]
  assert points_3d.shape[1] == points_3d.shape[1]
  assert points_3d.shape[0] == 3
  return points_3d

def get_rays_in_bbox(rays_o, rays_d, bbox_bounds, axis_aligned_mat):

    rays_o_bbox, rays_d_bbox = transform_rays_to_bbox_coordinates(
        rays_o, rays_d, axis_aligned_mat
    )

    # ids = np.random.choice(rays_o_bbox.shape[0], int(rays_o_bbox.shape[0]*0.5))
    # rays_o = rays_o_bbox[ids,:]
    # rays_d = rays_d_bbox[ids,:]

    # sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=5)
    # sphere = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
    # near = 0.2
    # far = 15.0
    # fig = pv.figure()
    # for j in range(2500):
    #     start = rays_o[j,:]
    #     end = rays_o[j,:] + rays_d[j,:]*near
    #     line = np.concatenate((start[None, :],end[None, :]), axis=0)
    #     fig.plot(line, c=(1.0, 0.5, 0.0))

    #     start = rays_o[j,:] + rays_d[j,:]*near
    #     end = rays_o[j,:] + rays_d[j,:]*far
    #     line = np.concatenate((start[None, :],end[None, :]), axis=0)
    #     fig.plot(line, c=(0.0, 1.0, 0.0))

    bbox_dimension =  np.array([(bbox_bounds[1,0]-bbox_bounds[0,0]), (bbox_bounds[1,1]-bbox_bounds[0,1]), (bbox_bounds[1,2]-bbox_bounds[0,2])])
    # box_canonical = o3d.geometry.OrientedBoundingBox(center = [0,0,0], R = np.eye(3), extent=bbox_dimension)
    # things_to_draw = []
    # things_to_draw.append(sphere)
    # things_to_draw.append(box_canonical)
    # for geometry in things_to_draw:
    #     fig.add_geometry(geometry)    
    # fig.show()
    
    # bbox_bounds = np.array([-bbox_dimension / 2, bbox_dimension / 2])
    bbox_mask, batch_near, batch_far = bbox_intersection_batch(
        bbox_bounds, rays_o_bbox, rays_d_bbox
    )
    bbox_mask, batch_near, batch_far = (
        torch.Tensor(bbox_mask).bool(),
        torch.Tensor(batch_near[..., None]),
        torch.Tensor(batch_far[..., None]),
    )
    return bbox_mask, batch_near, batch_far

def get_object_rays_in_bbox(rays_o, rays_d, RTs, canonical=False):
    if not canonical:
        instance_rotation = RTs['R']
        instance_translation = RTs['T']
        # scale = RTs['s']
        box = RTs['s']
        box_transformation = np.eye(4)
        box_transformation[:3, :3] = np.reshape(instance_rotation, (3, 3))
        box_transformation[:3, -1] = instance_translation
        axis_aligned_mat = np.linalg.inv(box_transformation)
        # axis_aligned_mat = box_transformation
    else:
        scale = RTs['s']
        axis_aligned_mat = None

    # scale = np.max([(box[1,0]-box[0,0]), (box[1,1]-box[0,1]), (box[1,2]-box[0,2])])
    # bbox_bounds = np.array([-scale / 2, scale / 2])
    bbox_bounds = box
    # print("scale", scale)
    bbox_mask, batch_near_obj, batch_far_obj = get_rays_in_bbox(rays_o, rays_d, bbox_bounds, axis_aligned_mat)
    
    return bbox_mask, batch_near_obj, batch_far_obj

def get_ray_directions(H, W, focal):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W, focal: image height, width and focal length

    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    directions = \
        torch.stack([(i-W/2)/focal, -(j-H/2)/focal, -torch.ones_like(i)], -1) # (H, W, 3)
    return directions


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


def draw_bboxes_mpl_glow(img, img_pts, color):
    img_pts = np.int32(img_pts).reshape(-1, 2)
    color_ground = color
    n_lines = 8
    diff_linewidth = 1.05
    alpha_value = 0.03
    for i, j in zip([4, 5, 6, 7], [5, 7, 4, 6]):
        for n in range(1, n_lines+1):
            plt.plot([img_pts[i][0], img_pts[j][0]], [img_pts[i][1], img_pts[j][1]], color=color_ground, linewidth=1, marker='o')
            plt.plot([img_pts[i][0], img_pts[j][0]], [img_pts[i][1], img_pts[j][1]],
                marker='o',
                linewidth=1+(diff_linewidth*n),
                alpha=alpha_value,
                color=color_ground)
    for i, j in zip(range(4), range(4, 8)):
        for n in range(1, n_lines+1):
            plt.plot([img_pts[i][0], img_pts[j][0]], [img_pts[i][1], img_pts[j][1]], color=color_ground, linewidth=1, marker='o')
            plt.plot([img_pts[i][0], img_pts[j][0]], [img_pts[i][1], img_pts[j][1]],
                marker='o',
                linewidth=1+(diff_linewidth*n),
                alpha=alpha_value,
                color=color_ground)
    for i, j in zip([0, 1, 2, 3], [1, 3, 0, 2]):
        for n in range(1, n_lines+1):
            plt.plot([img_pts[i][0], img_pts[j][0]], [img_pts[i][1], img_pts[j][1]], color=color_ground, linewidth=1, marker='o')
            plt.plot([img_pts[i][0], img_pts[j][0]], [img_pts[i][1], img_pts[j][1]],
                marker='o',
                linewidth=1+(diff_linewidth*n),
                alpha=alpha_value,
                color=color_ground)
    return img

def get_rays(directions, c2w, output_view_dirs = False, output_radii = False):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate

    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:, :3].T # (H, W, 3)
    #rays_d /= torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, 3].expand(rays_d.shape) # (H, W, 3)

    if output_radii:
        rays_d_orig = directions @ c2w[:, :3].T
        dx = torch.sqrt(torch.sum((rays_d_orig[:-1, :, :] - rays_d_orig[1:, :, :]) ** 2, dim=-1))
        dx = torch.cat([dx, dx[-2:-1, :]], dim=0)
        radius = dx[..., None] * 2 / torch.sqrt(torch.tensor(12, dtype=torch.int8))
        radius = radius.reshape(-1)
    
    if output_view_dirs:
        viewdirs = rays_d
        viewdirs /= torch.norm(viewdirs, dim=-1, keepdim=True)
        rays_d = rays_d.view(-1, 3)
        rays_o = rays_o.view(-1, 3)
        viewdirs = viewdirs.view(-1, 3)
        if output_radii:
            return rays_o, viewdirs, rays_d, radius
        else:
            return rays_o, viewdirs, rays_d  
    else:
        rays_d /= torch.norm(rays_d, dim=-1, keepdim=True)
        rays_d = rays_d.view(-1, 3)
        rays_o = rays_o.view(-1, 3)
        return rays_o, rays_d
        
# def get_rays(directions, c2w):
#     """
#     Get ray origin and normalized directions in world coordinate for all pixels in one image.
#     Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
#                ray-tracing-generating-camera-rays/standard-coordinate-systems

#     Inputs:
#         directions: (H, W, 3) precomputed ray directions in camera coordinate
#         c2w: (3, 4) transformation matrix from camera coordinate to world coordinate

#     Outputs:
#         rays_o: (H*W, 3), the origin of the rays in world coordinate
#         rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
#     """
#     # Rotate ray directions from camera coordinate to the world coordinate
#     rays_d = directions @ c2w[:, :3].T # (H, W, 3)
#     rays_d /= torch.norm(rays_d, dim=-1, keepdim=True)
#     # The origin of all rays is the camera origin in world coordinate
#     rays_o = c2w[:, 3].expand(rays_d.shape) # (H, W, 3)

#     rays_d = rays_d.view(-1, 3)
#     rays_o = rays_o.view(-1, 3)

#     return rays_o, rays_d

import math

def normalize(vec):
    return vec / (np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-9)

def look_at(cam_location, point):
    # Cam points in positive z direction
    forward = point - cam_location
    forward = normalize(forward)

    tmp = np.array([0., -1., 0.])

    right = np.cross(tmp, forward)
    right = normalize(right)

    up = np.cross(forward, right)
    up = normalize(up)

    mat = np.stack((right, up, forward, cam_location), axis=-1)

    hom_vec = np.array([[0., 0., 0., 1.]])

    if len(mat.shape) > 2:
        hom_vec = np.tile(hom_vec, [mat.shape[0], 1, 1])

    mat = np.concatenate((mat, hom_vec), axis=-2)
    return mat, forward

def get_archimedean_spiral(sphere_radius, num_steps=100):
    '''
    https://en.wikipedia.org/wiki/Spiral, section "Spherical spiral". c = a / pi
    '''
    a = 40
    r = sphere_radius

    translations = []

    i = a / 2
    while i < a:
        theta = i / a * math.pi
        x = r * math.sin(theta) * math.cos(-i)
        y = r * math.sin(-theta + math.pi) * math.sin(-i)
        z = r * - math.cos(theta)

        translations.append((x, y, z))
        i += a / (2 * num_steps)

    return np.array(translations)

def get_camera_frustum(img_size, focal, C2W, frustum_length=1, color=[0., 1., 0.]):
    W, H = img_size
    hfov = np.rad2deg(np.arctan(W / 2. / focal) * 2.)
    vfov = np.rad2deg(np.arctan(H / 2. / focal) * 2.)
    # print("hfov", hfov, vfov)
    half_w = frustum_length * np.tan(np.deg2rad(hfov / 2.))
    half_h = frustum_length * np.tan(np.deg2rad(vfov / 2.))

    # build view frustum for camera (I, 0)
    frustum_points = np.array([[0., 0., 0.],                          # frustum origin
                               [-half_w, -half_h, frustum_length],    # top-left image corner
                               [half_w, -half_h, frustum_length],     # top-right image corner
                               [half_w, half_h, frustum_length],      # bottom-right image corner
                               [-half_w, half_h, frustum_length]])    # bottom-left image corner
    frustum_lines = np.array([[0, i] for i in range(1, 5)] + [[i, (i+1)] for i in range(1, 4)] + [[4, 1]])
    frustum_colors = np.tile(np.array(color).reshape((1, 3)), (frustum_lines.shape[0], 1))
    #print("frustum_points", frustum_points)
    # frustum_colors = np.vstack((np.tile(np.array([[1., 0., 0.]]), (4, 1)),
    #                            np.tile(np.array([[0., 1., 0.]]), (4, 1))))

    # transform view frustum from (I, 0) to (R, t)
    # C2W = np.linalg.inv(W2C)
    frustum_points = np.dot(np.hstack((frustum_points, np.ones_like(frustum_points[:, 0:1]))), C2W.T)
    frustum_points = frustum_points[:, :3] / frustum_points[:, 3:4]

    #print("frustum_points afters", frustum_points)
    return frustum_points, frustum_lines, frustum_colors


def frustums2lineset(frustums):
    N = len(frustums)
    merged_points = np.zeros((N*5, 3))      # 5 vertices per frustum
    merged_lines = np.zeros((N*8, 2))       # 8 lines per frustum
    merged_colors = np.zeros((N*8, 3))      # each line gets a color

    for i, (frustum_points, frustum_lines, frustum_colors) in enumerate(frustums):
        merged_points[i*5:(i+1)*5, :] = frustum_points
        merged_lines[i*8:(i+1)*8, :] = frustum_lines + i*5
        merged_colors[i*8:(i+1)*8, :] = frustum_colors

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(merged_points)
    lineset.lines = o3d.utility.Vector2iVector(merged_lines)
    lineset.colors = o3d.utility.Vector3dVector(merged_colors)

    return lineset

def convert_pose(C2W):
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = np.matmul(C2W, flip_yz)
    return C2W

def convert_pose_spiral(C2W):
    convert_mat = np.zeros((4,4))
    convert_mat[0,1] = 1
    convert_mat[1, 0] = 1
    convert_mat[2, 2] = -1
    convert_mat[3,3] = 1
    C2W = np.matmul(C2W, convert_mat)
    return C2W

def intersect_sphere(rays_o, rays_d):
    """Compute the depth of the intersection point between this ray and unit sphere.
    Args:
        rays_o: [num_rays, 3]. Ray origins.
        rays_d: [num_rays, 3]. Ray directions.
    Returns:
        depth: [num_rays, 1]. Depth of the intersection point.
    """
    # note: d1 becomes negative if this mid point is behind camera

    d1 = -torch.sum(rays_d * rays_o, dim=-1, keepdim=True) / torch.sum(
        rays_d**2, dim=-1, keepdim=True
    )
    p = rays_o + d1 * rays_d
    # consider the case where the ray does not intersect the sphere
    rays_d_cos = 1.0 / torch.norm(rays_d, dim=-1, keepdim=True)
    p_norm_sq = torch.sum(p * p, dim=-1, keepdim=True)
    d2 = torch.sqrt(1.0 - p_norm_sq) * rays_d_cos

    return d1 + d2

def cast_rays(t_vals, origins, directions):
    return origins[..., None, :] + t_vals[..., None] * directions[..., None, :]

def sample_along_rays(
    rays_o,
    rays_d,
    num_samples,
    near,
    far,
    randomized,
    lindisp,
    in_sphere,
):
    bsz = rays_o.shape[0]
    t_vals = torch.linspace(0.0, 1.0, num_samples + 1, device=rays_o.device)

    if in_sphere:
        if lindisp:
            t_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)
        else:
            t_vals = near * (1.0 - t_vals) + far * t_vals
    else:
        t_vals = torch.broadcast_to(t_vals, (bsz, num_samples + 1))

    if randomized:
        mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
        upper = torch.cat([mids, t_vals[..., -1:]], -1)
        lower = torch.cat([t_vals[..., :1], mids], -1)
        t_rand = torch.rand((bsz, num_samples + 1), device=rays_o.device)
        t_vals = lower + (upper - lower) * t_rand
    else:
        t_vals = torch.broadcast_to(t_vals, (bsz, num_samples + 1))

    if in_sphere:
        coords = cast_rays(t_vals, rays_o, rays_d)
    else:
        t_vals = torch.flip(
            t_vals,
            dims=[
                -1,
            ],
        )  # 1.0 -> 0.0
        coords = depth2pts_outside(rays_o, rays_d, t_vals)

    return t_vals, coords

import colorsys
def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    # random.shuffle(colors)
    return colors

def sample_rays_in_bbox(RTs, rays_o, view_dirs):
    all_R = RTs['R']
    all_T = RTs['T']
    all_s = RTs['s']
    for Rot,Tran,sca in zip(all_R, all_T, all_s):
        RTS_single = {'R': np.array(Rot), 'T': np.array(Tran), 's': np.array(sca)}
        _, near, far = get_object_rays_in_bbox(rays_o, view_dirs, RTS_single, canonical=False)

        new_near = torch.where((all_near==0) | (near==0), torch.maximum(near, all_near), torch.minimum(near, all_near))
        all_near = new_near
        new_far = torch.where((all_far==0) | (far==0), torch.maximum(far, all_far), torch.minimum(far, all_far))
        all_far = new_far
    bbox_mask = (all_near !=0) & (all_far!=0)
    return all_near, all_far, bbox_mask

def plot_rays(c2w, focal, W, H, fig, RTs = None, asset_pose_inv= None, box=None):
    focal = focal
    directions = get_ray_directions(H, W, focal) # (h, w, 3)
    c2w = torch.FloatTensor(c2w)[:3, :4]
    # rays_o, rays_d = get_rays(directions, c2w)
    rays_o, view_dirs, rays_d, radii = get_rays(directions, c2w, output_view_dirs=True, output_radii=True)

    all_rays_boxes = []
    all_viewdirs_boxes = []
    # all_near = []
    # all_far = []
    all_near = torch.zeros((rays_o.shape[0], 1))
    all_far = torch.zeros((rays_o.shape[0], 1))
    all_far = torch.zeros((rays_o.shape[0], 1))
    if RTs is not None:
        rays_o = rays_o.numpy()
        rays_d = rays_d.numpy()
        view_dirs = view_dirs.numpy()
        if isinstance(RTs, dict):
            all_R = RTs['R']
            all_T = RTs['T']
            all_s = RTs['s']
            for i, (Rot,Tran,sca) in enumerate(zip(all_R, all_T, all_s)):
                # if i >0:
                #     continue
                RTS_single = {'R': np.array(Rot), 'T': np.array(Tran), 's': np.array(sca)}
                bbox_mask, near, far = get_object_rays_in_bbox(rays_o, view_dirs, RTS_single, canonical=False)

                new_near = torch.where((all_near==0) | (near==0), torch.maximum(near, all_near), torch.minimum(near, all_near))
                all_near = new_near
                new_far = torch.where((all_far==0) | (far==0), torch.maximum(far, all_far), torch.minimum(far, all_far))
                all_far = new_far

        
        else:
            rays_o, rays_d, near, far = get_object_rays_in_bbox(rays_o, rays_d, RTs, canonical=True)


    if asset_pose_inv is not None:
        axis_aligned_mat = torch.FloatTensor(asset_pose_inv)
        rays_o, rays_d, viewdirs, scale_factor = transform_rays_to_bbox_coordinates(rays_o, rays_d, axis_aligned_mat, box)

    # rays_o = rays_o[bbox_mask, :]
    # rays_d = rays_d[bbox_mask, :]
    ids = np.random.choice(rays_o.shape[0], int(rays_o.shape[0]*1.0))

    bbox_mask = (all_near !=0) & (all_far!=0)

    
    # print("all_near", all_near, all_far)
    # img = Image.open('/home/zubairirshad/pd-api-py/PDMultiObjv2/single_street/SF_GrantAndCalifornia65/val/rgb')
    # # plt.imshow(np.array(img))
    # # plt.show()
    # print(np.array(img).shape)
    # H,W,_ = np.array(img).shape
    # img = (np.array(img)/255).reshape(-1,3)

    # img_masked = np.zeros(img.shape)
    # bbox_mask_img = bbox_mask.view(-1, 1).repeat(1, 3)
    # img_masked[bbox_mask_img] = img[bbox_mask_img]
    # plt.imshow(img_masked.reshape(H,W,3))
    # plt.show()

    # colors = random_colors(1)
    # bbox_mask_vis = bbox_mask.reshape(H,W)
    # # plt.imshow(bbox_mask)

    # colored_mask = np.zeros([bbox_mask_vis.shape[0], bbox_mask_vis.shape[1], 3])
    # colored_mask[bbox_mask_vis == True, :] = colors[0]
    # # plt.imshow(colored_mask, alpha=0.5)
    # # plt.show()
    # print(bbox_mask_vis.shape)

    # print("rays_o", rays_o.shape)
    rays_o = rays_o[ids, :]
    rays_d = rays_d[ids, :]
    # near = near[ids, :].squeeze(-1).numpy()
    # far = far[ids, :].squeeze(-1).numpy()

    near = all_near[ids, :].squeeze(-1).numpy()
    far = all_far[ids, :].squeeze(-1).numpy()

    # print("bbox_mask", bbox_mask)
    # near = np.ones_like(bbox_mask)*0.2
    # near[bbox_mask] = np.zeros_like(near[bbox_mask])

    # far = np.ones_like(bbox_mask)*3.0
    # far[bbox_mask] = np.zeros_like(far[bbox_mask])
    # print("rays_o, near", rays_o.shape, near.shape)

    # near = near[ids, :]
    # far = far[ids, :]


    # rays_o = rays_o[ids, :]
    # rays_d = rays_d[ids, :]

    far = intersect_sphere(torch.FloatTensor(rays_o), torch.FloatTensor(rays_d))
    print("far", far.shape, far)
    print("far", far)
    # invalid_mask = torch.isnan(far).squeeze()
    # print("invalid mask", invalid_mask.shape)
    # plt.imshow(invalid_mask.reshape(480,640).numpy())
    # plt.show()
    rays_o = torch.FloatTensor(rays_o)
    rays_d = torch.FloatTensor(rays_d)
    near = torch.full_like(rays_o[..., -1:], 1e-4)

    # rays_o = rays_o[~invalid_mask, :]
    # far = far[~invalid_mask, :]
    # near = near[~invalid_mask]
    # rays_d = rays_d[~invalid_mask]

    # scale = np.array([1,1,1])
    # bbox_bounds = np.array([-scale, scale])

    # rays_o = torch.FloatTensor(rays_o)
    # rays_d = torch.FloatTensor(rays_d)
    # bbox_bounds = torch.Tensor(bbox_bounds)
    # rays_o, rays_d, near, far = get_rays_in_bbox(rays_o, rays_d, bbox_bounds, None, None)

    # near = torch.FloatTensor(near)
    # far = torch.FloatTensor(far)

    # near = torch.full_like(rays_o[..., -1:], 2.0)
    # far = torch.full_like(rays_o[..., -1:], 6.0)

    # print("far", far.shape)

    # print("rays_o", rays_o.shape, rays_d.shape)


    # # t_vals, coords = sample_along_rays(rays_o, rays_d, 129, near, far, False, True, True )

    # # print("torch min coords max coords", torch.min(coords), torch.max(coords))

    # far = far.numpy()
    # rays_o = rays_o.numpy()
    # rays_d = rays_d.numpy()
    # coords = rays_o = 3.0* rays_d
    for j in range(500):
        start = rays_o[j,:]
        # end = rays_o[j,:] + rays_d[j,:]*near[j]
        start = rays_o[j,:]
        end = rays_o[j,:] + rays_d[j,:]*0.2
        line = np.concatenate((start[None, :],end[None, :]), axis=0)
        fig.plot(line, c=(1.0, 0.5, 0.0))

        # start = rays_o[j,:] + rays_d[j,:]*near[j]
        # end = rays_o[j,:] + rays_d[j,:]*far[j]

        start = rays_o[j,:] + rays_d[j,:]*0.2
        end = rays_o[j,:] + rays_d[j,:]*3.0
        line = np.concatenate((start[None, :],end[None, :]), axis=0)
        fig.plot(line, c=(0.0, 1.0, 0.0))


# def transform_rays_to_bbox_coordinates(rays_o, rays_d, axis_align_mat, box):
#     rays_o_bbox = rays_o
#     rays_d_bbox = rays_d
#     T_box_orig = axis_align_mat
#     rays_o_bbox = (T_box_orig[:3, :3] @ rays_o_bbox.T).T + T_box_orig[:3, 3]
#     rays_d_bbox = (T_box_orig[:3, :3] @ rays_d.T).T

#     scale_factor = np.max([(box[1,0]-box[0,0]), (box[1,1]-box[0,1]), (box[1,2]-box[0,2])])
#     rays_o_bbox/=scale_factor
#     viewdirs_bbox = rays_d_bbox
#     viewdirs_bbox /= torch.norm(viewdirs_bbox, dim=-1, keepdim=True)

#     return rays_o_bbox, rays_d_bbox, viewdirs_bbox, scale_factor

def transform_rays_to_bbox_coordinates(rays_o, rays_d, axis_align_mat):
    rays_o_bbox = rays_o
    rays_d_bbox = rays_d
    T_box_orig = axis_align_mat
    rays_o_bbox = (T_box_orig[:3, :3] @ rays_o_bbox.T).T + T_box_orig[:3, 3]
    rays_d_bbox = (T_box_orig[:3, :3] @ rays_d.T).T
    return rays_o_bbox, rays_d_bbox


def visualize_poses(all_c2w, all_c2w_test, fov, obj_location, 
                    sphere_radius=1, bbox_dimension= None, RTs=None,
                    W=None, asset_pose_inv=None, box=None):
   
    

    T = np.eye(4)
    T[:3,3] = obj_location
    things_to_draw = []
    focal = (640 / 2) / np.tan((fov / 2) / (180 / np.pi))
    frustums = []
    fig = pv.figure()
    # c2w_single = np.expand_dims(all_c2w[0], axis=0)
    c2w = all_c2w[0]
    # all_c2w = all_c2w[1:]
    for i, C2W in enumerate(all_c2w):
        img_size = (640, 480)
        if asset_pose_inv is not None:
            scale_factor = np.max([(box[1,0]-box[0,0]), (box[1,1]-box[0,1]), (box[1,2]-box[0,2])])
            C2W = asset_pose_inv @ C2W
            C2W[:3,3]/=scale_factor
            C2W[3,3] = 1
        if i==0:
            frustums.append(get_camera_frustum(img_size, focal, C2W, frustum_length=0.3, color=[1,0,0]))
        else:
            frustums.append(get_camera_frustum(img_size, focal, C2W, frustum_length=0.05, color=[1,0,0]))
    for C2W in all_c2w_test:
        if asset_pose_inv is not None:
            scale_factor = np.max([(box[1,0]-box[0,0]), (box[1,1]-box[0,1]), (box[1,2]-box[0,2])])
            C2W = asset_pose_inv @ C2W
            C2W[:3,3]/=scale_factor
            C2W[3,3] = 1
        frustums.append(get_camera_frustum(img_size, focal, C2W, frustum_length=0.05, color=[0,0,1]))


    # print("frustums", len(frustums))
    if bbox_dimension is not None:
        if isinstance(bbox_dimension, dict):
            all_R = bbox_dimension['R']
            all_T = bbox_dimension['T']
            all_s = bbox_dimension['s']
            
            for Rot,Tran,sca in zip(all_R, all_T, all_s):
                bbox = o3d.geometry.OrientedBoundingBox(center = Tran, R = Rot, extent=sca)
                things_to_draw.append(bbox)
            # locations = get_archimedean_spiral(sphere_radius=1.5)
            # all_test_poses = [look_at(loc, [0,0,0])[0] for loc in locations]
            # all_test_poses = convert_pose_spiral(all_test_poses)
            #all_test_poses = create_spheric_poses(radius = 1.5)
        # else: 
        #     bbox_dimension = bbox_dimension/scale_factor
        #     bbox_canonical = o3d.geometry.OrientedBoundingBox(center = [0,0,0], R = np.eye(3), extent=bbox_dimension)
        #     things_to_draw.append(bbox_canonical)
        #     RTs = {'R':np.eye(3), 'T': [0,0,0], 's':bbox_dimension}
            # scale_factor = np.max(bbox_dimension)
            # print("JEREEEEEEEEEEEEEE")
            # locations = get_archimedean_spiral(sphere_radius=1.6)
            # all_test_poses = [look_at(loc, obj_location)[0] for loc in locations]
            # all_test_poses = convert_pose_spiral(all_test_poses)
            #all_test_poses = create_spheric_poses(radius = 1.7)


    
    cameras = frustums2lineset(frustums)
    things_to_draw.append(cameras)

    grid_size=[256, 256, 256]
    sfactor=8
    side_length = 1

    world_grid = get_world_grid([[-side_length, side_length],
                                        [-side_length, side_length],
                                        [0, side_length],
                                        ], [int(grid_size[0]/sfactor), int(grid_size[1]/sfactor), int(grid_size[2]/sfactor)] )  # (1, grid_size**3, 3)
    # world_grid_o3d = o3d.geometry.PointCloud()
    # world_grid_o3d.points = o3d.utility.Vector3dVector(world_grid.squeeze(0).numpy())
    # things_to_draw.append(world_grid_o3d)

    world_grids = world_grid.clone().view(-1,3).unsqueeze(0)
    c2w_single = torch.FloatTensor(all_c2w[0]).unsqueeze(0)
    camera_grids_w2c = world2camera(world_grids, c2w_single, NS=1)

    camera_grid_o3d = o3d.geometry.PointCloud()
    camera_grid_o3d.points = o3d.utility.Vector3dVector(camera_grids_w2c.squeeze(0).numpy())
    things_to_draw.append(camera_grid_o3d)

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius, resolution=5)
    sphere = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=np.array([0., 0., 0.]))
    sphere.paint_uniform_color((0, 1, 0))
    # sphere.transform(T)
    # coordinate_frame.transform(T)
    things_to_draw.append(sphere)
    things_to_draw.append(coordinate_frame)
    for geometry in things_to_draw:
        fig.add_geometry(geometry)

    
    for C2W in all_c2w:
        if asset_pose_inv is not None:
            scale_factor = np.max([(box[1,0]-box[0,0]), (box[1,1]-box[0,1]), (box[1,2]-box[0,2])])
            C2W = asset_pose_inv @ C2W
            C2W[:3,3]/=scale_factor
            C2W[3,3] = 1
        fig.plot_transform(A2B=C2W, s=0.05, strict_check=False)

    for C2W in all_c2w_test:
        if asset_pose_inv is not None:
            scale_factor = np.max([(box[1,0]-box[0,0]), (box[1,1]-box[0,1]), (box[1,2]-box[0,2])])
            C2W = asset_pose_inv @ C2W
            C2W[:3,3]/=scale_factor
            C2W[3,3] = 1
        fig.plot_transform(A2B=C2W, s=0.05, strict_check=False)
    plot_rays(all_c2w[0], focal, img_size[0], img_size[1], fig, RTs=RTs, asset_pose_inv= asset_pose_inv, box=box)
    #plot_rays(all_c2w[1], focal, img_size[0], img_size[1], fig, RTs=RTs, asset_pose_inv= asset_pose_inv, box=box, c2w = c2w)  
    
    #plot world_grid
    fig.show()

def pose_spherical_nerf(euler, radius=0.01):
    c2ws_render = np.eye(4)
    c2ws_render[:3,:3] =  R.from_euler('xyz', euler, degrees=True).as_matrix()
    c2ws_render[:3,3]  = c2ws_render[:3,:3] @ np.array([0.0,0.0,-radius])
    return c2ws_render

def create_spheric_poses(radius, n_poses=50):
    """
    Create circular poses around z axis.
    Inputs:
        radius: the (negative) height and the radius of the circle.

    Outputs:
        spheric_poses: (n_poses, 3, 4) the poses in the circular path
    """
    def spheric_pose(theta, phi, radius):
        trans_t = lambda t : np.array([
            [1,0,0,0],
            [0,1,0,0.3*t],
            [0,0,1,t],
            [0,0,0,1],
        ])

        rot_phi = lambda phi : np.array([
            [1,0,0,0],
            [0,np.cos(phi),-np.sin(phi),0],
            [0,np.sin(phi), np.cos(phi),0],
            [0,0,0,1],
        ])

        rot_z = lambda phi : np.array([
            [np.cos(phi),-np.sin(phi),0,0],
            [np.sin(phi),np.cos(phi),0,0],
            [0,0, 1,0],
            [0,0,0,1],
        ])

        rot_theta = lambda th : np.array([
            [np.cos(th),0,-np.sin(th),0],
            [0,1,0,0],
            [np.sin(th),0, np.cos(th),0],
            [0,0,0,1],
        ])
        c2w =  rot_theta(theta) @ trans_t(radius) @ rot_phi(phi)
        # c2w = rot_theta(theta) @ rot_phi(phi) @ trans_t(radius)
        c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
        # c2w = rot_phi(phi) @ c2w
        return c2w
    spheric_poses = []
    for th in np.linspace(0, 2*np.pi, n_poses+1)[:-1]:
        #spheric_poses += [spheric_pose(th, -np.pi/5, radius)] # 36 degree view downwards
        spheric_poses += [spheric_pose(th, -np.pi/15, radius)] # 36 degree view downwards
    return np.stack(spheric_poses, 0)

def move_camera_pose(pose, progress):
    # control the camera move (spiral pose)
    t = progress * np.pi * 4
    # radii = 0
    radii = 0.005
    center = np.array([np.cos(t), -np.sin(t), -np.sin(0.5 * t)]) * radii
    pose[:3, 3] += pose[:3, :3] @ center
    return pose

def get_pure_rotation(progress_11: float, max_angle: float = 360):
    trans_pose = np.eye(4)
    print("progress_11", progress_11)
    print("max angle")
    print("progress_11 * max_angle", progress_11 * max_angle)
    trans_pose[:3, :3] = Rotation.from_euler(
        "z", progress_11 * max_angle, degrees=True
    ).as_matrix()
    return trans_pose

def get_pure_translation(progress_11: float, axis = 'x', max_distance = 2):
    trans_pose = np.eye(4)
    if axis == 'x':
        trans_pose[0, 3] = progress_11 * max_distance
    elif axis == 'y':
        trans_pose[1, 3] = progress_11 * max_distance
    elif axis =='z':
        trans_pose[2, 3] = progress_11 * max_distance
    return trans_pose

def load_intrinsic(intrinsic_path):
    with open(intrinsic_path, 'r') as f:
        lines = f.readlines()
        focal = float(lines[0].split()[0])
        H, W = lines[-1].split()
        H, W = int(H), int(W)
    return focal, H, W

# def read_poses(pose_dir, new=False):
#     pose_file = os.path.join(pose_dir, 'pose.json')
#     with open(pose_file, "r") as read_content:
#         data = json.load(read_content)
#     focal = data['focal']
#     fov = data['fov']
#     img_wh = data['img_size']
#     all_c2w = []
#     all_image_names = []
#     for k,v in data['transform'].items():
#         all_c2w.append(np.array(v))
#         all_image_names.append(k)
#     if not new:
#         bbox_dimensions = data['bbox_dimensions']
#         return all_c2w, focal, fov, img_wh, bbox_dimensions
#     else:
#         bbox_dimensions = data['bbox_dimensions']
#         all_translations = data['obj_translations']
#         all_rotations = data["obj_rotations"]
#         return all_c2w, focal, fov, img_wh, bbox_dimensions, all_rotations, all_translations


def convert_pose_PD_to_NeRF(C2W):

    flip_axes = np.array([[1,0,0,0],
                         [0,0,-1,0],
                         [0,1,0,0],
                         [0,0,0,1]])
    C2W = np.matmul(C2W, flip_axes)
    return C2W

def pad_poses(p: np.ndarray) -> np.ndarray:
  """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
  bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
  return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p: np.ndarray) -> np.ndarray:
  """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
  return p[..., :3, :4]

from typing import Tuple

def transform_poses_pca(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Transforms poses so principal components lie on XYZ axes.
  Args:
    poses: a (N, 3, 4) array containing the cameras' camera to world transforms.
  Returns:
    A tuple (poses, transform), with the transformed poses and the applied
    camera_to_world transforms.
  """
  t = poses[:, :3, 3]
  t_mean = t.mean(axis=0)
  t = t - t_mean

  eigval, eigvec = np.linalg.eig(t.T @ t)
  # Sort eigenvectors in order of largest to smallest eigenvalue.
  inds = np.argsort(eigval)[::-1]
  eigvec = eigvec[:, inds]
  rot = eigvec.T
  if np.linalg.det(rot) < 0:
    rot = np.diag(np.array([1, 1, -1])) @ rot

  transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
  poses_recentered = unpad_poses(transform @ pad_poses(poses))

  # Just make sure it's it in the [-1, 1]^3 cube
  scale_factor = 1. / np.max(np.abs(poses_recentered[:, :3, 3]))
  poses_recentered[:, :3, 3] *= scale_factor
  transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform

  return poses_recentered, transform, scale_factor


def read_poses_with_bbox(pose_dir, new=False):
    pose_file = os.path.join(pose_dir, 'pose.json')
    with open(pose_file, "r") as read_content:
        data = json.load(read_content)
    focal = data['focal']
    fov = data['fov']
    img_wh = data['img_size']
    asset_pose_ = data["vehicle_pose"]
    all_c2w = []
    all_image_names = []
    for k,v in data['transform'].items():
        all_c2w.append(convert_pose_PD_to_NeRF(np.linalg.inv(asset_pose_) * np.array(v)))
        all_image_names.append(k)

    all_c2w = np.array(all_c2w)
    bbox_dimensions = data['bbox_dimensions']
    print("all_c2w", all_c2w.shape)

    #scale to fit inside a unit bounding box
    pose_scale_factor = 1. / np.max(np.abs(all_c2w[:, :3, 3]))

    print("pose_scale_factor", pose_scale_factor)
    bbox_dimensions = np.array(bbox_dimensions)*pose_scale_factor
    all_c2w[:, :3, 3] *= pose_scale_factor
    # if not new:
    
    return all_c2w, focal, fov, img_wh, bbox_dimensions

def read_poses_new(pose_dir, new=False):
    pose_file = os.path.join(pose_dir, 'pose.json')
    with open(pose_file, "r") as read_content:
        data = json.load(read_content)
    focal = data['focal']
    fov = data['fov']
    img_wh = data['img_size']
    obj_location = np.array(data["obj_location"])
    all_c2w = []
    all_image_names = []
    for k,v in data['transform'].items():
        c2w = np.array(v)
        c2w[:3, 3] = c2w[:3, 3] - obj_location
        all_c2w.append(convert_pose_PD_to_NeRF(c2w))
        all_image_names.append(k)

    all_c2w = np.array(all_c2w)

    pose_scale_factor = 1. / np.max(np.abs(all_c2w[:, :3, 3]))
    all_c2w[:, :3, 3] *= pose_scale_factor

    all_boxes = []
    for k,v in data['bbox_dimensions'].items():
            bbox = np.array(v)
            bbox_dimension =  [(bbox[1,0]-bbox[0,0]), (bbox[1,1]-bbox[0,1]), (bbox[1,2]-bbox[0,2])]
            all_boxes.append(np.array(bbox_dimension)*pose_scale_factor)
    all_translations = (np.array(data['obj_translations'])- obj_location)*pose_scale_factor
    all_rotations = data["obj_rotations"]
    return all_c2w, focal, fov, img_wh, all_boxes, all_rotations, all_translations

def read_poses(pose_dir_train, img_files_train):
    pose_file_train = os.path.join(pose_dir_train, 'pose.json')
    with open(pose_file_train, "r") as read_content:
        data = json.load(read_content)

    focal = data['focal']
    fov = data['fov']
    img_wh = data['img_size']
    obj_location = np.array(data["obj_location"])
    all_c2w_train = []

    for img_file in img_files_train:
        c2w = np.array(data['transform'][img_file.split('.')[0]])
        c2w[:3, 3] = c2w[:3, 3] - obj_location
        all_c2w_train.append(convert_pose_PD_to_NeRF(c2w))

    all_c2w_train = np.array(all_c2w_train)
    pose_scale_factor = 1. / np.max(np.abs(all_c2w_train[:, :3, 3]))
    all_c2w_train[:, :3, 3] *= pose_scale_factor
    # c2w = all_c2w_train[0]

    # # all_c2w_train_transformed = []
    # # for i in range(all_c2w_train.shape[0]):
    # #     all_c2w_train_transformed.append(np.linalg.inv(c2w)@all_c2w_train[i,:,:])
    # # all_c2w_train = np.array(all_c2w_train_transformed)


    all_c2w_val = all_c2w_train[100:]
    all_c2w_train = all_c2w_train[:100]

    all_boxes = []
    all_boxes_new = []
    all_boxes_newer = []

    all_boxes = []
    all_translations= []
    all_rotations = []

    for k,v in data['bbox_dimensions'].items():
            bbox = np.array(v)
            bbox_dimension =  [(bbox[1,0]-bbox[0,0]), (bbox[1,1]-bbox[0,1]), (bbox[1,2]-bbox[0,2])]
            all_boxes.append(np.array(bbox_dimension)*pose_scale_factor)
            all_boxes_new.append(bbox*pose_scale_factor)
            # all_boxes.append(np.array(bbox_dimension))
            # all_boxes_new.append(bbox)
            
            #New scene 200 uncomment here

            box_transform = np.eye(4)
            box_transform[:3,:3] = np.array(data["obj_rotations"][k])
            box_transform[:3, 3] = (np.array(data['obj_translations'][k])- obj_location)*pose_scale_factor 

            # box_transform = np.linalg.inv(c2w) @ box_transform
            # Rot_transformed = box_transform[:3,:3]
            # Tran_transformed = box_transform[:3, 3]

            # all_rotations.append(Rot_transformed)
            # all_translations.append(Tran_transformed)

            # all_rotations.append(data["obj_rotations"][k])
            # translation = (np.array(data['obj_translations'][k])- obj_location)*pose_scale_factor 
            # all_translations.append(translation)

    # for k,v in data['bbox_dimensions'].items():
    #         bbox = np.array(v)
    #         bbox_dimension =  [(bbox[1,0]-bbox[0,0]), (bbox[1,1]-bbox[0,1]), (bbox[1,2]-bbox[0,2])]
    #         all_boxes.append(np.array(bbox_dimension)*pose_scale_factor)
    #         all_boxes_new.append(bbox*pose_scale_factor)

    # all_translations = (np.array(data['obj_translations'])- obj_location)*pose_scale_factor
    # all_rotations = data["obj_rotations"]

    #bbox_dimension_modified = {'R': all_rotations, 'T': all_translations_new, 's': all_boxes_new}
    bbox_dimension_modified = {'R': all_rotations, 'T': all_translations, 's': all_boxes}
    RTs = {'R': all_rotations, 'T': all_translations, 's': all_boxes_new}
    return all_c2w_train, all_c2w_val, focal, fov, img_wh, bbox_dimension_modified, RTs, pose_scale_factor, obj_location


def read_poses_new_all(pose_dir_train, pose_dir_val, new=False):
    pose_file_train = os.path.join(pose_dir_train, 'pose.json')
    pose_file_val = os.path.join(pose_dir_val, 'pose.json')
    with open(pose_file_train, "r") as read_content:
        data = json.load(read_content)
    with open(pose_file_val, "r") as read_content:
        data_val = json.load(read_content)

    focal = data['focal']
    fov = data['fov']
    img_wh = data['img_size']
    obj_location = np.array(data["obj_location"])
    all_c2w = []
    all_image_names = []
    for k,v in data['transform'].items():
        c2w = np.array(v)
        c2w[:3, 3] = c2w[:3, 3] - obj_location
        all_c2w.append(convert_pose_PD_to_NeRF(c2w))
        all_image_names.append(k)

    # for k,v in data_val['transform'].items():
    #     c2w = np.array(v)
    #     c2w[:3, 3] = c2w[:3, 3] - obj_location
    #     all_c2w.append(convert_pose_PD_to_NeRF(c2w))
    #     all_image_names.append(k)

    all_c2w = np.array(all_c2w)

    pose_scale_factor = 1. / np.max(np.abs(all_c2w[:, :3, 3]))

    print("pose_scale_factor", pose_scale_factor)
    all_c2w[:, :3, 3] *= pose_scale_factor

    all_c2w_train = all_c2w[:100, :, :]
    all_c2w_test = all_c2w[100:, :, :]

    all_boxes = []
    all_boxes_new = []
    all_boxes_newer = []

    all_boxes = []
    all_translations= []
    all_rotations = []

    for k,v in data['bbox_dimensions'].items():
            bbox = np.array(v)
            bbox_dimension =  [(bbox[1,0]-bbox[0,0]), (bbox[1,1]-bbox[0,1]), (bbox[1,2]-bbox[0,2])]
            all_boxes.append(np.array(bbox_dimension)*pose_scale_factor)
            all_boxes_new.append(bbox*pose_scale_factor)
            
            #New scene 200 uncomment here
            all_rotations.append(data["obj_rotations"][k])
            translation = (np.array(data['obj_translations'][k])- obj_location)*pose_scale_factor 
            all_translations.append(translation)

    # for k,v in data['bbox_dimensions'].items():
    #         bbox = np.array(v)
    #         bbox_dimension =  [(bbox[1,0]-bbox[0,0]), (bbox[1,1]-bbox[0,1]), (bbox[1,2]-bbox[0,2])]
    #         all_boxes.append(np.array(bbox_dimension)*pose_scale_factor)
    #         all_boxes_new.append(bbox*pose_scale_factor)

    # all_translations = (np.array(data['obj_translations'])- obj_location)*pose_scale_factor
    # all_rotations = data["obj_rotations"]

    #bbox_dimension_modified = {'R': all_rotations, 'T': all_translations_new, 's': all_boxes_new}
    bbox_dimension_modified = {'R': all_rotations, 'T': all_translations, 's': all_boxes}
    RTs = {'R': all_rotations, 'T': all_translations, 's': all_boxes_new}
    return all_c2w_train, all_c2w_test, focal, fov, img_wh, bbox_dimension_modified, RTs, pose_scale_factor, obj_location


# def read_poses(pose_dir, new=False):
#     pose_file = os.path.join(pose_dir, 'pose.json')
#     with open(pose_file, "r") as read_content:
#         data = json.load(read_content)
#     focal = data['focal']
#     fov = data['fov']
#     img_wh = data['img_size']
#     asset_pose_ = data["vehicle_pose"]
#     all_c2w = []
#     all_image_names = []
#     for k,v in data['transform'].items():
#         all_c2w.append(convert_pose_PD_to_NeRF(np.array(v)))
#         all_image_names.append(k)

#     all_c2w = np.array(all_c2w)
#     bbox_dimensions = data['bbox_dimensions']
#     asset_pose_inv = np.linalg.inv(asset_pose_)
#     return all_c2w, focal, fov, img_wh, bbox_dimensions, asset_pose_inv

def get_masked_textured_pointclouds(depth,rgb, intrinsics, width, height):
    xmap = np.array([[y for y in range(width)] for z in range(height)])
    ymap = np.array([[z for y in range(width)] for z in range(height)])
    cam_cx = intrinsics[0,2]
    cam_fx = intrinsics[0,0]
    cam_cy = intrinsics[1,2]
    cam_fy = intrinsics[1,1]

    depth_masked = depth.reshape(-1)[:, np.newaxis]
    xmap_masked = xmap.flatten()[:, np.newaxis]
    ymap_masked = ymap.flatten()[:, np.newaxis]
    rgb = rgb.reshape(-1,3)/255.0
    pt2 = depth_masked
    pt0 = (xmap_masked - cam_cx) * pt2 / cam_fx
    pt1 = (ymap_masked - cam_cy) * pt2 / cam_fy
    points = np.concatenate((pt0, pt1, pt2), axis=1)
    return points, rgb

def convert_nerf_to_PD(C2W):

    flip_axes = np.array([[1,0,0,0],
                         [0,0,-1,0],
                         [0,1,0,0],
                         [0,0,0,1]])
    # flip_yz = np.eye(4)
    # flip_yz[1, 1] = -1
    # flip_yz[2, 2] = -1
    C2W = np.matmul(C2W, np.linalg.inv(flip_axes))
    return C2W

def custom_draw_geometry_with_key_callback(pcd):

    def change_background_to_black(vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        return False

    def load_render_option(vis):
        vis.get_render_option().load_from_json(
            "../../TestData/renderoption.json")
        return False

    def capture_depth(vis):
        depth = vis.capture_depth_float_buffer()
        plt.imshow(np.asarray(depth))
        plt.show()
        return False

    def capture_image(vis):
        image = vis.capture_screen_float_buffer()
        plt.imshow(np.asarray(image))
        plt.show()
        return False

    def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(0.5, 0.0)
        return False

    key_to_callback = {}
    key_to_callback[ord("K")] = change_background_to_black
    key_to_callback[ord("R")] = load_render_option
    key_to_callback[ord(",")] = capture_depth
    key_to_callback[ord(".")] = capture_image
    key_to_callback[ord("o")] = rotate_view

    o3d.visualization.draw_geometries_with_key_callbacks(pcd, key_to_callback)


def custom_draw_geometry_with_rotation(pcd):

    def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(0.8, 0.0)
        return False

    o3d.visualization.draw_geometries_with_animation_callback(pcd,
                                                              rotate_view)
                                                              
def convert_pose(C2W):
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = np.matmul(C2W, flip_yz)
    return C2W

if __name__ == '__main__':
    import os
    import json
    new = True
    # base_dir = '/home/zubair/pd-api-py/multi_neural_rendering_data_test_3'
    
    scene_path = '/home/zubairirshad/pd-api-py/PDMultiObjv4/train/SF_6thAndMission_medium19'
    if new:
        base_dir_train = os.path.join(scene_path, 'train')
        base_dir_val = os.path.join(scene_path, 'val')

        img_files_train = os.listdir(os.path.join(base_dir_train, 'rgb'))
        img_files_train.sort()
        
        #all_c2w, focal, fov, img_size, bbox_dimensions, all_rotations, all_translations = read_poses_new(pose_dir = os.path.join(base_dir_train, 'pose'), new=new)
        #all_c2w_test, focal_test, fov_test, img_size_test, _, _, _ = read_poses_new(pose_dir = os.path.join(base_dir_val, 'pose'), new=new)

        all_c2w, all_c2w_test,  qgmagma, fov, img_size, \
           bbox_dimension_modified, RTs, pose_scale_factor, obj_location = read_poses(pose_dir_train = os.path.join(base_dir_train, 'pose'), img_files_train = img_files_train)
        #RTS = {'R': all_rotations, 'T': all_translations, 's': bbox_dimensions}
        
        # for i, bbox in enumerate(bbox_dimensions):
        #     v = bbox['asset_name']
        #     scale_factor = np.max(v)
        #     all_bboxes.append(np.array(v)/scale_factor)
        #     all_T.append(np.array(all_translations[i])/scale_factor)
        #     all_R.append(np.array(all_rotations[i]/scale_factor))

        # print("fov", fov, focal)
        #visualize_poses(all_c2w, all_c2w_test, fov=fov, sphere_radius = 1.0, obj_location = [0,0,0], bbox_dimension= bbox_dimension_modified, RTs=RTs, pose_scale_factor = pose_scale_factor, obj_location=obj_location, asset_pose_inv = None, box= None)
        visualize_poses(all_c2w, all_c2w_test, fov=fov, sphere_radius = 1.0, obj_location = [0,0,0], bbox_dimension= bbox_dimension_modified, RTs=RTs)
    else:
        base_dir = '/home/zubairirshad/pd-api-py/PDMultiObjv3/SF_6thAndMission_medium9'
        # all_c2w, focal, fov, img_size, bbox_dimensions = read_poses_with_bbox(pose_dir = os.path.join(base_dir, 'train', 'pose'))
        # all_c2w_test, focal_test, fov_test, img_size_test, bbox_dimensions_test = read_poses_with_bbox(pose_dir = os.path.join(base_dir,'val', 'pose'))

        all_c2w, focal, fov, img_size, bbox_dimensions, asset_pose_inv = read_poses(pose_dir = os.path.join(base_dir, 'train', 'pose'))
        all_c2w_test, focal_test, fov_test, img_size_test, bbox_dimensions_test, asset_pose_inv = read_poses(pose_dir = os.path.join(base_dir,'val', 'pose'))
        bbox_dimensions = np.array(bbox_dimensions)
        # bbox_dimension_modified = bbox_dimensions
        bbox_dimension_modified =  bbox_dimensions[1,:] - bbox_dimensions[0,:]
        # scale_factor = np.max(bbox_dimensions)
        # bbox_dimension_modified = bbox_dimensions/scale_factor
        print("fov", fov, focal)
        visualize_poses(all_c2w, all_c2w_test, fov=fov, sphere_radius = 1.0, obj_location = [0,0,0], bbox_dimension= bbox_dimension_modified, asset_pose_inv = asset_pose_inv, box= bbox_dimensions)

