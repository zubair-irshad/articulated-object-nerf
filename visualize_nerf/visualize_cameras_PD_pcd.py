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
# from datasets.ray_utils import world_to_ndc, get_ray_bbox_intersections, get_rays_in_bbox
import torchvision.transforms as T
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Rotation as R
import cv2

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

def get_object_rays_in_bbox(rays_o, rays_d, RTs, canonical=False):
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1

    if not canonical:
        instance_rotation = RTs['R']
        instance_translation = RTs['T']
        scale = RTs['s']

        box_transformation = np.eye(4)
        box_transformation[:3, :3] = np.reshape(instance_rotation, (3, 3))
        box_transformation[:3, -1] = instance_translation
        axis_aligned_mat = np.linalg.inv(box_transformation)
    else:
        scale = RTs['s']
        axis_aligned_mat = None
    bbox_bounds = np.array([-scale / 2, scale / 2])
    rays_o_obj, rays_d_obj, batch_near_obj, batch_far_obj = get_rays_in_bbox(rays_o, rays_d, bbox_bounds, None, axis_aligned_mat)
    
    return rays_o_obj, rays_d_obj, batch_near_obj, batch_far_obj

def transform_coordinates_3d(coordinates, RT):
    """
    Args:
        coordinates: [3, N]
        sRT: [4, 4]
    Returns:
        new_coordinates: [3, N]
    """
    # assert coordinates.shape[0] == 3
    # coordinates = np.vstack([coordinates, np.ones((1, coordinates.shape[1]), dtype=np.float32)])
    # new_coordinates = sRT @ coordinates
    # new_coordinates = new_coordinates[:3, :] / new_coordinates[3, :]

    unit_box_homopoints = convert_points_to_homopoints(coordinates)
    # morphed_box_homopoints = pose.camera_T_object @ (pose.scale_matrix @ unit_box_homopoints)
    morphed_box_homopoints = RT @ unit_box_homopoints
    new_coordinates = convert_homopoints_to_points(morphed_box_homopoints)

    return new_coordinates

def calculate_2d_projections(coordinates_3d, intrinsics):
    """
    Args:
        coordinates_3d: [3, N]
        intrinsics: [3, 3]
    Returns:
        projected_coordinates: [N, 2]
    """
    projected_coordinates = intrinsics @ coordinates_3d
    projected_coordinates = projected_coordinates[:2, :] / projected_coordinates[2, :]
    projected_coordinates = projected_coordinates.transpose()
    projected_coordinates = np.array(projected_coordinates, dtype=np.int32)

    return projected_coordinates

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
    print("directions", directions.shape)
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


def draw_bboxes_mpl_glow(img, img_pts, axes, color):
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
            
    # draw axes
    img = cv2.arrowedLine(img, tuple(axes[0]), tuple(axes[1]), (0, 0, 255), 4)
    img = cv2.arrowedLine(img, tuple(axes[0]), tuple(axes[3]), (255, 0, 0), 4)
    img = cv2.arrowedLine(img, tuple(axes[0]), tuple(axes[2]), (0, 255, 0), 4) ## y last

    return img

def get_rays(directions, c2w):
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
    rays_d /= torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, 3].expand(rays_d.shape) # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d

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

def plot_rays(c2w, focal, W, H, fig):
    focal = focal
    directions = get_ray_directions(H, W, focal) # (h, w, 3)
    c2w = torch.FloatTensor(c2w)[:3, :4]
    rays_o, rays_d = get_rays(directions, c2w)

    rays_o = rays_o.numpy()
    rays_d = rays_d.numpy()
    ids = np.random.choice(rays_o.shape[0], int(rays_o.shape[0]*0.5))

    rays_o = rays_o[ids, :]
    rays_d = rays_d[ids, :]
    
    for j in range(2500):
        start = rays_o[j,:]
        end = rays_o[j,:] + rays_d[j,:]*0.02
        line = np.concatenate((start[None, :],end[None, :]), axis=0)
        fig.plot(line, c=(1.0, 0.5, 0.0))

        start = rays_o[j,:] + rays_d[j,:]*0.02
        end = rays_o[j,:] + rays_d[j,:]*2
        line = np.concatenate((start[None, :],end[None, :]), axis=0)
        fig.plot(line, c=(0.0, 1.0, 0.0))



def visualize_poses(all_c2w, all_c2w_test, fov, obj_location, sphere_radius=1, bbox_dimension= None, W=None):
    T = np.eye(4)
    T[:3,3] = obj_location
    things_to_draw = []
    focal = (640 / 2) / np.tan((fov / 2) / (180 / np.pi))
    frustums = []
    fig = pv.figure()
    for C2W in all_c2w:
        img_size = (640, 480)
        frustums.append(get_camera_frustum(img_size, focal, C2W, frustum_length=0.8, color=[0,1,0]))
    for C2W in all_c2w_test:
        frustums.append(get_camera_frustum(img_size, focal, C2W, frustum_length=0.2, color=[0,0,1]))

    # print("frustums", len(frustums))
    if bbox_dimension is not None:
        if isinstance(bbox_dimension, dict):
            all_R = bbox_dimension['R']
            all_T = bbox_dimension['T']
            all_s = bbox_dimension['s']
            
            for Rot,Tran,sca in zip(all_R, all_T, all_s):
                bbox = o3d.geometry.OrientedBoundingBox(center = Tran, R = Rot, extent=sca)
                things_to_draw.append(bbox)
            locations = get_archimedean_spiral(sphere_radius=1.5)
            all_test_poses = [look_at(loc, [0,0,0])[0] for loc in locations]
            all_test_poses = convert_pose_spiral(all_test_poses)
            #all_test_poses = create_spheric_poses(radius = 1.5)
        else: 
            bbox_canonical = o3d.geometry.OrientedBoundingBox(center = [0,0,0], R = np.eye(3), extent=bbox_dimension)
            things_to_draw.append(bbox_canonical)
            # scale_factor = np.max(bbox_dimension)
            # print("JEREEEEEEEEEEEEEE")
            # locations = get_archimedean_spiral(sphere_radius=1.6)
            # all_test_poses = [look_at(loc, obj_location)[0] for loc in locations]
            # all_test_poses = convert_pose_spiral(all_test_poses)
            #all_test_poses = create_spheric_poses(radius = 1.7)


    cameras = frustums2lineset(frustums)
    things_to_draw.append(cameras)

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius, resolution=5)
    sphere = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=np.array([0., 0., 0.]))
    sphere.paint_uniform_color((1, 0, 0))
    sphere.transform(T)
    things_to_draw.append(sphere)
    things_to_draw.append(coordinate_frame)
    for geometry in things_to_draw:
        fig.add_geometry(geometry)
    for C2W in all_c2w:
        fig.plot_transform(A2B=C2W, s=0.2, strict_check=False)

    for C2W in all_c2w_test:
        fig.plot_transform(A2B=C2W, s=0.08, strict_check=False)
    plot_rays(all_c2w[1], focal, img_size[0], img_size[1], fig)  
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

def convert_pose_PD_to_NeRF(C2W):

    flip_axes = np.array([[1,0,0,0],
                         [0,0,-1,0],
                         [0,1,0,0],
                         [0,0,0,1]])
    C2W = np.matmul(C2W, flip_axes)
    return C2W

def read_poses_new(pose_dir, img_files):
    pose_file = os.path.join(pose_dir, 'pose.json')
    with open(pose_file, "r") as read_content:
        data = json.load(read_content)
    focal = data['focal']
    fov = data['fov']
    img_wh = data['img_size']
    obj_location = np.array(data["obj_location"])
    all_c2w = []

    for img_file in img_files:
        c2w = np.array(data['transform'][img_file.split('.')[0]])
        all_c2w.append(convert_pose_PD_to_NeRF(c2w))

    # for k,v in data['transform'].items():
    #     c2w = np.array(v)
    #     # c2w[:3, 3] = c2w[:3, 3] - obj_location
        # all_c2w.append(convert_pose_PD_to_NeRF(c2w))

    all_c2w = np.array(all_c2w)

    pose_scale_factor = 1. / np.max(np.abs(all_c2w[:, :3, 3]))
    # all_c2w[:, :3, 3] *= pose_scale_factor

    all_boxes = []
    all_translations = []
    all_rotations= []
    for k,v in data['bbox_dimensions'].items():
            bbox = np.array(v)
            bbox_dimension =  [(bbox[1,0]-bbox[0,0]), (bbox[1,1]-bbox[0,1]), (bbox[1,2]-bbox[0,2])]
            # all_boxes.append(np.array(bbox_dimension)*pose_scale_factor)
            all_boxes.append(np.array(bbox_dimension))
            #translation = (np.array(data['obj_translations'][k])- obj_location)*pose_scale_factor
            translation = (np.array(data['obj_translations'][k]))
            all_translations.append(translation)
            all_rotations.append(data["obj_rotations"][k])
    # all_translations = (np.array(data['obj_translations'])- obj_location)*pose_scale_factor
    # all_rotations = data["obj_rotations"]
    return all_c2w, focal, img_wh, all_boxes, all_rotations, all_translations
    # return all_c2w, focal, fov, img_wh, all_boxes, all_rotations, all_translations

# def read_poses(pose_dir, new=False):
#     pose_file = os.path.join(pose_dir, 'pose.json')
#     with open(pose_file, "r") as read_content:
#         data = json.load(read_content)
#     focal = data['focal']
#     img_wh = data['img_size']
#     all_c2w = []
#     all_image_names = []
#     for k,v in data['transform'].items():
#         all_c2w.append(np.array(v))
#         all_image_names.append(k)
    
#     if not new:
#         bbox_dimensions = data['bbox_dimensions']
#         return all_c2w, focal, img_wh, bbox_dimensions
#     else:
#         bbox_dimensions = data['bbox_dimensions']
#         all_translations = data['obj_translations']
#         all_rotations = data["obj_rotations"]
#         return all_c2w, focal, img_wh, bbox_dimensions, all_rotations, all_translations

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
    # all_c2w_train[:, :3, 3] *= pose_scale_factor
    all_c2w_val = all_c2w_train[100:]
    all_c2w_train = all_c2w_train[:100]
    # all_c2w_train = all_c2w_val

    bbox_dimensions = []

    all_translations= []
    all_rotations = []

    for k,v in data['bbox_dimensions'].items():
            bbox = np.array(v)
            # bbox_dimension =  [(bbox[1,0]-bbox[0,0]), (bbox[1,1]-bbox[0,1]), (bbox[1,2]-bbox[0,2])]
            # bbox_dimensions.append(np.array(bbox_dimension)*pose_scale_factor)

            bbox_dimensions.append(np.array(v))
            # all_boxes_new.append(bbox*pose_scale_factor)
            
            #New scene 200 uncomment here
            all_rotations.append(data["obj_rotations"][k])
            # translation = np.array(data['obj_translations'][k])

            translation = np.array(data['obj_translations'][k] - obj_location)
            all_translations.append(translation)

    return all_c2w_train, focal, img_wh, bbox_dimensions, all_rotations, all_translations

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

def get_RTs(obj_poses):
    all_boxes = []
    all_translations = []
    all_rotations= []
    for i, bbox in enumerate(obj_poses['bbox_dimensions']):
            all_boxes.append(bbox)
            translation = np.array(obj_poses['obj_translations'][i])
            all_translations.append(translation)
            all_rotations.append(obj_poses["obj_rotations"][i])
    RTs = {'R': all_rotations, 'T': all_translations, 's': all_boxes}
    return RTs

def draw_combined_pcds_boxes(obj_pose, all_depth_pc, all_rgb_pc, all_c2w):
    RTs = get_RTs(obj_pose)
    all_pcd = []
    all_rotations = RTs['R']
    all_translations = RTs['T']
    bbox_dimensions = RTs['s']
    for Rot,Tran,bbox in zip(all_rotations, all_translations, bbox_dimensions):
        Tran = np.array(Tran)
        box_transform = np.eye(4)
        box_transform[:3,:3] = np.array(Rot)
        box_transform[:3, 3] = np.array(Tran)
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=np.array([0., 0., 0.]))
        coordinate_frame.transform(box_transform)
        sca = bbox
        bbox = o3d.geometry.OrientedBoundingBox(center = Tran, R = Rot, extent=sca)
        all_pcd.append(bbox)
        all_pcd.append(coordinate_frame)

    src_view_num = [0]

    print("len all",len(all_c2w), len(all_depth_pc), len(all_c2w))

    all_depth_pc_src = [all_depth_pc[i] for i in src_view_num]
    all_rgb_pc_src = [all_rgb_pc[i] for i in src_view_num]
    all_c2w_src = [all_c2w[i] for i in src_view_num]

    all_depth_pc_rest = all_depth_pc[100:]
    all_rgb_pc_rest = all_rgb_pc[100:]
    all_c2w_rest = all_c2w[100:]

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=5)
    sphere = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=np.array([0., 0., 0.]))
    sphere.paint_uniform_color((1, 0, 0))
    all_pcd.append(sphere)
    all_pcd.append(coordinate_frame)

    for pc, rgb, pose in zip(all_depth_pc_src, all_rgb_pc_src, all_c2w_src):
        pcd_vis = o3d.geometry.PointCloud()
        pcd_vis.points = o3d.utility.Vector3dVector(pc)
        pcd_vis.colors = o3d.utility.Vector3dVector(rgb)
        pcd_vis.transform(convert_pose(pose))

        # pcd_vis.transform(convert_pose(pose))
        all_pcd.append(pcd_vis)
        # display camera origin
        FOR_cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=np.array([0., 0., 0.]))
        FOR_cam.transform(convert_pose(pose))
        # FOR_cam.transform(pose)
        # PLOT CAMERAS HEREEEE
        all_pcd.append(FOR_cam)
        frustums = []
        for C2W in all_c2w_src:
            img_size = (640, 480)
            frustums.append(get_camera_frustum(img_size, focal, convert_pose(C2W), frustum_length=1.0, color=[1,0,0]))
            #frustums.append(get_camera_frustum(img_size, focal, C2W, frustum_length=1.0, color=[1,0,0]))

        cameras = frustums2lineset(frustums)
        # PLOT CAMERAS HEREEEE
        all_pcd.append(cameras)

    for pc, rgb, pose in zip(all_depth_pc_rest, all_rgb_pc_rest, all_c2w_rest):
        pcd_vis = o3d.geometry.PointCloud()
        
        pcd_vis.points = o3d.utility.Vector3dVector(pc)
        pcd_vis.colors = o3d.utility.Vector3dVector(rgb)
        pcd_vis.transform(convert_pose(pose))
        # pcd_vis.transform(pose)
        all_pcd.append(pcd_vis)
        # display camera origin
        FOR_cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=np.array([0., 0., 0.]))
        FOR_cam.transform(convert_pose(pose))
        # FOR_cam.transform(pose)
        # PLOT CAMERAS HEREEEE
        all_pcd.append(FOR_cam)
        frustums = []
        for C2W in all_c2w_rest:
            img_size = (640, 480)
            frustums.append(get_camera_frustum(img_size, focal, convert_pose(C2W), frustum_length=0.5, color=[0,1,0]))
            # frustums.append(get_camera_frustum(img_size, focal, C2W, frustum_length=0.5, color=[0,1,0]))

        cameras = frustums2lineset(frustums)
        # PLOT CAMERAS HEREEEE
        all_pcd.append(cameras)
    
    custom_draw_geometry_with_key_callback(all_pcd)

def separate_lists(lst, indices):
    first_list = [lst[i] for i in range(len(lst)) if i in indices]
    second_list = [lst[i] for i in range(len(lst)) if i not in indices]
    return first_list, second_list

def draw_pcd_and_box(depth, rgb_pc, obj_pose, c2w):
    RTs = get_RTs(obj_pose)
    all_pcd = []
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(depth.flatten(1).t()))

    # pcd.colors = o3d.utility.Vector3dVector(rgb_pc)
    all_pcd.append(pcd)
    all_rotations = RTs['R']
    all_translations = RTs['T']
    bbox_dimensions = RTs['s']

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=5)
    sphere = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=np.array([0., 0., 0.]))
    sphere.paint_uniform_color((1, 0, 0))
    all_pcd.append(sphere)
    all_pcd.append(coordinate_frame)

    for Rot,Tran,bbox in zip(all_rotations, all_translations, bbox_dimensions):
        Tran = np.array(Tran)
        box_transform = np.eye(4)
        box_transform[:3,:3] = np.array(Rot)
        box_transform[:3, 3] = np.array(Tran)

        box_transform = np.linalg.inv(convert_pose(c2w)) @ box_transform
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=np.array([0., 0., 0.]))
        coordinate_frame.transform(box_transform)

        sca = bbox

        Rot_transformed = box_transform[:3,:3]
        Tran_transformed = box_transform[:3, 3]

        bbox = o3d.geometry.OrientedBoundingBox(center = Tran_transformed, R = Rot_transformed, extent=sca)
        all_pcd.append(bbox)
        all_pcd.append(coordinate_frame)


        frustums = []
        img_size = (640, 480)
        frustums.append(get_camera_frustum(img_size, focal, convert_pose(c2w), frustum_length=1.0, color=[1,0,0]))
        cameras = frustums2lineset(frustums)
        FOR_cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=np.array([0., 0., 0.]))
        FOR_cam.transform(convert_pose(pose))
        all_pcd.append(FOR_cam)
        all_pcd.append(cameras)

    custom_draw_geometry_with_key_callback(all_pcd)

def preprocess_RTS_for_vis(RTS):
    all_R = RTS['R']
    all_T = RTS['T']
    all_s = RTS['s']

    obj_poses = {}
    obj_poses["bbox_dimensions"] = []
    obj_poses["obj_translations"] = []
    obj_poses["obj_rotations"] = []

    for Rot,Tran,sca in zip(all_R, all_T, all_s):
        bbox_extent = np.array([(sca[1,0]-sca[0,0]), (sca[1,1]-sca[0,1]), (sca[1,2]-sca[0,2])])
        cam_t = Tran
        bbox = np.array(sca)
        bbox_diff = bbox[0,2]+((bbox[1,2]-bbox[0,2])/2)
        cam_t[2] += bbox_diff
        cam_rot = np.array(Rot)[:3, :3]

        obj_poses["bbox_dimensions"].append(bbox_extent)
        obj_poses["obj_translations"].append(cam_t)
        obj_poses["obj_rotations"].append(cam_rot)
    return obj_poses

if __name__ == '__main__':
    import os
    import json
    new = True
    # base_dir = '/home/zubair/pd-api-py/multi_neural_rendering_data_test_3'
    # base_dir = '/home/zubairirshad/pd-api-py/PDMultiObjv3/SF_6thAndMission_medium3/val'
    base_dir = '/home/zubairirshad/pd-api-py/PD_v3_eval/test_novelobj/SF_6thAndMission_medium0'
    img_files = os.listdir(os.path.join(base_dir, 'train','rgb'))
    img_files.sort()

    if new:
        all_c2w, focal, img_size, bbox_dimensions, all_rotations, all_translations = read_poses(pose_dir_train = os.path.join(base_dir,'train', 'pose'), img_files_train = img_files)
        all_bboxes = []
        all_T = []
        all_R = []
        print("all_rotations, all_translations, bbox_dimensions", all_rotations, all_translations, bbox_dimensions)
        RTS_raw = {'R': all_rotations, 'T': all_translations, 's': bbox_dimensions}
        obj_poses = preprocess_RTS_for_vis(RTS_raw)

        
        # for i, bbox in enumerate(bbox_dimensions):
        #     v = bbox['asset_name']
        #     scale_factor = np.max(v)
        #     all_bboxes.append(np.array(v)/scale_factor)
        #     all_T.append(np.array(all_translations[i])/scale_factor)
        #     all_R.append(np.array(all_rotations[i]/scale_factor))
        # bbox_dimension_modified = {'R': all_R, 'T': all_T, 's': all_bboxes}
    else:
        all_c2w, focal, img_size, bbox_dimensions = read_poses(pose_dir = os.path.join(base_dir, 'train', 'pose'))
        #all_c2w_test, focal_test, img_size_test, bbox_dimensions_test = read_poses(pose_dir = os.path.join(base_dir,'val', 'pose'))
        bbox_dimensions = np.array(bbox_dimensions)
        bbox_dimension_modified =  bbox_dimensions[1,:] - bbox_dimensions[0,:]
        # scale_factor = np.max(bbox_dimensions)
        # bbox_dimension_modified = bbox_dimensions/scale_factor

    import kornia as kn
    # from numpy import load
    # load dict of arrays
    depth_folder = os.path.join(base_dir, 'train','depth')
    #depth_folder = '/home/zubairirshad/pd-api-py/PDMultiObjv3/train/SF_6thAndMission_medium0/train/depth'
    # depth_paths = [f for f in os.listdir(depth_folder) if f.endswith('.npz')]
    depth_paths = os.listdir(depth_folder)
    depth_paths.sort()
    all_depths = [np.clip(np.load(os.path.join(depth_folder, depth_path), allow_pickle=True)['arr_0'], 0,100) for depth_path in depth_paths]
    # all_depths = [torch.Tensor(x).squeeze(-1).unsqueeze(0).unsqueeze(0) for x in all_depth_path]
    for idx in range(len(all_depths)):
        all_depths[idx][all_depths[idx] >1000.0] = 0.0

    fov = 80
    focal = (640 / 2) / np.tan(( fov/ 2) / (180 / np.pi))
    intrinsics = np.array([
            [focal, 0., 640 / 2.0],
            [0., focal, 480 / 2.0],
            [0., 0., 1.],
        ])
    print("intrrinsics", intrinsics)
    # all_intrinsics = [torch.Tensor(intrinsics).unsqueeze(0) for _ in range(len(all_depths))]
    # all_pcs = [kn.geometry.depth.depth_to_3d(depth, K, normalize_points=False) for (depth,K) in zip(all_depths, all_intrinsics)]

    #depth_pc = all_pcs[0].squeeze().numpy().reshape(3,-1).transpose()

    image_folder = os.path.join(base_dir, 'train','rgb')
    #image_folder = '/home/zubairirshad/pd-api-py/PDMultiObjv3/train/SF_6thAndMission_medium0/train/rgb'
    image_paths = os.listdir(image_folder)
    image_paths.sort()

    all_rgbs = [np.array(Image.open(os.path.join(image_folder, img_path))) for img_path in image_paths]
    #rgb_pc = np.array(all_rgbs[0]).reshape(-1,3)
    
    # depth = load(os.path.join(depth_folder, 'midsize_muscle_01-002.png.npz'))['arr_0']
    # rgb = np.array(Image.open(os.path.join(image_folder, 'midsize_muscle_01-002.png')))
    
    c2w = all_c2w[0]
    
    K_matrix = np.eye(4)
    K_matrix[:3,:3] = intrinsics
    im = all_rgbs[0]
    box_obb = []
    axes = []

    RTs = get_RTs(obj_poses)
    
    all_rotations = RTs['R']
    all_translations = RTs['T']
    bbox_dimensions = RTs['s']
    

    for rotation, translation, size in zip(all_rotations, all_translations, bbox_dimensions):
        print("size", size)
        box = get_3d_bbox(size)
        print("box", box.shape)
        pose = np.eye(4)
        pose[:3,:3] = np.array(rotation)
        pose[:3, 3] = np.array(translation)
        pose = np.linalg.inv(convert_pose(c2w)) @ pose

        unit_box_homopoints = convert_points_to_homopoints(box.T)
        morphed_box_homopoints = pose @ unit_box_homopoints
        rotated_box = convert_homopoints_to_points(morphed_box_homopoints).T
        points_obb = convert_points_to_homopoints(np.array(rotated_box).T)

        box_obb.append(project(K_matrix, points_obb).T)

        xyz_axis = 1.0*np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).transpose()
        print("xyz_axis",xyz_axis.shape)

        transformed_axes = transform_coordinates_3d(xyz_axis, pose)
        projected_axes = calculate_2d_projections(transformed_axes, K_matrix[:3,:3])

        axes.append(projected_axes)

    colors_box = [(63, 234, 237)]
    colors_mpl = ['#EAED3F']

    plt.figure()
    plt.xlim((0, im.shape[1]))
    plt.ylim((0, im.shape[0]))
    plt.gca().invert_yaxis()
    plt.axis('off')
    for k in range(len(colors_box)):
        for points_2d, axis in zip(box_obb, axes):
            print("axis", axis)
            points_2d = np.array(points_2d)
            im = draw_bboxes_mpl_glow(im, points_2d, axis, colors_mpl[k])

    # import cv2
    # cv2.imwrite(im[...,::-1],'boxes.png')
    plt.imshow(im)
    # plt.show()
    plt.savefig('boxes.png', bbox_inches='tight',pad_inches = 0)
    # print("depth, egb", depth.shape, rgb.shape)

    # all_depths = all_depths[100:]
    # all_rgbs = all_rgbs[100:]

    all_depth_pc = []
    all_rgb_pc = []
    for depth, rgb in zip(all_depths, all_rgbs):
        depth_pc, rgb_pc = get_masked_textured_pointclouds(depth, rgb, intrinsics, width=640, height=480)
        all_depth_pc.append(depth_pc)
        all_rgb_pc.append(rgb_pc)

    pcd = o3d.geometry.PointCloud()
    H = 480
    W = 640

    rgb_pc = all_rgb_pc[0]
    depth_pc = all_depth_pc[0]

    xyz_orig = torch.from_numpy(depth_pc).reshape(H,W,3).permute(2,0,1)

    print("len(all_depth_pc)", len(all_depth_pc), len(all_rgb_pc))


    draw_pcd_and_box(xyz_orig, rgb_pc, obj_poses, c2w = c2w)
    draw_combined_pcds_boxes(obj_poses, all_depth_pc, all_rgb_pc, all_c2w)