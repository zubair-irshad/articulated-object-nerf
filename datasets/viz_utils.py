import matplotlib.pyplot as plt
import numpy as np
import torch
import open3d as o3d
import sys
import json
import numpy as np
import pytransform3d.transformations as pt
import pytransform3d.camera as pc
import pytransform3d.visualizer as pv
from .nocs_utils import Pose
from .ray_utils import world_to_ndc
import colorsys
import os
import trimesh
import cv2

def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2


class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.15):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = np.array(
            lines) if lines is not None else self.lines_from_ordered_points(self.points)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length)
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                cylinder_segment = cylinder_segment.rotate(
                    R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a), 
                    center=cylinder_segment.get_center())
                # cylinder_segment = cylinder_segment.rotate(
                #   axis_a, center=True, type=o3d.geometry.RotationType.AxisAngle)
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)

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

def unit_cube():
  points = np.array([
      [0, 0, 0],
      [1, 0, 0],
      [0, 1, 0],
      [1, 1, 0],
      [0, 0, 1],
      [1, 0, 1],
      [0, 1, 1],
      [1, 1, 1],
  ]) - 0.5
  lines = [
      [0, 1],
      [0, 2],
      [1, 3],
      [2, 3],
      [4, 5],
      [4, 6],
      [5, 7],
      [6, 7],
      [0, 4],
      [1, 5],
      [2, 6],
      [3, 7],
  ]

  colors = random_colors(len(lines))
  line_set = LineMesh(points, lines,colors=colors, radius=0.008)
  line_set = line_set.cylinder_segments
  return line_set

def line_set_mesh(points_array):
  open_3d_lines = [
        [0, 1],
        [7,3],
        [1, 3],
        [2, 0],
        [3, 2],
        [0, 4],
        [1, 5],
        [2, 6],
        # [4, 7],
        [7, 6],
        [6, 4],
        [4, 5],
        [5, 7],
    ]
  colors = random_colors(len(open_3d_lines))
  open_3d_lines = np.array(open_3d_lines)
  line_set = LineMesh(points_array, open_3d_lines,colors=colors, radius=0.001)
  line_set = line_set.cylinder_segments
  return line_set

class NOCS_Real():
  def __init__(self, height=480, width=640, scale_factor=1.):
    # This is to go from mmt to pyrender frame
    self.height = int(height / scale_factor)
    self.width = int(width / scale_factor)
    self.f_x = 591.0125
    self.f_y = 590.16775
    self.c_x = 322.525
    self.c_y = 244.11084
    self.stereo_baseline = 0.119559
    self.intrinsics = np.array([
            [self.f_x, 0., self.c_x, 0.0],
            [0., self.f_y, self.c_y, 0.0],
            [0., 0., 1., 0.0],
            [0., 0., 0., 1.],
        ])

x_width = 1.0
y_depth = 1.0
z_height = 1.0
_WORLD_T_POINTS = np.array([
    [0, 0, 0],  #0
    [0, 0, z_height],  #1
    [0, y_depth, 0],  #2
    [0, y_depth, z_height],  #3
    [x_width, 0, 0],  #4
    [x_width, 0, z_height],  #5
    [x_width, y_depth, 0],  #6
    [x_width, y_depth, z_height],  #7
]) - 0.5

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


def convert_homopoints_to_points(points_4d):
  """Project 4d homogenous points (4xN) to 3d points (3xN)"""
  assert len(points_4d.shape) == 2
  assert points_4d.shape[0] == 4
  points_3d = points_4d[:3, :] / points_4d[3:4, :]
  assert points_3d.shape[1] == points_3d.shape[1]
  assert points_3d.shape[0] == 3
  return points_3d


def vis_ray_segmented(masks, class_ids, rays_o, rays_d, img, W, H):
    seg_mask = np.zeros([H, W])
    print("seg_mask", seg_mask.shape)
    for i in range(len(class_ids)):
        # plt.imshow(masks[:,:,i] > 0)
        # plt.show()
        seg_mask[masks[:,:,i] > 0] = np.array(class_ids)[i]
    
    ray_od = torch.stack([rays_o, rays_d], dim=1)
    print("ray_od", ray_od.shape)
    print("img[:, None, :]", img[:, None, :].shape)
    rays_rgb = np.concatenate([ray_od.numpy(), img[:, None, :]], 1)
    rays_rgb_obj = []
    rays_rgb_obj_dir = []
    select_inds=[]
    N_rays=2048

    for i in range(len(class_ids)):
        rays_on_obj = np.where(seg_mask.flatten() == class_ids[i])[0]
        print("rays_on_obj", rays_on_obj.shape)
        rays_on_obj = rays_on_obj[np.random.choice(rays_on_obj.shape[0], N_rays)]
        select_inds.append(rays_on_obj)

        obj_mask = np.zeros(len(rays_rgb), np.bool)
        obj_mask[rays_on_obj] = 1
        # rays_rgb_debug = np.array(rays_rgb)
        # rays_rgb_debug[rays_on_obj, :] += np.random.rand(3) #0.
        # img_sample = np.reshape(rays_rgb_debug[:, 2, :],[H, W, 3])
        # # plt.imshow(img_sample)
        # # plt.show()
        rays_rgb_obj.append(rays_o[rays_on_obj, :])
        rays_rgb_obj_dir.append(rays_d[rays_on_obj, :])

    select_inds = np.concatenate(select_inds, axis=0)
    obj_mask = np.zeros(len(rays_rgb), np.bool)
    obj_mask[select_inds] = 1


    print("select_inds", select_inds.shape, rays_rgb.shape)
    rays_rgb_debug = np.array(rays_rgb)

    rays_rgb_debug[select_inds, :] += np.random.rand(3) #0.
    img_sample = np.reshape(rays_rgb_debug[:, 2, :],[H, W, 3])
    plt.imshow(img_sample)
    plt.show()

    seg_vis = np.zeros([H, W]).flatten()
    seg_vis[obj_mask>0] = 1
    seg_vis = np.reshape(seg_vis, [H,W])
    plt.imshow(seg_vis)
    plt.show()
    return rays_rgb_obj, rays_rgb_obj_dir


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

def get_pointclouds_abspose(pose, pc, is_inverse = False
):
    if is_inverse:
        pose.scale_matrix = np.linalg.inv(pose.scale_matrix)
        pose.camera_T_object = np.linalg.inv(pose.camera_T_object)

    print("pc", pc.shape)
    if is_inverse:
        pc_homopoints = convert_points_to_homopoints(pc.T)
        morphed_pc_homopoints = pose.scale_matrix  @ (pose.camera_T_object @ pc_homopoints)
        morphed_pc_homopoints = convert_homopoints_to_points(morphed_pc_homopoints).T

    else:
        pc_homopoints = convert_points_to_homopoints(pc.T)
        morphed_pc_homopoints = pose.camera_T_object @ (pose.scale_matrix @ pc_homopoints)
        morphed_pc_homopoints = convert_homopoints_to_points(morphed_pc_homopoints).T

    pc_hp = convert_points_to_homopoints(pc.T)
    scaled_homopoints = (pose.scale_matrix @ pc_hp)
    scaled_homopoints = convert_homopoints_to_points(scaled_homopoints).T
    size = 2 * np.amax(np.abs(scaled_homopoints), axis=0)
    box = get_3d_bbox(size)
    unit_box_homopoints = convert_points_to_homopoints(box.T)
    morphed_box_homopoints = pose.camera_T_object @ unit_box_homopoints
    morphed_box_points = convert_homopoints_to_points(morphed_box_homopoints).T
    return morphed_pc_homopoints, morphed_box_points, size


def transform_rays_w2o(pose, pc, scale_matrix, is_inverse = False
):
    if is_inverse:
        scale_matrix[0,0] = 1/scale_matrix[0,0]
        scale_matrix[1,1] = 1/scale_matrix[1,1]
        scale_matrix[2,2] = 1/scale_matrix[2,2]
        pose = np.linalg.inv(pose)

    if is_inverse:
        pc_homopoints = convert_points_to_homopoints(pc.T)
        morphed_pc_homopoints = scale_matrix  @ (pose @ pc_homopoints)
        #morphed_pc_homopoints = (pose @ pc_homopoints)
        morphed_pc_homopoints = convert_homopoints_to_points(morphed_pc_homopoints).T

    else:
        pc_homopoints = convert_points_to_homopoints(pc.T)
        morphed_pc_homopoints =  pose @ (scale_matrix @ pc_homopoints)
        #morphed_pc_homopoints = pose @ ( pc_homopoints)
        morphed_pc_homopoints = convert_homopoints_to_points(morphed_pc_homopoints).T

    return morphed_pc_homopoints

def get_gt_pointclouds(pose, pc
):
    pc_homopoints = convert_points_to_homopoints(pc.T)
    unit_box_homopoints = convert_points_to_homopoints(_WORLD_T_POINTS.T)
    morphed_pc_homopoints = pose.camera_T_object @ (pose.scale_matrix @ pc_homopoints)
    morphed_box_homopoints = pose.camera_T_object @ (pose.scale_matrix @ unit_box_homopoints)
  
    morphed_pc_homopoints = convert_homopoints_to_points(morphed_pc_homopoints).T
    morphed_box_points = convert_homopoints_to_points(morphed_box_homopoints).T
    # box_points.append(morphed_box_points)
    return morphed_pc_homopoints, morphed_box_points


def viz_pcd_out(model_points, abs_poses):

    rotated_pc_o3d = []
    rotated_pc_box = []
    for i in range(len(model_points)):
        rotated_pc, rotated_box = get_gt_pointclouds(abs_poses[i], model_points[i])
        rotated_pc_o3d.append(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(rotated_pc)))
        rotated_pc_box.append(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(rotated_box)))
    o3d.visualization.draw_geometries(rotated_pc_o3d)


def transform_rays_orig(rays_o, directions, c2w, scale_matrix):
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:3, :3].T # (H, W, 3)
    #rays_d /= torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = rays_o + np.broadcast_to(c2w[:, 3][:3], rays_d.shape)  # (H, W, 3)
    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)
    #print("rays_o", rays_o.shape, np.linalg.inv(scale_matrix).shape, np.linalg.inv(scale_matrix))
    # rays_o = ( rays_o @ torch.from_numpy(np.linalg.inv(scale_matrix))[:3,:3])
    # rays_d = ( rays_d @ torch.from_numpy(np.linalg.inv(scale_matrix))[:3, :3] )

    
    rays_d /= torch.norm(rays_d, dim=-1, keepdim=True)
    return rays_o, rays_d


def transform_rays(rays_o, directions, o2w):
    # Rotate ray directions from camera coordinate to the world coordinate

    print("directions", directions.shape)
    rays_d = directions @ o2w[:3, :3] # (H, W, 3)
    rays_d = rays_d.view(-1, 3)
    rays_d /= torch.norm(rays_d, dim=-1, keepdim=True)

    # Shift the camera rays to object center location in world frame and rotate by o2w
    rays_o = rays_o - np.broadcast_to(o2w[:, 3][:3], rays_d.shape)
    rays_o = rays_o @ o2w[:3, :3] # (H, W, 3)
    rays_o = rays_o.view(-1, 3)

    print("rays_d", rays_d.shape)
    print("o2w[:, 3]", o2w[:, 3].shape)

    return rays_o, rays_d

def convert_pose(C2W):
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = np.matmul(C2W, flip_yz)
    return C2W

def plot_NDC_trajectory(model_points, abs_poses, camera_poses, rays_o_all, rays_d_all, num, W, H, focal):
    
    camera_real = NOCS_Real()
    M = camera_real.intrinsics[:3,:3]
    sensor_size = (float(camera_real.width), float(camera_real.height))
    transformation_matrices = np.empty((len(camera_poses), 4, 4))
    for i, camera_pose in enumerate(camera_poses):
        R = camera_pose[:, :3] 
        p = camera_pose[:, 3] 
        transformation_matrices[i] = pt.transform_from(R=R, p=p)
    rotated_pcds = []
    rotated_boxes = []
    mesh_frames = []
    objs2world = []
    for i in range(len(model_points)):
        flip_yz = np.eye(4)
        flip_yz[1, 1] = -1
        flip_yz[2, 2] = -1
        print(" model_points[i]",  model_points[i].shape)
        # opengl_pose = convert_pose(transformation_matrices[num])
        #obj2world = opengl_pose @ abs_poses[i].camera_T_object
        obj2world = transformation_matrices[num] @ np.linalg.inv(flip_yz) @ abs_poses[i].camera_T_object
        #obj2world = np.linalg.inv(flip_yz) @ abs_poses[i].camera_T_object
        obj2world = Pose(camera_T_object=obj2world, scale_matrix=abs_poses[i].scale_matrix)
        rotated_pc, rotated_box, _ = get_pointclouds_abspose(obj2world, model_points[i])
        rotated_pcds.append(rotated_pc)
        rotated_boxes.append(rotated_box)
        T =  obj2world.camera_T_object
        objs2world.append(T)
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        mesh_frame = mesh_frame.transform(T)
        mesh_frames.append(mesh_frame)

    rotated_pcds_ndc = []
    rotated_boxes_ndc = []
    for rotated_pcd, rotated_box in zip(rotated_pcds, rotated_boxes):
        ndc_pcd = world_to_ndc(rotated_pcd, W, H, focal, near=1.0)
        ndc_box = world_to_ndc(rotated_box, W, H, focal, near=1.0)
        rotated_pcds_ndc.append(ndc_pcd)
        rotated_boxes_ndc.append(ndc_box)

    fig = pv.figure()
    # # Add geometries for mesh frame, linset and rotated pcds
    for mesh_frame in mesh_frames:
        fig.add_geometry(mesh_frame)
    for pcds in rotated_pcds:
        fig.add_geometry(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcds)))
    for pcds in rotated_pcds_ndc:
        fig.add_geometry(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcds)))
    for rotated_box in rotated_boxes:
        cylinder_segments = line_set_mesh(rotated_box)
        for k in range(len(cylinder_segments)):
            fig.add_geometry(cylinder_segments[k])

    for rotated_box in rotated_boxes_ndc:
        cylinder_segments = line_set_mesh(rotated_box)
        for k in range(len(cylinder_segments)):
            fig.add_geometry(cylinder_segments[k])

    # Plot rays for each obj
    for rays_o, rays_d in zip(rays_o_all, rays_d_all):
        print("OBJ RAYS", rays_o.shape, rays_d.shape)
        for j in range(300):
            start = rays_o[j,:]
            end = rays_o[j,:] + rays_d[j,:]*1
            line = np.concatenate((start[None, :],end[None, :]), axis=0)
            fig.plot(line, c=(1.0, 0.5, 0.0))
    unitcube = unit_cube()
    for k in range(len(unitcube)):
        fig.add_geometry(unitcube[k]) 

    # Plot origin cameras
    R = np.zeros((3,3))
    p= np.zeros((3))
    origin = pt.transform_from(R=R, p=p)
    fig.plot_camera(M=M, virtual_image_distance=0.1, sensor_size=sensor_size)
    # fig.plot_transform(s=0.1)

    fig.plot_camera(M=M, cam2world=transformation_matrices[0], virtual_image_distance=0.1, sensor_size=sensor_size)
    fig.plot_transform(A2B=transformation_matrices[0], s=0.1)


    # Plot all camera trajectories
    for pose in transformation_matrices:
        fig.plot_transform(A2B=pose, s=0.1)
        fig.plot_camera(M=M, cam2world=pose, virtual_image_distance=0.1, sensor_size=sensor_size)
    fig.show()

def plot_camera_trajectory(model_points, abs_poses, camera_poses, rays_o_all, rays_d_all, num):
    
    camera_real = NOCS_Real()
    M = camera_real.intrinsics[:3,:3]
    sensor_size = (float(camera_real.width), float(camera_real.height))
    transformation_matrices = np.empty((len(camera_poses), 4, 4))
    for i, camera_pose in enumerate(camera_poses):
        R = camera_pose[:, :3] 
        p = camera_pose[:, 3] 
        transformation_matrices[i] = pt.transform_from(R=R, p=p)
    rotated_pcds = []
    rotated_boxes = []
    mesh_frames = []
    objs2world = []
    for i in range(len(model_points)):
        flip_yz = np.eye(4)
        flip_yz[1, 1] = -1
        flip_yz[2, 2] = -1
        print(" model_points[i]",  model_points[i].shape)
        # opengl_pose = convert_pose(transformation_matrices[num])
        #obj2world = opengl_pose @ abs_poses[i].camera_T_object
        obj2world = transformation_matrices[num] @ np.linalg.inv(flip_yz) @ abs_poses[i].camera_T_object
        #obj2world = np.linalg.inv(flip_yz) @ abs_poses[i].camera_T_object
        obj2world = Pose(camera_T_object=obj2world, scale_matrix=abs_poses[i].scale_matrix)
        rotated_pc, rotated_box, _ = get_pointclouds_abspose(obj2world, model_points[i])
        rotated_pcds.append(rotated_pc)
        rotated_boxes.append(rotated_box)
        T =  obj2world.camera_T_object
        objs2world.append(T)
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        mesh_frame = mesh_frame.transform(T)
        mesh_frames.append(mesh_frame)

    fig = pv.figure()
    # # Add geometries for mesh frame, linset and rotated pcds
    for mesh_frame in mesh_frames:
        fig.add_geometry(mesh_frame)
    for pcds in rotated_pcds:
        fig.add_geometry(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcds)))
    for rotated_box in rotated_boxes:
        cylinder_segments = line_set_mesh(rotated_box)
        for k in range(len(cylinder_segments)):
            fig.add_geometry(cylinder_segments[k])
    # Plot rays for each obj
    for rays_o, rays_d in zip(rays_o_all, rays_d_all):
        print("OBJ RAYS", rays_o.shape, rays_d.shape)
        for j in range(300):
            start = rays_o[j,:]
            end = rays_o[j,:] + rays_d[j,:]*10
            line = np.concatenate((start[None, :],end[None, :]), axis=0)
            fig.plot(line, c=(1.0, 0.5, 0.0))
    unitcube = unit_cube()
    for k in range(len(unitcube)):
        fig.add_geometry(unitcube[k]) 

    # Plot origin cameras
    R = np.zeros((3,3))
    p= np.zeros((3))
    origin = pt.transform_from(R=R, p=p)
    fig.plot_camera(M=M, virtual_image_distance=0.1, sensor_size=sensor_size)
    # fig.plot_transform(s=0.1)

    fig.plot_camera(M=M, cam2world=transformation_matrices[0], virtual_image_distance=0.1, sensor_size=sensor_size)
    fig.plot_transform(A2B=transformation_matrices[0], s=0.1)


    # Plot all camera trajectories
    for pose in transformation_matrices:
        fig.plot_transform(A2B=pose, s=0.1)
        fig.plot_camera(M=M, cam2world=pose, virtual_image_distance=0.1, sensor_size=sensor_size)
    fig.show()

def plot_canonical_pcds(model_points, abs_poses, camera_poses, rays_o_all, rays_d_all, num):
    
    camera_real = NOCS_Real()
    M = camera_real.intrinsics[:3,:3]
    sensor_size = (float(camera_real.width), float(camera_real.height))

    transformation_matrices = np.empty((len(camera_poses), 4, 4))
    for i, camera_pose in enumerate(camera_poses):
        R = camera_pose[:, :3] 
        p = camera_pose[:, 3] 
        transformation_matrices[i] = pt.transform_from(R=R, p=p)

    rotated_pcds = []
    rotated_boxes = []
    mesh_frames = []
    objs2world = []
    for i in range(len(model_points)):
        flip_yz = np.eye(4)
        flip_yz[1, 1] = -1
        flip_yz[2, 2] = -1
        print(" model_points[i]",  model_points[i].shape)
        # opengl_pose = convert_pose(transformation_matrices[num])
        #obj2world = opengl_pose @ abs_poses[i].camera_T_object
        obj2world = transformation_matrices[num] @ np.linalg.inv(flip_yz) @ abs_poses[i].camera_T_object
        #obj2world = np.linalg.inv(flip_yz) @ abs_poses[i].camera_T_object
        obj2world = Pose(camera_T_object=obj2world, scale_matrix=abs_poses[i].scale_matrix)
        rotated_pc, rotated_box, _ = get_pointclouds_abspose(obj2world, model_points[i])

        rotated_pcds.append(rotated_pc)
        rotated_boxes.append(rotated_box)

        T =  obj2world.camera_T_object
        objs2world.append(T)

        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        mesh_frame = mesh_frame.transform(T)
        # mesh_frame = mesh_frame.transform(obj2world.camera_T_object)
        # mesh_frame = mesh_frame.transform(np.linalg.inv(flip_yz))
        mesh_frames.append(mesh_frame)

    id_num = 1
    # transform back to canonical frame for one object
    canonical_pcds = []
    for i in range(len(rotated_pcds)):
        print("rotated_pcds[i]", rotated_pcds[1].shape)
        # opengl_pose = convert_pose(transformation_matrices[num])
        # obj2world = opengl_pose @ abs_poses[0].camera_T_object
        obj2world = transformation_matrices[num] @ np.linalg.inv(flip_yz) @ abs_poses[0].camera_T_object
        #obj2world = np.linalg.inv(flip_yz) @ abs_poses[i].camera_T_object
        obj2world = convert_pose(obj2world)

        print("IS EQUALLLL", np.equal (obj2world, abs_poses[i].camera_T_object))

        print("np.linalg.inv(flip_yz)", np.linalg.inv(flip_yz) @ flip_yz)
        obj2world = Pose(camera_T_object=obj2world, scale_matrix=abs_poses[i].scale_matrix)
        canonical_pc, canonical_box, _ = get_pointclouds_abspose(obj2world, rotated_pcds[i], is_inverse=True)
        canonical_pcds.append(canonical_pc)

    # transform rays to caonical frames
    canonical_rays_all = []
    canonical_rays_d_all = []
    for ray_id, (rays_o, rays_d) in enumerate(zip(rays_o_all, rays_d_all)):
        if ray_id ==id_num:
            for pose_id in range(len(abs_poses)):
                # obj2world = transformation_matrices[num] @ np.linalg.inv(flip_yz) @ abs_poses[0].camera_T_object
                obj2world = np.linalg.inv(flip_yz) @ abs_poses[pose_id].camera_T_object
                obj2world = convert_pose(obj2world)
                # obj2world = np.linalg.inv(objs2world)
                scale_matrix=abs_poses[pose_id].scale_matrix
                # transform_rays_orig()
                #canonical_rays = transform_rays_w2o(obj2world, rays_o, scale_matrix,  is_inverse=False)
                #canonical_rays_d = transform_rays_w2o(obj2world, rays_d, scale_matrix,  is_inverse=False)
                # canonical_rays_d /= np.linalg.norm(canonical_rays_d, axis=-1, keepdims=True)

                canonical_rays, canonical_rays_d = transform_rays_orig(rays_o, rays_d, obj2world, scale_matrix)
                canonical_rays_all.append(canonical_rays)
                canonical_rays_d_all.append(canonical_rays_d)

    fig = pv.figure()
    # # Add geometries for mesh frame, linset and rotated pcds
    for mesh_frame in mesh_frames:
        fig.add_geometry(mesh_frame)
    for pcds in rotated_pcds:
        fig.add_geometry(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcds)))
    for rotated_box in rotated_boxes:
        cylinder_segments = line_set_mesh(rotated_box)
        for k in range(len(cylinder_segments)):
            fig.add_geometry(cylinder_segments[k])
    # Plot rays for each obj
    for rays_o, rays_d in zip(rays_o_all, rays_d_all):
        print("OBJ RAYS", rays_o.shape, rays_d.shape)
        for j in range(300):
            start = rays_o[j,:]
            end = rays_o[j,:] + rays_d[j,:]*1.5
            line = np.concatenate((start[None, :],end[None, :]), axis=0)
            fig.plot(line, c=(1.0, 0.5, 0.0))

    #Plot canonical pcds, unit cube and caonical rays
    # for canonical_pc in canonical_pcds:
    #     fig.add_geometry(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(canonical_pc)))
    # unitcube = unit_cube()
    # for canonical_pc in canonical_pcds:
    #     for k in range(len(unitcube)):
    #         fig.add_geometry(unitcube[k]) 

    # for iter, (rays_o, rays_d) in enumerate(zip(canonical_rays_all, canonical_rays_d_all)):
    #     print("OBJ RAYS", rays_o.shape, rays_d.shape)
    #     # if iter==0:
    #     for j in range(300):
    #         start = rays_o[j,:]
    #         end = rays_o[j,:] + rays_d[j,:]*1.5
    #         line = np.concatenate((start[None, :],end[None, :]), axis=0)
    #         fig.plot(line, c=(1.0, 0.5, 0.0))

    # plot camera for all meshes:

    for obj2world in objs2world:
        obj2world_opengl = convert_pose(obj2world)
        # obj2world_opengl = np.linalg.inv(objs2world[0])
        # plot camera for first mesh
        fig.plot_camera(M=M, cam2world=obj2world_opengl, virtual_image_distance=0.1, sensor_size=sensor_size)
        fig.plot_transform(A2B=obj2world_opengl, s=0.3)

    # Plot origin cameras
    R = np.zeros((3,3))
    p= np.zeros((3))
    origin = pt.transform_from(R=R, p=p)
    fig.plot_camera(M=M, virtual_image_distance=0.1, sensor_size=sensor_size)
    # fig.plot_transform(s=0.1)

    fig.plot_camera(M=M, cam2world=transformation_matrices[0], virtual_image_distance=0.1, sensor_size=sensor_size)
    fig.plot_transform(A2B=transformation_matrices[0], s=0.1)


    # Plot all camera trajectories
    # for pose in transformation_matrices:
    #     fig.plot_transform(A2B=pose, s=0.1)
    #     fig.plot_camera(M=M, cam2world=pose, virtual_image_distance=0.1, sensor_size=sensor_size)
    fig.show()

def get_trimesh_scales(obj_trimesh
):
    bounding_box = obj_trimesh.bounds
    current_scale = np.array([
        bounding_box[1][0] - bounding_box[0][0], bounding_box[1][1] - bounding_box[0][1],
        bounding_box[1][2] - bounding_box[0][2]
    ])
    return current_scale

def get_3D_rotated_box(pose, sizes
):
    box = get_3d_bbox(sizes)
    unit_box_homopoints = convert_points_to_homopoints(box.T)
    morphed_box_homopoints = pose.camera_T_object @ (pose.scale_matrix @ unit_box_homopoints)
    morphed_box_points = convert_homopoints_to_points(morphed_box_homopoints).T
    return morphed_box_points

def project(K, p_3d):
    projections_2d = np.zeros((2, p_3d.shape[1]), dtype='float32')
    p_2d = np.dot(K, p_3d)
    projections_2d[0, :] = p_2d[0, :]/p_2d[2, :]
    projections_2d[1, :] = p_2d[1, :]/p_2d[2, :]
    return projections_2d

def draw_saved_mesh_and_pose(abs_pose_outputs, model_lists, color_img):
    mesh_dir_path = '/home/zubair/Downloads/nocs_data/obj_models_transformed/real_test'
    rotated_meshes = []
    pcds = []
    camera = NOCS_Real()
    for j in range(len(abs_pose_outputs)):
        mesh_file_name = os.path.join(mesh_dir_path, model_lists[j]+'.obj')
        obj_trimesh = trimesh.load(mesh_file_name)
        sizes = get_trimesh_scales(obj_trimesh)
        obj_trimesh.apply_transform(abs_pose_outputs[j].scale_matrix)
        obj_trimesh.apply_transform(abs_pose_outputs[j].camera_T_object)
        obj_trimesh = obj_trimesh.as_open3d
        obj_trimesh.compute_vertex_normals()

        # single_mesh = obj_trimesh.as_open3d
        single_pcd = obj_trimesh.sample_points_uniformly(200000)
        points_mesh = convert_points_to_homopoints(np.array(single_pcd.points).T)
        points_2d_mesh = project(camera.intrinsics, points_mesh)
        points_2d_mesh = points_2d_mesh.T
        colors = []
        print(points_2d_mesh.shape)
        for k in range(points_2d_mesh.shape[0]):
            # im = Image.fromarray(np.uint8(color_img/255.0))
            color = color_img.getpixel((int(points_2d_mesh[k,0]), int(points_2d_mesh[k,1])))
            color = np.array(color)
            colors.append(color/255.0)
        single_pcd.colors  = o3d.utility.Vector3dVector(np.array(colors))
        single_pcd.normals = o3d.utility.Vector3dVector(np.zeros(
            (1, 3)))  # invalidate existing normals
        pcds.append(single_pcd)
        # o3d.visualization.draw_geometries([single_pcd])

        box_3D = get_3D_rotated_box(abs_pose_outputs[j], sizes)
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        T = abs_pose_outputs[j].camera_T_object
        mesh_frame = mesh_frame.transform(T)
        rotated_meshes.append(obj_trimesh)
        rotated_meshes.append(mesh_frame)
        pcds.append(mesh_frame)

        cylinder_segments = line_set_mesh(box_3D)
        for k in range(len(cylinder_segments)):
            rotated_meshes.append(cylinder_segments[k])
            pcds.append(cylinder_segments[k])

    unitcube = unit_cube()
    for k in range(len(unitcube)):
        rotated_meshes.append(unitcube[k])
        pcds.append(unitcube[k])
        # fig.add_geometry(unitcube[k]) 

    #custom_draw_geometry_with_rotation(pcds, 4)
    o3d.visualization.draw_geometries(pcds)
    #custom_draw_geometry_with_rotation(pcds, 5)
    o3d.visualization.draw_geometries(rotated_meshes)
    #custom_draw_geometry_with_rotation(rotated_meshes, 5)
    return pcds


def transform_coordinates_3d(coordinates, RT):
  """
  Input: 
    coordinates: [3, N]
    RT: [4, 4]
  Return 
    new_coordinates: [3, N]

  """
  assert RT.shape == (4, 4)
  assert len(coordinates.shape) == 2
  assert coordinates.shape[0] == 3
  coordinates = np.vstack([coordinates, np.ones((1, coordinates.shape[1]), dtype=np.float32)])
  new_coordinates = RT @ coordinates
  new_coordinates = new_coordinates[:3, :] / new_coordinates[3, :]
  assert new_coordinates.shape[0] == 3
  return new_coordinates

def calculate_2d_projections(coordinates_3d, intrinsics):
    """
    Input: 
        coordinates: [3, N]
        intrinsics: [3, 3]
    Return 
        projected_coordinates: [N, 2]
    """
    projected_coordinates = intrinsics @ coordinates_3d
    projected_coordinates = projected_coordinates[:2, :] / projected_coordinates[2, :]
    projected_coordinates = projected_coordinates.transpose()
    projected_coordinates = np.array(projected_coordinates, dtype=np.int32)

    return projected_coordinates

def draw_bboxes_mpl_glow(img, img_pts, axes, color):

    img = cv2.arrowedLine(img, tuple(axes[0]), tuple(axes[1]), (0, 0, 255), 4)
    img = cv2.arrowedLine(img, tuple(axes[0]), tuple(axes[3]), (255, 0, 0), 4)
    img = cv2.arrowedLine(img, tuple(axes[0]), tuple(axes[2]), (0, 255, 0), 4) ## y last

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


def plot_3d_bbox(object_pose_and_size, img_vis, output_path, Tc2_c1, c2w = None):
    
    # img_vis = img_vis[:,:,::-1]
    box_obb = []
    axes = []
    colors_box = [(234, 237, 63)]
    colors_mpl = ['#08F7FE']

    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1

    # if Tc2_c1 is not None:
    #     Tc2_c1 = np.concatenate((Tc2_c1, np.array([[0,0,0,1]])), axis=0)
    
    # if c2w is not None:
    #     c2w = np.concatenate((c2w, np.array([[0,0,0,1]])), axis=0)
    #print("c2w", c2w.shape)
    camera = NOCS_Real()
    for obj_id, pose_and_size in object_pose_and_size.items():
        size = pose_and_size['size']
        pose = pose_and_size['pose']
        # print("c2w", c2w.shape)
        # print("pose", pose, pose.shape)
        # axis_align_mat = pose*4.4
        # axis_align_mat[3,3] = 1
        # obj2world = c2w @ np.linalg.inv(flip_yz) @ axis_align_mat
        # pose = obj2world/4.4
        # pose[3,3] = 1
        # print("pose", pose, pose.shape)

        # new_pose = Tc2_c1 @ np.linalg.inv(flip_yz) @ pose
        # new_pose = convert_pose(new_pose)
        
        # Tc2_c1[]
        # new_pose = Tc2_c1 @ pose
        # pose = new_pose
        # opengl_pose = convert_pose(Tc2_c1)

        # obj2world = Tc2_c1 @ np.linalg.inv(flip_yz) @ pose
        # pose = convert_pose(obj2world)

        # # print("pose, Tc2_c1", pose, Tc2_c1)

        # # pose[:3,3] += pose[:3, :3] @ Tc2_c1
        # # pose = opengl_pose @ pose
        # print("pose", pose)
        #pose = Tc2_c1 @ pose
        # pose = Tc2_c1 @ pose 
        # print("pose after", pose)

        # print("================\n")
        # print("pose", pose)
        # pose = np.linalg.inv(np.linalg.inv(pose) @Tc2_c1)
        # print("pose", pose)
        box = get_3d_bbox(size)
        unit_box_homopoints = convert_points_to_homopoints(box.T)
        morphed_box_homopoints = pose @ unit_box_homopoints
        morphed_box_points = convert_homopoints_to_points(morphed_box_homopoints).T
        
        #2D output        
        points_obb = convert_points_to_homopoints(np.array(morphed_box_points).T)
        points_2d_obb = project(camera.intrinsics, points_obb)
        points_2d_obb = points_2d_obb.T
        box_obb.append(points_2d_obb)
        xyz_axis = 0.3*np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).transpose()
        scale = np.eye(4)
        scale[:3,:3] = np.eye(3) *0.2
        sRT = pose @ scale
        transformed_axes = transform_coordinates_3d(xyz_axis, sRT)
        projected_axes = calculate_2d_projections(transformed_axes, camera.intrinsics[:3,:3])
        axes.append(projected_axes)

        im = np.array(np.copy(img_vis)).copy()
        plt.figure()
        plt.xlim((0, im.shape[1]))
        plt.ylim((0, im.shape[0]))
        plt.gca().invert_yaxis()
        plt.axis('off')
        # for k in range(len(colors_box)):
        #     for points_2d, axis in zip(box_obb, axes):
        #         points_2d = np.array(points_2d)
        #         im = draw_bboxes_mpl_glow(im, points_2d, axis, colors_mpl[k])
        plt.imshow(im)
        # box_plot_name = str(output_path)+'/box3d_'+plot_name+str(num)+'.png'
        plt.savefig(output_path, bbox_inches='tight',pad_inches = 0)
        # plt.close()

