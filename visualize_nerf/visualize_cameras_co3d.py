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
import cv2
from PIL import Image
from datasets.ray_utils import homogenise_np
from datasets.ray_utils import world_to_ndc, get_ray_bbox_intersections, get_rays_in_bbox
import torchvision.transforms as T
from scipy.spatial.transform import Rotation

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


def visualize_cameras(ds, cameras,bbox, color, sphere_radius, camera_size=0.1, geometry_file=None):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius, resolution=10)
    sphere = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
    sphere.paint_uniform_color((1, 0, 0))

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0., 0., 0.])
    things_to_draw = [sphere, coord_frame]
    
    RT = bbox['RT']
    
    size = bbox['s']

    print("size", size)
    scale_factor = np.max(bbox['s'])
    size = size/scale_factor

    bbox_canonical = o3d.geometry.OrientedBoundingBox(center = [0,0,0], R = np.eye(3), extent=size)
    things_to_draw.append(bbox_canonical)
    idx = 0
    frustums = []
    for i, camera in enumerate(cameras):
        C2W = camera.get_pose_matrix().squeeze().cpu().numpy()
        print("C2W", C2W.shape, C2W)
        C2W[:3,3] /=scale_factor
        print("C2W after", C2W)
        C2W = convert_pose(C2W)
        K = camera.get_intrinsics().squeeze().cpu().numpy()
        H, W = camera.get_image_size()[0].cpu().numpy()
        im_data = ds.images[i]
        # print("im_data", im_data.shape)

        downsample_factor = 4

        H_d,W_d = int(H/downsample_factor), int(W/downsample_factor)
        resize_transform = T.Resize((H_d, W_d), antialias=True)
        img_resized = resize_transform(im_data.permute(2,0,1))
        # print("img_resized", img_resized.shape)

        # plt.imshow(img_resized.permute(1,2,0).cpu().numpy())
        # plt.show()

        mask = ds.co3d_masks[i]
        print(mask.shape)

        mask_resized = resize_transform(mask)
        print("mask_resized", mask_resized.shape)

        instance_mask = mask_resized>0
        # print("instance_mask", instance_mask.shape)

        # print("instance_mask VIEW", instance_mask.view(-1).shape)
        # plt.imshow(mask_resized.squeeze(0).cpu().numpy())
        # plt.colorbar()
        # plt.show()

        # print("img sizw",H,W)
        img_size = (H,W)
        focal = K[0,0]
        frustums.append(get_camera_frustum(img_size, focal, C2W, frustum_length=camera_size, color=color))
        # cnt += 1

    # print("frustums", len(frustums))
    cameras_frustum = frustums2lineset(frustums)
    things_to_draw.append(cameras_frustum)

    idx = 12
    camera = cameras[10]
    c2w = camera.get_pose_matrix().squeeze().cpu().numpy()
    c2w[:3,3] /=scale_factor
    c2w = convert_pose(c2w)
    c2w = torch.FloatTensor(c2w)[:3, :4]
    focal = camera.get_intrinsics().squeeze().cpu().numpy()[0,0]
    H, W = camera.get_image_size()[0].cpu().numpy()
    directions = get_ray_directions(H, W, focal) # (h, w, 3)
    print("directions", directions.shape)
    # c2w = np.linalg.inv(all_w2c[30])

    rays_o, rays_d = get_rays(directions, c2w)

    rays_o = rays_o.numpy()
    rays_d = rays_d.numpy()

    bbox['s'] = size
    rays_o, rays_d, batch_near, batch_far = get_object_rays_in_bbox(rays_o, rays_d, bbox, canonical=True)
    fig = pv.figure()
    ids = np.random.choice(rays_o.shape[0], int(rays_o.shape[0]*0.5))

    rays_o = rays_o[ids, :]
    rays_d = rays_d[ids, :]
    batch_near = batch_near[ids]
    batch_far = batch_far[ids]

    print("batch near", batch_near)
    print("batch far", batch_far)
    #print(batch_near, batch_far)
    for j in range(200):
        start = rays_o[j,:]
        end = rays_o[j,:] + rays_d[j,:]*batch_near[j, :].cpu().numpy()
        line = np.concatenate((start[None, :],end[None, :]), axis=0)
        fig.plot(line, c=(1.0, 0.5, 0.0))

        start = rays_o[j,:] + rays_d[j,:]*batch_near[j, :].cpu().numpy()
        end = rays_o[j,:] + rays_d[j,:]*batch_far[j, :].cpu().numpy()
        line = np.concatenate((start[None, :],end[None, :]), axis=0)
        fig.plot(line, c=(0.0, 1.0, 0.0))

    geometry_file = geometry_file/scale_factor
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(geometry_file)

    things_to_draw.append(pcd)

    all_c2w = []
    for camera in cameras:
        c2w = camera.get_pose_matrix().squeeze().cpu().numpy()
        c2w = convert_pose(c2w)
        c2w[:3,3] /= scale_factor
        all_c2w.append(c2w)
        # print("C2W", C2W.shape)
        #fig.plot_transform(A2B=np.array(c2w), s=0.1, strict_check=False)

    # generate test time object cameras around z-axis:
    # Draw test time novel trajectories here
    # axis_aligned_inv = np.eye(4)
    # total_frames = 90
    # test_idx = 12
    # all_object_pose = []
    # frustums = []
    # for idx in range(total_frames):
    #     progress = idx / total_frames
    #     # #rotation transform
    #     # trans_pose = get_pure_rotation(progress_11=(progress * 2 - 1))
    #     # transform = np.linalg.inv(axis_aligned_inv) @ trans_pose @ axis_aligned_inv
    #     # transform = np.linalg.inv(transform)
    #     # transform = transform @ all_c2w[test_idx] 

    #     # #translation transform
    #     # translation_transform = get_pure_translation(progress_11=(progress * 2 - 1))
    #     # translation_transform = np.linalg.inv(translation_transform)
    #     # translation_transform = translation_transform @ all_c2w[test_idx]
        
    #     # fig.plot_transform(A2B=np.array(translation_transform), s=0.1, strict_check=False)
    #     #translation + rotation transform
    #     t1 = 40
    #     t2 = 60
    #     t3 = total_frames
    #     if idx <=t1:
    #         parallel_park_transform_translation = get_pure_translation(progress_11=(progress * 2 - 1), axis='x', max_distance = 2)
    #         parallel_park_transform_translation_inv = np.linalg.inv(parallel_park_transform_translation)
    #         parallel_park_transform = parallel_park_transform_translation_inv @ all_c2w[test_idx]

    #     elif idx > t1 and idx <=t2:
    #         progress_rotation = (idx-t1) / (t2-t1)
    #         parallel_park_transform_rotation = get_pure_rotation(progress_11=(-progress_rotation), max_angle= 20)
    #         parallel_park_transform_rotation = parallel_park_transform_rotation @ parallel_park_transform_translation
    #         parallel_park_transform_rotation_inv = np.linalg.inv(parallel_park_transform_rotation)
    #         parallel_park_transform = parallel_park_transform_rotation_inv @ all_c2w[test_idx]

    #     elif idx > t2 and idx < t3:
    #         progress_translation = (idx-t2) / (t3-t2)
    #         parallel_park_transform_translation2 = get_pure_translation(progress_11=(progress_translation), axis='y', max_distance = 1)
    #         parallel_park_transform_translation2 = parallel_park_transform_translation2 @ parallel_park_transform_rotation
    #         parallel_park_transform_translation2_inv = np.linalg.inv(parallel_park_transform_translation2)
    #         parallel_park_transform = parallel_park_transform_translation2_inv @ all_c2w[test_idx]

    #     fig.plot_transform(A2B=np.array(parallel_park_transform), s=0.1, strict_check=False)
    
    # cameras_frustum = frustums2lineset(frustums)
    # things_to_draw.append(cameras_frustum)

    for geometry in things_to_draw:
        fig.add_geometry(geometry)
        # all_object_pose.append(trans_pose)

    fig.show()

    # o3d.visualization.draw_geometries(things_to_draw)

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

if __name__ == '__main__':
    import os

    from datasets.co3d import Dataset as Dataset_Co3D

    device = torch.device('cuda')

    ds = Dataset_Co3D(data_dir = '/home/zubair/snes/data/co3d', category = 'car', instance = '106_12662_23043',  device=device)
    train_cameras = ds.get_cameras()

    sphere_radius = 1.
    camera_size = 0.1

    pts = ds.point_cloud_xyz.squeeze().numpy()
    rgb = ds.point_cloud_rgb.squeeze().numpy()
    if ds.global_alignment is not None:
        canonical_alignment = ds.global_alignment.cpu().numpy()
        pts = np.einsum('ji,ni->nj', canonical_alignment, homogenise_np(pts))[:, :3]

        mask = np.all(
            np.concatenate(
                [pts >= ds.object_bbox_min, pts <= ds.object_bbox_max],
                axis=1),
            axis=1)

        rotation = ds.RT[:3,:3]
        translation = ds.RT[:3, 3]
        size = ds.size
        bbox= {'RT': ds.RT, 's': size}
        # pts = pts[mask, :]
        # rgb = rgb[mask, :]

    visualize_cameras(ds, train_cameras, bbox, [0, 0, 1], sphere_radius, 
                      camera_size=camera_size, geometry_file=pts)