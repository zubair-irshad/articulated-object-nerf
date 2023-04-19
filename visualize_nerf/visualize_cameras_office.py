import open3d as o3d
import json
import numpy as np
import json

import torch
from kornia import create_meshgrid
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.visualizer as pv
from datasets.google_scanned_utils import *
import cv2
from PIL import Image
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



def get_camera_frustum(img_size, focal, W2C, frustum_length=1, color=[0., 1., 0.]):
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
    C2W = W2C
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

def visualize_cameras(all_w2c, focal, color, sphere_radius, camera_size=0.1):
    # sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius, resolution=10)
    # sphere = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
    # sphere.paint_uniform_color((1, 0, 0))

    # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0., 0., 0.])
    # things_to_draw = [sphere, coord_frame]
    things_to_draw = []
    idx = 0
    frustums = []
    for W2C in all_w2c:
        idx += 1

        cnt = 0
        
        img_size = (512, 512)
        frustums.append(get_camera_frustum(img_size, focal, W2C, frustum_length=camera_size, color=color))
        cnt += 1

    # print("frustums", len(frustums))
    cameras = frustums2lineset(frustums)
    things_to_draw.append(cameras)

    # for color, camera_dict in colored_camera_dicts:
    #     idx += 1

    #     cnt = 0
    #     frustums = []
    #     for img_name in sorted(camera_dict.keys()):
    #         K = np.array(camera_dict[img_name]['K']).reshape((4, 4))
    #         W2C = np.array(camera_dict[img_name]['W2C']).reshape((4, 4))
    #         C2W = np.linalg.inv(W2C)
    #         img_size = camera_dict[img_name]['img_size']
    #         frustums.append(get_camera_frustum(img_size, K, W2C, frustum_length=camera_size, color=color))
    #         cnt += 1
    #     cameras = frustums2lineset(frustums)
    #     things_to_draw.append(cameras)

    # print(K[0,0])
    # focal = K[0,0]
    # directions = get_ray_directions(1600, 1600, focal) # (h, w, 3)
    # print("directions", directions.shape)
    # c2w = np.linalg.inv(all_w2c[30])
    # c2w = torch.FloatTensor(c2w)[:3, :4]
    # rays_o, rays_d = get_rays(directions, c2w)

    # rays_o = rays_o.numpy()
    # rays_d = rays_d.numpy()
    fig = pv.figure()
    # for j in range(2500):
    #     start = rays_o[j,:]
    #     end = rays_o[j,:] + rays_d[j,:]*0.03
    #     line = np.concatenate((start[None, :],end[None, :]), axis=0)
    #     fig.plot(line, c=(1.0, 0.5, 0.0))

    #     start = rays_o[j,:] + rays_d[j,:]*0.03
    #     end = rays_o[j,:] + rays_d[j,:]*4.5
    #     line = np.concatenate((start[None, :],end[None, :]), axis=0)
    #     fig.plot(line, c=(0.0, 1.0, 0.0))


    # if geometry_file is not None:
    #     if geometry_type == 'mesh':
    #         geometry = o3d.io.read_triangle_mesh(geometry_file)
    #         geometry.compute_vertex_normals()
    #     elif geometry_type == 'pointcloud':
    #         geometry = o3d.io.read_point_cloud(geometry_file)
    #     else:
    #         raise Exception('Unknown geometry_type: ', geometry_type)

    #     things_to_draw.append(geometry)
    for geometry in things_to_draw:
        fig.add_geometry(geometry)
    fig.show()

    # o3d.visualization.draw_geometries(things_to_draw)

from numpy import loadtxt
if __name__ == '__main__':
    import os

    base_dir = '/home/zubair/Downloads/left_mid_undistorted'

    # json_files = [pos_json for pos_json in os.listdir(base_dir) if pos_json.endswith('.json')]
    with open('/home/zubair/Downloads/outfile') as f:
        lines = f.readlines()
    # lines = np.genfromtxt("/home/zubair/Downloads/outfile", delimiter="\n")
    # print(len(lines))
    # print(lines[0])
    fx=264.0317 
    fy=264.0317 
    cx=258.8 
    cy=258.8

    pose_all = []
    for line in lines:
        array = line.split()
        print(len(array))
        pose = []
        for num in array:
            print(num)
            pose.append(float((num)))
        pose = np.array(pose).reshape(3,4)
        pose_4by4 = np.eye(4)
        pose_4by4[:3, :4] = pose
        pose_all.append(pose_4by4)

    sphere_radius = 1.
    camera_size = 0.1

    # geometry_file = os.path.join(base_dir, 'scene.ply')
    geometry_type = 'mesh'

    visualize_cameras(pose_all, fx, [0, 0, 1], sphere_radius, 
                      camera_size=camera_size)