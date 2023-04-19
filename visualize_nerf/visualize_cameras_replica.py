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

def visualize_cameras(all_c2w, focal, color, sphere_radius, camera_size=0.1, geometry_file=None):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius, resolution=10)
    sphere = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
    sphere.paint_uniform_color((1, 0, 0))

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0., 0., 0.])
    things_to_draw = [sphere, coord_frame]
    # things_to_draw = []
    idx = 0
    frustums = []
    for C2W in all_c2w:
        idx += 1

        cnt = 0
        
        img_size = (128, 128)
        frustums.append(get_camera_frustum(img_size, focal, C2W, frustum_length=camera_size, color=color))
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
    focal = focal
    directions = get_ray_directions(128, 128, focal) # (h, w, 3)
    print("directions", directions.shape)
    # c2w = np.linalg.inv(all_w2c[30])
    c2w = all_c2w[2]
    c2w = torch.FloatTensor(c2w)[:3, :4]
    rays_o, rays_d = get_rays(directions, c2w)

    rays_o = rays_o.numpy()
    rays_d = rays_d.numpy()
    fig = pv.figure()
    for j in range(2500):
        start = rays_o[j,:]
        end = rays_o[j,:] + rays_d[j,:]*0.8
        line = np.concatenate((start[None, :],end[None, :]), axis=0)
        fig.plot(line, c=(1.0, 0.5, 0.0))

        start = rays_o[j,:] + rays_d[j,:]*0.8
        end = rays_o[j,:] + rays_d[j,:]*1.8
        line = np.concatenate((start[None, :],end[None, :]), axis=0)
        fig.plot(line, c=(0.0, 1.0, 0.0))


    # if geometry_file is not None:
    #     if geometry_type == 'mesh':
    #         geometry = o3d.io.read_triangle_mesh(geometry_file)
    #         geometry.compute_vertex_normals()
    #     elif geometry_type == 'pointcloud':
    #         geometry = o3d.io.read_point_cloud(geometry_file)
    #     else:
    #         raise Exception('Unknown geometry_type: ', geometry_type)

    # for pcd in geometry_file:
    #     things_to_draw.append(pcd)
    for geometry in things_to_draw:
        fig.add_geometry(geometry)
    for C2W in all_c2w:
        fig.plot_transform(A2B=C2W, s=0.1)
    fig.show()
    fig.show()

    # o3d.visualization.draw_geometries(things_to_draw)

def load_intrinsic(intrinsic_path):
    with open(intrinsic_path, 'r') as f:
        lines = f.readlines()
        focal = float(lines[0].split()[0])
        H, W = lines[-1].split()
        H, W = int(H), int(W)
    return focal, H, W

from PIL import Image
import kornia as kn
if __name__ == '__main__':
    import os

    camera_file = '/home/zubair/ml-gsn/data/replica_all/train/00/cameras.json'

    with open(camera_file, 'r') as f:
        data = json.loads(f.read())


    all_c2w = []
    all_intrinsics = []
    for item in data:
        all_c2w.append(np.linalg.inv(np.array(item['Rt'])))
        
        K = np.array(item['K'])[:3, :3]
        fx = 256.0 / np.tan(np.deg2rad(90.0) / 2)
        fy = 256.0 / np.tan(np.deg2rad(90.0) / 2)
        K[0, 0] = K[0, 0] * fx
        K[1, 1] = fy
        K[0,2] = 256
        K[1,2] = 256
        K[2,2] = 1

        print("K", K)

        all_intrinsics.append(K)

    depth_paths = [fileName for fileName in os.listdir('/home/zubair/ml-gsn/data/replica_all/train/00') if fileName.endswith(".tiff")]

    base_dir = '/home/zubair/ml-gsn/data/replica_all/train/00'
    all_depths = []
    for path in depth_paths:
        depth_path = os.path.join(base_dir, path)
        depth = Image.open(depth_path)
        depth = np.array(depth) * 1000
        np.clip(depth, 0.0, 200.)
        depth = torch.tensor(depth).unsqueeze(0).unsqueeze(0)
        depth[depth >80.] = 0.0
        all_depths.append(depth)
        
        
    all_pcs = [kn.geometry.depth.depth_to_3d(depth, torch.tensor(K).unsqueeze(0), normalize_points=False) for (depth,K) in zip(all_depths, all_intrinsics)]

    viz_clouds = []
    for pc, pose in zip(all_pcs, all_c2w):
        pc = pc.squeeze().numpy().reshape(3,-1).transpose()
        pcd_vis = o3d.geometry.PointCloud()
        # points
        pcd_vis.points = o3d.utility.Vector3dVector(pc)
        # colors
        pcd_vis.paint_uniform_color((np.random.rand(), np.random.rand(), np.random.rand()))
        pcd_vis.transform(pose)
        viz_clouds.append(pcd_vis)
    # focal = np.array(data[0]['K'][0])
    focal = 256
    
    sphere_radius = 1.
    # train_cam_dict = json.load(open(os.path.join(base_dir, 'train/cam_dict_norm.json')))
    # test_cam_dict = json.load(open(os.path.join(base_dir, 'test/cam_dict_norm.json')))
    # path_cam_dict = json.load(open(os.path.join(base_dir, 'camera_path/cam_dict_norm.json')))
    camera_size = 0.1
    # colored_camera_dicts = [([0, 1, 0], train_cam_dict),
    #                         ([0, 0, 1], test_cam_dict),
    #                         ([1, 1, 0], path_cam_dict)
    #                         ]

    # intrinsic_path = os.path.join('/home/zubair/1a1dcd236a1e6133860800e6696b8284', 'intrinsics.txt')
    # focal, H, W = load_intrinsic(intrinsic_path)
    # geometry_file = '/home/zubair/1a1dcd236a1e6133860800e6696b8284/model_normalized.obj'
    # geometry_type = 'mesh'

    visualize_cameras(all_c2w, focal, [0, 0, 1], sphere_radius, 
                      camera_size=camera_size, geometry_file=viz_clouds)