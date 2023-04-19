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



def get_camera_frustum(img_size, K, W2C, frustum_length=1, color=[0., 1., 0.]):
    W, H = img_size
    hfov = np.rad2deg(np.arctan(W / 2. / K[0, 0]) * 2.)
    vfov = np.rad2deg(np.arctan(H / 2. / K[1, 1]) * 2.)
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
    C2W = np.linalg.inv(W2C)
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

def visualize_cameras(all_w2c, all_intrinsics, color, sphere_radius, camera_size=0.1, geometry_file=None, geometry_type='mesh'):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius, resolution=10)
    sphere = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
    sphere.paint_uniform_color((1, 0, 0))

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0., 0., 0.])
    things_to_draw = [sphere, coord_frame]

    idx = 0
    frustums = []
    for W2C, K, in zip(all_w2c, all_intrinsics):
        idx += 1

        cnt = 0
        
        img_size = (1600, 1600)
        frustums.append(get_camera_frustum(img_size, K, W2C, frustum_length=camera_size, color=color))
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
    focal = K[0,0]
    directions = get_ray_directions(1600, 1600, focal) # (h, w, 3)
    print("directions", directions.shape)
    c2w = np.linalg.inv(all_w2c[30])
    c2w = torch.FloatTensor(c2w)[:3, :4]
    rays_o, rays_d = get_rays(directions, c2w)

    rays_o = rays_o.numpy()
    rays_d = rays_d.numpy()
    fig = pv.figure()
    for j in range(2500):
        start = rays_o[j,:]
        end = rays_o[j,:] + rays_d[j,:]*0.03
        line = np.concatenate((start[None, :],end[None, :]), axis=0)
        fig.plot(line, c=(1.0, 0.5, 0.0))

        start = rays_o[j,:] + rays_d[j,:]*0.03
        end = rays_o[j,:] + rays_d[j,:]*4.5
        line = np.concatenate((start[None, :],end[None, :]), axis=0)
        fig.plot(line, c=(0.0, 1.0, 0.0))


    if geometry_file is not None:
        if geometry_type == 'mesh':
            geometry = o3d.io.read_triangle_mesh(geometry_file)
            geometry.compute_vertex_normals()
        elif geometry_type == 'pointcloud':
            geometry = o3d.io.read_point_cloud(geometry_file)
        else:
            raise Exception('Unknown geometry_type: ', geometry_type)

        things_to_draw.append(geometry)
    for geometry in things_to_draw:
        fig.add_geometry(geometry)
    fig.show()

    # o3d.visualization.draw_geometries(things_to_draw)


if __name__ == '__main__':
    import os

    base_dir = '/home/zubair/Downloads/google_scanned/00000'

    json_files = [pos_json for pos_json in os.listdir(base_dir) if pos_json.endswith('.json')]
    
    json_files.sort()
    all_w2c = []
    all_intrinsics = []
    for json_file in json_files:
        print(os.path.join(base_dir, json_file))
        file = os.path.join(base_dir, json_file)
        with open(file, 'r') as f:
            data = json.loads(f.read())
            # data = json.loads(f)
        cam2world = data["camera_data"]["cam2world"]
        print(np.array(cam2world).shape)
        # w2c = np.linalg.inv(np.array(cam2world))
        intrinsics = data["camera_data"]["intrinsics"]

        # img_name = json_file.split('.')[0]+ '.exr'
        # seg_name = json_file.split('.')[0]+ '.seg.exr'
        # depth_name = json_file.split('.')[0]+ '.depth.exr'
        
        # print("os.path.join(base_dir, img_name)", os.path.join(base_dir, img_name))
        # img = load_image_from_exr(os.path.join(base_dir, img_name))
        # # img = Image.fromarray(np.uint8(img*255), 'RGB')
        # # img = Image.fromarray(np.uint8(img*255))
        # img = Image.fromarray(np.clip(img*255, 0, 255).astype(np.uint8))

        # img = img.resize((400, 400), Image.LANCZOS)
        # seg_masks = load_seg_from_exr(os.path.join(base_dir, seg_name))
        # seg_masks = cv2.resize(seg_masks,(400,400), interpolation=cv2.INTER_NEAREST)
        # # depth = load_seg_from_exr(os.path.join(base_dir, depth_name))
        # # depth.astype(np.uint16)
        # seg_masks = seg_masks.astype(np.uint8)
        # img_write_name =  img_name.split('.')[0]+'.png'
        # seg_write_name =  img_name.split('.')[0]+'.seg.png'
        # # depth_write_name = img_name.split('.')[0]+'.depth.png'
        # img.save(os.path.join(base_dir, img_write_name))
        # cv2.imwrite(os.path.join(base_dir, seg_write_name), seg_masks)
        # # cv2.imwrite(os.path.join(base_dir, depth_write_name), depth)

        print(len(data["objects"]))
        for k in range(len(data["objects"])):
            print(data["objects"][k]["segmentation_id"])
        print("intrinsics", intrinsics)
        intrinsics_matrix = np.array([
                [intrinsics["fx"], 0., intrinsics["cx"], 0.0],
                [0., intrinsics["fy"], intrinsics["cy"], 0.0],
                [0., 0., 1., 0.0],
                [0., 0., 0., 1.],
            ])
        print("intrinsics_matrix", intrinsics_matrix)
        print("cam2world", np.array(cam2world).T)
        print("cam2world", np.linalg.inv(np.array(cam2world).T))
        all_w2c.append(np.linalg.inv(np.array(cam2world).T))
        all_intrinsics.append(intrinsics_matrix)

    print("json_files", json_files)
    sphere_radius = 1.
    # train_cam_dict = json.load(open(os.path.join(base_dir, 'train/cam_dict_norm.json')))
    # test_cam_dict = json.load(open(os.path.join(base_dir, 'test/cam_dict_norm.json')))
    # path_cam_dict = json.load(open(os.path.join(base_dir, 'camera_path/cam_dict_norm.json')))
    camera_size = 0.1
    # colored_camera_dicts = [([0, 1, 0], train_cam_dict),
    #                         ([0, 0, 1], test_cam_dict),
    #                         ([1, 1, 0], path_cam_dict)
    #                         ]

    geometry_file = os.path.join(base_dir, 'scene.ply')
    geometry_type = 'mesh'

    visualize_cameras(all_w2c, all_intrinsics, [0, 0, 1], sphere_radius, 
                      camera_size=camera_size, geometry_file=geometry_file, geometry_type=geometry_type)