import open3d as o3d
import json
import numpy as np
import json

import torch
from kornia import create_meshgrid
import numpy as np
import torch.nn.functional as F
import math
from torch import linalg as LA
import os
import matplotlib.pyplot as plt

def world2camera_matrix(w_xyz, c2w):
    # invert c2w to get world-to-camera matrix w2c
    w2c = torch.linalg.inv(c2w.squeeze(0))
    # transform world grid points to camera frame
    camera_grid = torch.matmul(w_xyz, w2c[:3, :3].T) + w2c[:3, 3]
    return camera_grid

def w2i_projection(w_xyz, cam2world, intrinsics):
    """Converts the points in world coordinates to camera view.
    :param xyz: points in world coordinates (SB*NV, NC, 3)
    :param poses: camera matrix (SB*NV, 4, 4)
    :output points in camera coordinates (SB*NV, NC, 3)
    : SB batch size
    : NV number of views in each scene
    : NC number of coordinate points
    """
    w_xyz = torch.cat([w_xyz, torch.ones_like(w_xyz[..., :1])], dim=-1)  # [n_points, 4]
    cam_xyz = torch.inverse(cam2world).bmm(w_xyz.permute(0,2,1))
    camera_grids = cam_xyz.permute(0,2,1)[:,:,:3]
    projections = intrinsics[None, ...].repeat(cam2world.shape[0], 1, 1).bmm(cam_xyz[:,:3,:])
    projections = projections.permute(0,2,1)
    uv = projections[..., :2] / projections[..., 2:3]  # [n_views, n_points, 2]
    im_z = projections[..., 2:3]
    return camera_grids, uv, im_z

def transform_rays_to_bbox_coordinates(rays_o, rays_d, axis_align_mat):
    rays_o_bbox = rays_o
    rays_d_bbox = rays_d
    T_box_orig = axis_align_mat
    rays_o_bbox = (T_box_orig[:3, :3] @ rays_o_bbox.T).T + T_box_orig[:3, 3]
    rays_d_bbox = (T_box_orig[:3, :3] @ rays_d.T).T
    return rays_o_bbox, rays_d_bbox

# def read_poses(pose_dir_train, img_files_train):
#     pose_file_train = os.path.join(pose_dir_train, 'pose.json')
#     with open(pose_file_train, "r") as read_content:
#         data = json.load(read_content)

#     focal = data['focal']
#     fov = data['fov']
#     img_wh = data['img_size']
#     obj_location = np.array(data["obj_location"])
#     all_c2w_train = []
#     all_c2w_test =[]

#     for i, img_file in enumerate(img_files_train):
#         print("image file", i, img_file)
#         c2w = np.array(data['transform'][img_file.split('.')[0]])
#         c2w[:3, 3] = c2w[:3, 3] - obj_location
#         all_c2w_train.append(convert_pose_PD_to_NeRF(c2w))


#     all_c2w_train = np.array(all_c2w_train)
#     # all_c2w_test = np.array(all_c2w_test)
    
#     pose_scale_factor = 1. / np.max(np.abs(all_c2w_train[:, :3, 3]))
#     all_c2w_train[:, :3, 3] *= pose_scale_factor
#     # all_c2w_test[:, :3, 3] *= pose_scale_factor


#     all_c2w_val = all_c2w_train[100:]
#     all_c2w_train = all_c2w_train[:100]

#     all_boxes = []
#     all_boxes_new = []
#     all_boxes_newer = []

#     all_boxes = []
#     all_translations= []
#     all_rotations = []

#     for k,v in data['bbox_dimensions'].items():
            
#             bbox = np.array(v)
#             bbox_dimension =  [(bbox[1,0]-bbox[0,0]), (bbox[1,1]-bbox[0,1]), (bbox[1,2]-bbox[0,2])]
#             all_boxes.append(np.array(bbox_dimension)*pose_scale_factor)
#             all_boxes_new.append(bbox*pose_scale_factor)

#             all_rotations.append(data["obj_rotations"][k])
#             translation = (np.array(data['obj_translations'][k])- obj_location)*pose_scale_factor 
#             # translation = (np.array(data['obj_translations'][k])) 
#             all_translations.append(translation)

#     #bbox_dimension_modified = {'R': all_rotations, 'T': all_translations_new, 's': all_boxes_new}
#     bbox_dimension_modified = {'R': all_rotations, 'T': all_translations, 's': all_boxes}
#     RTs = {'R': all_rotations, 'T': all_translations, 's': all_boxes_new}
#     return all_c2w_train, all_c2w_val, focal, fov, img_wh, bbox_dimension_modified, RTs, pose_scale_factor, obj_location


def read_poses(pose_dir_train, img_files_train, pose_dir_test, img_files_test):
    pose_file_train = os.path.join(pose_dir_train, 'pose.json')
    with open(pose_file_train, "r") as read_content:
        data = json.load(read_content)

    if pose_dir_test is not None:
        pose_file_test = os.path.join(pose_dir_test, 'pose.json')
        with open(pose_file_test, "r") as read_content:
            data_test = json.load(read_content)

    focal = data['focal']
    fov = data['fov']
    img_wh = data['img_size']
    obj_location = np.array(data["obj_location"])
    all_c2w_train = []
    all_c2w_test =[]

    for i, (img_file, img_fil_test) in enumerate(zip(img_files_train, img_files_test)):
        print("image file", i, img_file)
        c2w = np.array(data['transform'][img_file.split('.')[0]])
        c2w[:3, 3] = c2w[:3, 3] - obj_location
        all_c2w_train.append(convert_pose_PD_to_NeRF(c2w))

        c2w = np.array(data_test['transform'][img_fil_test.split('.')[0]])
        c2w[:3, 3] = c2w[:3, 3] - obj_location
        all_c2w_test.append(convert_pose_PD_to_NeRF(c2w))

    all_c2w_train = np.array(all_c2w_train)
    all_c2w_test = np.array(all_c2w_test)
    
    pose_scale_factor = 1. / np.max(np.abs(all_c2w_train[:, :3, 3]))
    all_c2w_train[:, :3, 3] *= pose_scale_factor
    all_c2w_test[:, :3, 3] *= pose_scale_factor

    # pose_scale_factor = None

    # c2w = all_c2w_train[49]
    # all_c2w_train_transformed = []
    # for i in range(all_c2w_train.shape[0]):
    #     all_c2w_train_transformed.append(np.linalg.inv(c2w)@all_c2w_train[i,:,:])
    # all_c2w_train = np.array(all_c2w_train_transformed)


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

            # box_transform = np.eye(4)
            # box_transform[:3,:3] = np.array(data["obj_rotations"][k])
            # box_transform[:3, 3] = (np.array(data['obj_translations'][k])- obj_location)*pose_scale_factor 

            # box_transform = np.linalg.inv(c2w) @ box_transform
            # Rot_transformed = box_transform[:3,:3]
            # Tran_transformed = box_transform[:3, 3]

            # all_rotations.append(Rot_transformed)
            # all_translations.append(Tran_transformed)

            all_rotations.append(data["obj_rotations"][k])
            translation = (np.array(data['obj_translations'][k])- obj_location)*pose_scale_factor 
            # translation = (np.array(data['obj_translations'][k])) 
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
    return all_c2w_train, all_c2w_val, all_c2w_test, focal, fov, img_wh, bbox_dimension_modified, RTs, pose_scale_factor, obj_location


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

def depth2pts_outside(rays_o, rays_d, depth):
    """Compute the points along the ray that are outside of the unit sphere.
    Args:
        rays_o: [num_rays, 3]. Ray origins of the points.
        rays_d: [num_rays, 3]. Ray directions of the points.
        depth: [num_rays, num_samples along ray]. Inverse of distance to sphere origin.
    Returns:
        pts: [num_rays, 4]. Points outside of the unit sphere. (x', y', z', 1/r)
    """
    # note: d1 becomes negative if this mid point is behind camera
    rays_o = rays_o[..., None, :].expand(
        list(depth.shape) + [3]
    )  #  [N_rays, num_samples, 3]
    rays_d = rays_d[..., None, :].expand(
        list(depth.shape) + [3]
    )  #  [N_rays, num_samples, 3]
    d1 = -torch.sum(rays_d * rays_o, dim=-1, keepdim=True) / torch.sum(
        rays_d**2, dim=-1, keepdim=True
    )

    p_mid = rays_o + d1 * rays_d
    p_mid_norm = torch.norm(p_mid, dim=-1, keepdim=True)
    rays_d_cos = 1.0 / torch.norm(rays_d, dim=-1, keepdim=True)

    check_pos = 1.0 - p_mid_norm * p_mid_norm
    assert torch.all(check_pos >= 0), "1.0 - p_mid_norm * p_mid_norm should be greater than 0"

    d2 = torch.sqrt(1.0 - p_mid_norm * p_mid_norm) * rays_d_cos
    p_sphere = rays_o + (d1 + d2) * rays_d

    rot_axis = torch.cross(rays_o, p_sphere, dim=-1)
    rot_axis = rot_axis / torch.norm(rot_axis, dim=-1, keepdim=True)
    phi = torch.asin(p_mid_norm)
    theta = torch.asin(p_mid_norm * depth[..., None])  # depth is inside [0, 1]
    rot_angle = phi - theta  # [..., 1]

    # now rotate p_sphere
    # Rodrigues formula: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    p_sphere_new = (
        p_sphere * torch.cos(rot_angle)
        + torch.cross(rot_axis, p_sphere, dim=-1) * torch.sin(rot_angle)
        + rot_axis
        * torch.sum(rot_axis * p_sphere, dim=-1, keepdim=True)
        * (1.0 - torch.cos(rot_angle))
    )
    p_sphere_new = p_sphere_new / (
        torch.norm(p_sphere_new, dim=-1, keepdim=True) + 1e-10
    )
    pts = torch.cat((p_sphere_new, depth.unsqueeze(-1)), dim=-1)

    return pts

def sample_along_rays(
    rays_o,
    rays_d,
    num_samples,
    near,
    far,
    randomized,
    lindisp,
    in_sphere,
    far_uncontracted = 3.0
):
    bsz = rays_o.shape[0]
    t_vals = torch.linspace(0.0, 1.0, num_samples + 1, device=rays_o.device)

    if in_sphere:
        if lindisp:
            t_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)
        else:
            t_vals = near * (1.0 - t_vals) + far * t_vals
    else:
        t_vals_linear = far * (1.0 - t_vals) + far_uncontracted * t_vals
        t_vals = torch.broadcast_to(t_vals, (bsz, num_samples + 1))

    if randomized:
        mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
        upper = torch.cat([mids, t_vals[..., -1:]], -1)
        lower = torch.cat([t_vals[..., :1], mids], -1)
        t_rand = torch.rand((bsz, num_samples + 1), device=rays_o.device)
        t_vals = lower + (upper - lower) * t_rand

        if not in_sphere:
            mids = 0.5 * (t_vals_linear[..., 1:] + t_vals_linear[..., :-1])
            upper = torch.cat([mids, t_vals_linear[..., -1:]], -1)
            lower = torch.cat([t_vals_linear[..., :1], mids], -1)
            t_rand = torch.rand((bsz, num_samples + 1), device=rays_o.device)
            t_vals_linear = lower + (upper - lower) * t_rand
    else:
        t_vals = torch.broadcast_to(t_vals, (bsz, num_samples + 1))

    if in_sphere:
        coords = cast_rays(t_vals, rays_o, rays_d)
        return t_vals, coords
    else:
        t_vals = torch.flip(
            t_vals,
            dims=[
                -1,
            ],
        )  # 1.0 -> 0.0
        t_vals_linear = torch.flip(
            t_vals_linear,
            dims=[
                -1,
            ],
        )  # 3.0 -> sphere 
        coords_linear = cast_rays(t_vals_linear, rays_o, rays_d)
        coords = depth2pts_outside(rays_o, rays_d, t_vals)
        return t_vals, coords, coords_linear

# def sample_along_rays(
#     rays_o,
#     rays_d,
#     num_samples,
#     near,
#     far,
#     randomized,
#     lindisp,
#     in_sphere,
# ):
#     bsz = rays_o.shape[0]
#     t_vals = torch.linspace(0.0, 1.0, num_samples + 1, device=rays_o.device)

#     if in_sphere:
#         if lindisp:
#             t_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)
#         else:
#             t_vals = near * (1.0 - t_vals) + far * t_vals
#     else:
#         t_vals = torch.broadcast_to(t_vals, (bsz, num_samples + 1))

#     if randomized:
#         mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
#         upper = torch.cat([mids, t_vals[..., -1:]], -1)
#         lower = torch.cat([t_vals[..., :1], mids], -1)
#         t_rand = torch.rand((bsz, num_samples + 1), device=rays_o.device)
#         t_vals = lower + (upper - lower) * t_rand
#     else:
#         t_vals = torch.broadcast_to(t_vals, (bsz, num_samples + 1))

#     if in_sphere:
#         coords = cast_rays(t_vals, rays_o, rays_d)
#     else:
#         t_vals = torch.flip(
#             t_vals,
#             dims=[
#                 -1,
#             ],
#         )  # 1.0 -> 0.0
#         coords = depth2pts_outside(rays_o, rays_d, t_vals)

#     return t_vals, coords

def convert_pose(C2W):
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = np.matmul(C2W, flip_yz)
    return C2W

def sample_along_rays_vanilla(
    rays_o,
    rays_d,
    num_samples,
    near,
    far,
    randomized,
    lindisp,
):
    bsz = rays_o.shape[0]
    t_vals = torch.linspace(0.0, 1.0, num_samples + 1, device=rays_o.device)
    if lindisp:
        t_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)
    else:
        t_vals = near * (1.0 - t_vals) + far * t_vals

    if randomized:
        mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
        upper = torch.cat([mids, t_vals[..., -1:]], -1)
        lower = torch.cat([t_vals[..., :1], mids], -1)
        t_rand = torch.rand((bsz, num_samples + 1), device=rays_o.device)
        t_vals = lower + (upper - lower) * t_rand
    else:
        t_vals = torch.broadcast_to(t_vals, (bsz, num_samples + 1))

    coords = cast_rays(t_vals, rays_o, rays_d)

    return t_vals, coords

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



def contract_samples(x, order=float('inf')):
    mag = LA.norm(x, order, dim=-1)[..., None]
    return torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag)), mag

def inverse_contract_samples(x, mag_origial,order=float('inf')):
    mag = LA.norm(x, order, dim=-1)[..., None]
    return torch.where(mag < 1, x, (x*mag_origial)/(2-(1/mag_origial)))

def _inverse_contract(x):
    x_mag_sq = torch.sum(x**2, dim=-1, keepdim=True).clip(min=1e-32)
    z = torch.where(
        x_mag_sq <= 1, x, x * (x_mag_sq / (2 * torch.sqrt(x_mag_sq) - 1))
    )
    return z


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