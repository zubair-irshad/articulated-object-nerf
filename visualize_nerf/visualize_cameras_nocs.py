import open3d as o3d
import json
import numpy as np
import json
import struct
import torch
from kornia import create_meshgrid
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.visualizer as pv
from datasets.google_scanned_utils import *
import cv2
from PIL import Image
from objectron.schema import annotation_data_pb2 as annotation_protocol
from objectron.schema import a_r_capture_metadata_pb2 as ar_metadata_protocol
import colorsys
from datasets.ray_utils import world_to_ndc, get_ray_bbox_intersections, get_rays_in_bbox
from datasets.viz_utils import line_set_mesh
import torch.nn.functional as F
import trimesh
from datasets.nocs_utils import load_depth, process_data, get_GT_poses, rebalance_mask
from datasets.viz_utils import get_object_rays_in_bbox_unscaled
import kornia as kn

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
            [0,1,0,-0.9*t],
            [0,0,1,t],
            [0,0,0,1],
        ])

        rot_phi = lambda phi : np.array([
            [1,0,0,0],
            [0,np.cos(phi),-np.sin(phi),0],
            [0,np.sin(phi), np.cos(phi),0],
            [0,0,0,1],
        ])

        rot_theta = lambda th : np.array([
            [np.cos(th),0,-np.sin(th),0],
            [0,1,0,0],
            [np.sin(th),0, np.cos(th),0],
            [0,0,0,1],
        ])

        c2w = rot_theta(theta) @ rot_phi(phi) @ trans_t(radius)
        c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
        return c2w

    spheric_poses = []
    for th in np.linspace(0, 2*np.pi, n_poses+1)[:-1]:
        spheric_poses += [spheric_pose(-np.pi/2, th, radius)] # 36 degree view downwards
    return np.stack(spheric_poses, 0)

def apply_symmetry_transform(rays_o, rays_d, reflection_dim):
    """ Applies symmetry transformation to a batch of 3D points.
    Args:
        x: point cloud with shape (..., 3)
    Returns:
        transformed point cloud with shape (..., 3)
    """
    rays_o = torch.FloatTensor(rays_o)
    rays_d = torch.FloatTensor(rays_d)
    canonical_symmetry_matrix = torch.eye(4)
    canonical_symmetry_matrix[reflection_dim, reflection_dim] *= -1
    canonical_symmetry_matrix = canonical_symmetry_matrix.unsqueeze(0) # add batch dim

    rays_o = F.pad(rays_o, (0, 1), mode='constant', value=0.0) # homogenise points
    rays_d = F.pad(rays_d, (0, 1), mode='constant', value=1.0) # homogenise points

    rays_o = torch.einsum('...ij,...j->...i', canonical_symmetry_matrix, rays_o)[..., :3]
    rays_d = torch.einsum('...ij,...j->...i', canonical_symmetry_matrix, rays_d)[..., :3]
    return rays_o.numpy(), rays_d.numpy()

def symmetry_plane(invT):
    invT = torch.Tensor(invT)
    reflection_dim = 2
    canonical_symmetry_matrix = torch.eye(4).cuda()
    canonical_symmetry_matrix[reflection_dim, reflection_dim] *= -1
    canonical_symmetry_matrix = canonical_symmetry_matrix.unsqueeze(0) # add batch dim

    reflection_dim = reflection_dim
    height = 1.0
    v = torch.tensor([
        [-1, -1, -height],
        [-1, -1, height],
        [1, 1, height],
        [1, 1, -height]
    ], dtype=torch.float32, device=invT.device)
    v[:, reflection_dim] = 0

    # v = F.pad(v, (0, 1), mode='constant', value=1.0) # homogenise
    # v_w = torch.einsum('...ij,...j->...i', invT, v)[..., :3]

    tris = torch.tensor([
        [0, 1, 2],
        [0, 2, 3]
    ], dtype=torch.int32, device=invT.device)
    return v.detach().cpu(), tris.cpu()


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

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def transform_rays_to_bbox_coordinates_nocs(rays_o, rays_d, axis_align_mat):
    rays_o_bbox = rays_o
    rays_d_bbox = rays_d
    T_box_orig = axis_align_mat
    rays_o_bbox = (T_box_orig[:3, :3] @ rays_o_bbox.T).T + T_box_orig[:3, 3]
    rays_d_bbox = (T_box_orig[:3, :3] @ rays_d.T).T
    return rays_o_bbox, rays_d_bbox

# def get_object_rays_in_bbox(rays_o, rays_d, RTs, canonical=False):
#     flip_yz = np.eye(4)
#     flip_yz[1, 1] = -1
#     flip_yz[2, 2] = -1

#     if not canonical:
#         instance_rotation = RTs['R']
#         instance_translation = RTs['T']
#         scale = RTs['s']

#         box_transformation = np.eye(4)
#         box_transformation[:3, :3] = np.reshape(instance_rotation, (3, 3))
#         box_transformation[:3, -1] = instance_translation
#         axis_aligned_mat = np.linalg.inv(box_transformation)
#     else:
#         scale = RTs['s']
#         axis_aligned_mat = None
#     bbox_bounds = np.array([-scale / 2, scale / 2])
#     rays_o_obj, rays_d_obj, batch_near_obj, batch_far_obj, bbox_mask = get_rays_in_bbox(rays_o, rays_d, bbox_bounds, None, axis_aligned_mat)
    
#     return rays_o_obj, rays_d_obj, batch_near_obj, batch_far_obj, bbox_mask

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


def convert_homopoints_to_points(points_4d):
  """Project 4d homogenous points (4xN) to 3d points (3xN)"""
  assert len(points_4d.shape) == 2
  assert points_4d.shape[0] == 4
  points_3d = points_4d[:3, :] / points_4d[3:4, :]
  assert points_3d.shape[1] == points_3d.shape[1]
  assert points_3d.shape[0] == 3
  return points_3d

def visualize_cameras(all_c2w, all_focal, points, RTs, color, all_model_points, all_abs_poses, depths_all,all_intrinsics, sphere_radius, camera_size=0.1):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius, resolution=10)
    sphere = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
    sphere.paint_uniform_color((1, 0, 0))
    # o3d.io.write_point_cloud("../../TestData/sync.ply", pcd)

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0., 0., 0.])
    things_to_draw = [sphere, coord_frame]
    # things_to_draw = []
    idx = 0
    frustums = []


    all_pcs = [kn.geometry.depth.depth_to_3d(torch.FloatTensor(depth*4.4/1000).unsqueeze(0).unsqueeze(0), torch.tensor(K).unsqueeze(0), normalize_points=False) for (depth,K) in zip(depths_all, all_intrinsics)]

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

    for pcd in viz_clouds:
        things_to_draw.append(pcd)
    # print(RTs)
    # instance_rotation = RTs['R']
    # instance_translation = RTs['T']
    # instance_scale = RTs['s']
    # Draw scene OBB


    # box = get_3d_bbox(instance_scale)
    # unit_box_homopoints = convert_points_to_homopoints(box.T)

    # box_transformation = np.eye(4)
    # box_transformation[:3, :3] = np.reshape(instance_rotation, (3, 3))
    # box_transformation[:3, -1] = instance_translation

    # morphed_box_homopoints = box_transformation @ unit_box_homopoints
    # morphed_box_points = convert_homopoints_to_points(morphed_box_homopoints).T
    
    # print("morphed_box_homopoints", morphed_box_homopoints.shape)
    # cylinder_segments = line_set_mesh(morphed_box_points)
    # for k in range(len(cylinder_segments)):
    #     things_to_draw.append(cylinder_segments[k])

    # print("instance_translation, instance_rotation, scale", instance_translation, instance_rotation, instance_scale)
    # bbox = o3d.geometry.OrientedBoundingBox(center=instance_translation, R=instance_rotation, extent=instance_scale)
    # bbox_canonical = o3d.geometry.OrientedBoundingBox(center = [0,0,0], R = np.eye(3), extent=instance_scale)
    # # things_to_draw.append(bbox)
    # things_to_draw.append(bbox_canonical)

    # print("bbox", np.array(bbox.get_box_points()))
    # print("morphed_box_points", morphed_box_points)

    # print("instance_scale", instance_scale)

    print("all_c2w.shape[0]", all_c2w.shape[0])
    for i in range(all_c2w.shape[0]):
        C2W = all_c2w[i]
        print("C2W", C2W.shape)
        idx += 1

        cnt = 0
        
        img_size = (640, 480)

        frustums.append(get_camera_frustum(img_size, focal, C2W, frustum_length=camera_size, color=color))
        cnt += 1

    # print("frustums", len(frustums))
    cameras = frustums2lineset(frustums)
    things_to_draw.append(cameras)

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # things_to_draw.append(pcd)

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

    # focal = all_focal[18]
    # focal = focal/12

    directions = get_ray_directions(480, 640, focal) # (h, w, 3)
    print("directions", directions.shape)
    # c2w = np.linalg.inv(all_c2w[30])
    c2w_ = all_c2w[18]
    model_points = all_model_points[18]
    abs_poses = all_abs_poses[18]
    depth = depths_all[18]
    c2w = torch.FloatTensor(c2w_)[:3, :4]
    #c2w = torch.FloatTensor(c2w_)
    # box_transformation = np.eye(4)
    # box_transformation[:3, :3] = np.reshape(instance_rotation, (3, 3))
    # box_transformation[:3, -1] = instance_translation
    # axis_aligned_mat = torch.FloatTensor(np.linalg.inv(box_transformation))

    #Plot depth
    # cam_cx = 320
    # cam_fx = focal
    # cam_cy = 240
    # cam_fy = focal


    # K = [[cam_fx,  ]

    # indices = depth.flatten().shape[0]
    # choose = np.random.choice(indices, 9113)
    # depth_masked = depth.flatten()[choose][:, np.newaxis]
    # xmap = np.array([[y for y in range(640)] for z in range(480)])
    # ymap = np.array([[z for y in range(640)] for z in range(480)])
    # xmap_masked = xmap.flatten()[choose][:, np.newaxis]
    # ymap_masked = ymap.flatten()[choose][:, np.newaxis]

    # pt2 = depth_masked/1000.0
    # pt0 = (xmap_masked - cam_cx) * pt2 / cam_fx
    # pt1 = (ymap_masked - cam_cy) * pt2 / cam_fy
    # points = np.concatenate((pt0, pt1, pt2), axis=1)
    # print("points", points.shape)
    # # pcd_depth = o3d.geometry.PointCloud()
    # # pcd_depth.points = o3d.utility.Vector3dVector(points)

    c2w_depth = c2w_

    # points = points
    # pc_homopoints = convert_points_to_homopoints(points.T)
    # morphed_pc_homopoints = c2w_depth @ pc_homopoints
    # points = convert_homopoints_to_points(morphed_pc_homopoints).T

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # things_to_draw.append(pcd)

    rays_o, rays_d = get_rays(directions, c2w)

    # verts_symm, tris_symm = symmetry_plane(np.linalg.inv(axis_aligned_mat.cpu().numpy()))
    # verts_symm = verts_symm.numpy().astype(np.float64)
    # tris_symm = tris_symm.numpy().astype(np.uint64)
    # mesh_symm = trimesh.Trimesh(verts_symm, tris_symm)
    # mesh = trimesh.util.concatenate([mesh, mesh_symm])

    # mesh = mesh_symm.as_open3d
    
    # rays_o, rays_d = transform_rays_to_bbox_coordinates_nocs(rays_o, rays_d, axis_aligned_mat)

    # poses_test = create_spheric_poses(0.28)

    rays_o = rays_o.numpy()
    rays_d = rays_d.numpy()
    fig = pv.figure()
    # things_to_draw.append(mesh)

    for abs_pose, points in zip(abs_poses, model_points):
        
        camera_T_object =  c2w_depth @ abs_pose.camera_T_object
        camera_T_object[3,3] = 1
        instance_translation = camera_T_object[:3,3]
        instance_rotation = camera_T_object[:3,:3]
        # instance_scale = abs_pose.scale_matrix[0,0]

        pc_hp = convert_points_to_homopoints(points.T)
        scaled_homopoints = (abs_pose.scale_matrix @ pc_hp)
        scaled_homopoints = convert_homopoints_to_points(scaled_homopoints).T
        size = 2 * np.amax(np.abs(scaled_homopoints), axis=0)

        bbox = o3d.geometry.OrientedBoundingBox(center=instance_translation, R=instance_rotation, extent=size)
        #bbox_canonical = o3d.geometry.OrientedBoundingBox(center = [0,0,0], R = np.eye(3), extent=instance_scale)
        things_to_draw.append(bbox)

    rays_o_all, rays_d_all, batch_near_obj, batch_far_obj = get_object_rays_in_bbox_unscaled(rays_o, rays_d, model_points, abs_poses, None, c2w=c2w)

    # # rays_o, rays_d, batch_near, batch_far, bbox_mask = get_object_rays_in_bbox(rays_o, rays_d, RTs)
    # rays_o, rays_d, batch_near, batch_far, bbox_mask = get_object_rays_in_bbox(rays_o, rays_d, RTs, canonical=True)
    # print(rays_o.shape, rays_d.shape)

    # print("bbox_mask", bbox_mask.shape)
    # bbox_mask = bbox_mask.reshape(120, 160)
    # image_save_dir = '/home/zubair/nerf_pl/Objectron_data/camera_batch-2_1/images_12'
    # mask_save_dir = '/home/zubair/nerf_pl/Objectron_data/camera_batch-2_1/masks_12'
    # img_pil = Image.open(os.path.join(image_save_dir, '00002.png'))
    # img_pil = img_pil.transpose(Image.ROTATE_90) 
    # mask = cv2.imread(os.path.join(mask_save_dir, 'mask_00000.png'), cv2.IMREAD_GRAYSCALE)
    # mask = np.rot90(np.array(mask), axes=(0,1))
    # mask = mask>0
    # colors = random_colors(1)
    # img = apply_mask(np.array(img_pil), bbox_mask, colors[0])
    # print("img", img.shape)
    # print("mask", mask.shape)
    # cv2.imshow('gfg', img)
    # cv2.waitKey(0)

    # rays_o_sym, rays_d_sym = apply_symmetry_transform(rays_o, rays_d, 2)


    # rays_o_sym = rays_o_sym[ids, :]
    # rays_d_sym = rays_d_sym[ids, :]

    # print("rays_o_sym", rays_o_sym.shape)

    
    # for i, (rays_o, rays_d, batch_near, batch_far) in enumerate(zip(rays_o_all, rays_d_all, batch_near_obj, batch_far_obj)):
    #     ids = np.random.choice(rays_o.shape[0], int(rays_o.shape[0]*0.5))
    #     print(batch_near, batch_far)
    #     rays_o = rays_o[ids, :]
    #     rays_d = rays_d[ids, :]
    #     batch_near = batch_near[ids]
    #     batch_far = batch_far[ids]
    #     for j in range(200):
    #         start = rays_o[j,:]
    #         end = rays_o[j,:] + rays_d[j,:]*batch_near[j, :].cpu().numpy()
    #         line = np.concatenate((start[None, :],end[None, :]), axis=0)
    #         fig.plot(line, c=(1.0, 0.5, 0.0))

    #         start = rays_o[j,:] + rays_d[j,:]*batch_near[j, :].cpu().numpy()
    #         end = rays_o[j,:] + rays_d[j,:]*batch_far[j, :].cpu().numpy()
    #         line = np.concatenate((start[None, :],end[None, :]), axis=0)
    #         fig.plot(line, c=(0.0, 1.0, 0.0))

        # start = rays_o_sym[j,:]
        # end = rays_o_sym[j,:] + rays_d_sym[j,:]*batch_near[j, :].cpu().numpy()
        # line = np.concatenate((start[None, :],end[None, :]), axis=0)
        # fig.plot(line, c=(1.0, 0.5, 0.0))

        # start = rays_o_sym[j,:] + rays_d_sym[j,:]*batch_near[j, :].cpu().numpy()
        # end = rays_o_sym[j,:] + rays_d_sym[j,:]*batch_far[j, :].cpu().numpy()
        # line = np.concatenate((start[None, :],end[None, :]), axis=0)
        # fig.plot(line, c=(0.0, 1.0, 0.0))

    for j in range(2500):
        start = rays_o[j,:]
        end = rays_o[j,:] + rays_d[j,:]*10
        line = np.concatenate((start[None, :],end[None, :]), axis=0)
        fig.plot(line, c=(1.0, 0.5, 0.0))

        start = rays_o[j,:] + rays_d[j,:]*10
        end = rays_o[j,:] + rays_d[j,:]*12
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

    #     things_to_draw.append(geometry)
    for geometry in things_to_draw:
        fig.add_geometry(geometry)
    for C2W in all_c2w:
        fig.plot_transform(A2B=C2W, s=0.1)
    
    # print("poses_test", poses_test.shape)
    # for C2W in poses_test:
    #     fig.plot_transform(A2B=C2W, s=0.1)

    # for C2W in all_c2w:
    #     C2W = axis_aligned_mat @ C2W
    #     # print("C2W", C2W)
    #     fig.plot_transform(A2B=C2W, s=0.1)
    fig.show()

    # o3d.visualization.draw_geometries(things_to_draw)


def make_poses_bounds_array(frame_data, near=0.2, far=10):
    # See https://github.com/Fyusion/LLFF#using-your-own-poses-without-running-colmap
    # Returns an array of shape (N, 17).
    adjust_matrix = np.array(
        [[0.,   1.,   0.],
        [1.,   0.,   0.],
        [0.,   0.,  -1.]])

    rows = []
    all_c2w = []
    all_focal = []
    for frame in frame_data:
        camera = frame.camera      
        focal = camera.intrinsics[0]
        all_focal.append(focal)
        cam_to_world = np.array(camera.transform).reshape(4,4)
        # cam_to_world = cam_to_world.T
        # cam_to_world[:3, :3] = np.matmul(adjust_matrix, cam_to_world[:3, :3])
        # cam_to_world = convert_pose(cam_to_world)
        # cam_to_world = cam_to_world* adjust_matrix
        
        all_c2w.append(cam_to_world)
    return all_c2w, all_focal



# def convert_pose(C2W):
#     flip_yz = np.eye(4)
#     flip_yz[1, 1] = -1
#     flip_yz[2, 2] = -1
#     C2W = np.matmul(C2W, flip_yz)
#     return C2W

def convert_pose(C2W):
    flip_yz = np.zeros((4,4))
    flip_yz[0, 1] = 1
    flip_yz[1, 0] = 1
    flip_yz[2, 2] = 1
    flip_yz[3, 3] = 1
    print(flip_yz)
    C2W = np.matmul(C2W, flip_yz)
    print(C2W)
    return C2W

def load_frame_data(geometry_filename):
    # See get_geometry_data in objectron-geometry-tutorial.ipynb
    frame_data = []
    with open(geometry_filename, 'rb') as pb:
        proto_buf = pb.read()

        i = 0
        while i < len(proto_buf):
            msg_len = struct.unpack('<I', proto_buf[i:i + 4])[0]
            i += 4
            message_buf = proto_buf[i:i + msg_len]
            i += msg_len
            frame = ar_metadata_protocol.ARFrame()
            frame.ParseFromString(message_buf)
            frame_data.append(frame)
    return frame_data

def feature_points_from_frame_data(frames):
    # Returns an array of shape (N, 3).
    points = []
    for frame in frames:
        for v in frame.raw_feature_points.point:
            points.append([v.x, v.y, v.z])
    return np.array(points)

def get_frame_annotation(annotation_filename):
    """Grab an annotated frame from the sequence."""
    result = []
    instances = []
    with open(annotation_filename, 'rb') as pb:
        sequence = annotation_protocol.Sequence()
        sequence.ParseFromString(pb.read())

        object_id = 0
        object_rotations = []
        object_translations = []
        object_scale = []
        num_keypoints_per_object = []
        object_categories = []
        annotation_types = []
        
        # Object instances in the world coordinate system, These are stored per sequence, 
        # To get the per-frame version, grab the transformed keypoints from each frame_annotation
        for obj in sequence.objects:
            rotation = np.array(obj.rotation).reshape(3, 3)
            translation = np.array(obj.translation)
            scale = np.array(obj.scale)
            points3d = np.array([[kp.x, kp.y, kp.z] for kp in obj.keypoints])
            instances.append((rotation, translation, scale, points3d))
        
        # Grab teh annotation results per frame
        for data in sequence.frame_annotations:
            # Get the camera for the current frame. We will use the camera to bring
            # the object from the world coordinate to the current camera coordinate.
            transform = np.array(data.camera.transform).reshape(4, 4)
            view = np.array(data.camera.view_matrix).reshape(4, 4)
            intrinsics = np.array(data.camera.intrinsics).reshape(3, 3)
            projection = np.array(data.camera.projection_matrix).reshape(4, 4)
        
            keypoint_size_list = []
            object_keypoints_2d = []
            object_keypoints_3d = []
            for annotations in data.annotations:
                num_keypoints = len(annotations.keypoints)
                keypoint_size_list.append(num_keypoints)
                for keypoint_id in range(num_keypoints):
                    keypoint = annotations.keypoints[keypoint_id]
                    object_keypoints_2d.append((keypoint.point_2d.x, keypoint.point_2d.y, keypoint.point_2d.depth))
                    object_keypoints_3d.append((keypoint.point_3d.x, keypoint.point_3d.y, keypoint.point_3d.z))
                num_keypoints_per_object.append(num_keypoints)
                object_id += 1
            result.append((object_keypoints_2d, object_keypoints_3d, keypoint_size_list, view, projection))

    return result, instances

import colmap_utils

def get_colamap_poses_and_points3D(scene_dir, img_wh = (640,480)):
    colmap_db = os.path.join(scene_dir, "sparse/0")
    cameras, images, points3D = colmap_utils.read_model(colmap_db, ext=".bin")
    print("cameras", cameras)
    print("len cameras", len(cameras))
    # assert len(cameras) == 1

    H = cameras[1].height
    W = cameras[1].width
    print("camera params", cameras[1].params)
    focal = cameras[1].params[0] * img_wh[0]/W

    f_x = cameras[1].params[0] 
    f_y = cameras[1].params[0] 
    p_x = cameras[1].params[1] 
    p_y = cameras[1].params[2] 
    intrinsics = np.array([
        [f_x, 0, p_x],
        [0, f_y, p_y],
        [0, 0, 1]
    ], dtype=np.float32)

    pts_world = np.zeros((1, 3, len(points3D))) # (1, 3, N_points)
    for i, k in enumerate(points3D):
        pts_world[0, :, i] = points3D[k].xyz

    # Parse colmap results
    w2c_mats = []
    bottom = np.array([0, 0, 0, 1.]).reshape(1, 4)
    all_model_points = []
    all_abs_poses = []
    depths_all = []
    all_intrinsics = []
    for id, k in enumerate(images):
        if id>50:
            continue
        im = images[k]
        R = im.qvec2rotmat()
        t = im.tvec.reshape(3, 1)
        w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]

        all_intrinsics.append(intrinsics)
        image_path = os.path.join(scene_dir, 'images', images[k].name)
        img_path = os.path.join(os.path.dirname(image_path), os.path.basename(image_path).split('.')[0].split('_')[0])
        print("img_path", img_path)
        depth_full_path = img_path + '_depth.png'
        depth = load_depth(depth_full_path)
        depths_all.append(depth)
        data_dir = '/home/zubair/Downloads/nocs_data'
        masks, coords, class_ids, instance_ids, model_list, bboxes = process_data(img_path, depth)                
        abs_poses, model_points, all_gt_class_ids = get_GT_poses(data_dir, img_path, class_ids, instance_ids, model_list, bboxes, is_pcd_out=False)
        all_model_points.append(model_points)
        all_abs_poses.append(abs_poses)

    w2c_mats = np.stack(w2c_mats, 0)
    poses = np.linalg.inv(w2c_mats) # (N_images, 3, 4) cam2world matrices
    poses = np.concatenate([poses[..., 0:1], -poses[..., 1:3], poses[..., 3:4]], -1)
    print("poses", poses.shape)
    pts_opengl = np.squeeze(pts_world, axis=0)
    return poses, pts_opengl, focal, all_model_points, all_abs_poses, depths_all, all_intrinsics
    
if __name__ == '__main__':
    import os

    # instance_name = 'camera_batch-2_1'
    # base_dir = '/home/zubair/nerf_pl/Objectron_data/' + instance_name

    scene_dir = '/home/zubair/Downloads/nocs_data/Real/test/scene_1'

    poses, pts_opengl, focal, all_model_points, all_abs_poses, depths_all, all_intrinsics = get_colamap_poses_and_points3D(scene_dir)
    sphere_radius = 1.
    camera_size = 0.1
    print("pts_opengl", pts_opengl.shape)
    pts_opengl = pts_opengl.transpose(1,0)[::20]
    visualize_cameras(poses, focal, pts_opengl, None, [0, 0, 1], all_model_points, all_abs_poses, depths_all, all_intrinsics, sphere_radius, 
                      camera_size=camera_size)