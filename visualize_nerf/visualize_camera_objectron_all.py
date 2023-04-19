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
    rays_o_obj, rays_d_obj, batch_near_obj, batch_far_obj, bbox_mask = get_rays_in_bbox(rays_o, rays_d, bbox_bounds, None, axis_aligned_mat)
    
    return rays_o_obj, rays_d_obj, batch_near_obj, batch_far_obj, bbox_mask

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

def visualize_cameras(all_c2w, all_focal, points, RTs, color, sphere_radius, camera_size=0.1):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius, resolution=10)
    sphere = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
    sphere.paint_uniform_color((1, 0, 0))
    # o3d.io.write_point_cloud("../../TestData/sync.ply", pcd)

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0., 0., 0.])
    things_to_draw = [sphere, coord_frame]
    # things_to_draw = []
    idx = 0
    frustums = []
    instance_rotation = RTs['R']
    instance_translation = RTs['T']
    instance_scale = RTs['s']
    # Draw scene OBB
    bbox = o3d.geometry.OrientedBoundingBox(center=instance_translation, R=instance_rotation, extent=instance_scale)
    bbox_canonical = o3d.geometry.OrientedBoundingBox(center = [0,0,0], R = np.eye(3), extent=instance_scale)
    things_to_draw.append(bbox)
    things_to_draw.append(bbox_canonical)

    for C2W, focal in zip(all_c2w, all_focal):
        idx += 1

        cnt = 0
        
        img_size = (120, 160)
        frustums.append(get_camera_frustum(img_size, focal, C2W, frustum_length=camera_size, color=color))
        cnt += 1
    cameras = frustums2lineset(frustums)
    things_to_draw.append(cameras)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    things_to_draw.append(pcd)

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

    focal = all_focal[0]
    focal = focal/12

    directions = get_ray_directions(120, 160, focal) # (h, w, 3)
    # c2w = np.linalg.inv(all_c2w[30])
    c2w_ = all_c2w[0]

    
    c2w = torch.FloatTensor(c2w_)[:3, :4]
    #c2w = torch.FloatTensor(c2w_)

    box_transformation = np.eye(4)
    box_transformation[:3, :3] = np.reshape(instance_rotation, (3, 3))
    box_transformation[:3, -1] = instance_translation
    axis_aligned_mat = torch.FloatTensor(np.linalg.inv(box_transformation))

    rays_o, rays_d = get_rays(directions, c2w)

    rays_o, rays_d = transform_rays_to_bbox_coordinates_nocs(rays_o, rays_d, axis_aligned_mat)

    rays_o = rays_o.numpy()
    rays_d = rays_d.numpy()
    fig = pv.figure()


    # rays_o, rays_d, batch_near, batch_far, bbox_mask = get_object_rays_in_bbox(rays_o, rays_d, RTs)
    rays_o, rays_d, batch_near, batch_far, bbox_mask = get_object_rays_in_bbox(rays_o, rays_d, RTs, canonical=True)

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

    ids = np.random.choice(rays_o.shape[0], int(rays_o.shape[0]*0.5))

    rays_o = rays_o[ids, :]
    rays_d = rays_d[ids, :]
    batch_near = batch_near[ids]
    batch_far = batch_far[ids]

    print("min max near", torch.min(batch_near), torch.max(batch_near))
    print("min max far", torch.min(batch_far), torch.max(batch_far))
    for j in range(200):
        start = rays_o[j,:]
        end = rays_o[j,:] + rays_d[j,:]*batch_near[j, :].cpu().numpy()
        line = np.concatenate((start[None, :],end[None, :]), axis=0)
        fig.plot(line, c=(1.0, 0.5, 0.0))

        start = rays_o[j,:] + rays_d[j,:]*batch_near[j, :].cpu().numpy()
        end = rays_o[j,:] + rays_d[j,:]*batch_far[j, :].cpu().numpy()
        line = np.concatenate((start[None, :],end[None, :]), axis=0)
        fig.plot(line, c=(0.0, 1.0, 0.0))

    # for j in range(2500):
    #     start = rays_o[j,:]
    #     end = rays_o[j,:] + rays_d[j,:]*0.02
    #     line = np.concatenate((start[None, :],end[None, :]), axis=0)
    #     fig.plot(line, c=(1.0, 0.5, 0.0))

    #     start = rays_o[j,:] + rays_d[j,:]*0.02
    #     end = rays_o[j,:] + rays_d[j,:]*1.5
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
    for C2W in all_c2w:
        fig.plot_transform(A2B=C2W, s=0.1)

    for C2W in all_c2w:
        C2W = axis_aligned_mat @ C2W
        fig.plot_transform(A2B=C2W, s=0.1)
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
    C2W = np.matmul(C2W, flip_yz)
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

if __name__ == '__main__':
    import os

    # instance_name = 'camera_batch-2_1'
    # base_dir = '/home/zubair/nerf_pl/Objectron_data/' + instance_name
    base_dir = '/home/zubair/camera'
    folders = os.listdir(base_dir)
    folders.sort()
    for i, instance_name in enumerate(folders):    
        # if folder in list_folders:
        #     continue
        # if i<17:
        #     continue
        print("number", i)
        print("=======================\n\n")
        print("instance_name name", instance_name)
        print("=======================\n\n")
        base_dir_objectron = base_dir + '/'+ instance_name
        print("base_dir", base_dir)
        annotation_data, instances = get_frame_annotation(os.path.join(base_dir_objectron, instance_name+'.pbdata'))
        instance_rotation, instance_translation, instance_scale, instance_vertices_3d = instances[0]
        instance_rotation = np.reshape(instance_rotation, (3, 3))

        RTs = {'R':instance_rotation, 'T':instance_translation, 's':instance_scale.T}


        sfm_arframe_filename = base_dir_objectron+'/'+instance_name+'_sfm_arframe.pbdata' 
        #json_files = [pos_json for pos_json in os.listdir(base_dir) if pos_json.endswith('.json')]
        
        frame_data = load_frame_data(sfm_arframe_filename)
        points = feature_points_from_frame_data(frame_data)
        dists = np.linalg.norm(points-points.mean(0), axis=1)
        # Get rid of outliers for visualization.
        points = points[dists<np.percentile(dists, 92)]
        all_c2w, all_focal = make_poses_bounds_array(frame_data, near=0.2, far=10)
        sphere_radius = 1.
        # train_cam_dict = json.load(open(os.path.join(base_dir, 'train/cam_dict_norm.json')))
        # test_cam_dict = json.load(open(os.path.join(base_dir, 'test/cam_dict_norm.json')))
        # path_cam_dict = json.load(open(os.path.join(base_dir, 'camera_path/cam_dict_norm.json')))
        camera_size = 0.1
        # colored_camera_dicts = [([0, 1, 0], train_cam_dict),
        #                         ([0, 0, 1], test_cam_dict),
        #                         ([1, 1, 0], path_cam_dict)
        #                         ]

        # geometry_file = os.path.join(base_dir, 'scene.ply')
        # all_c2w = all_c2w[::5]
        # points = points[::5]
        # all_focal = all_focal[::5]
        geometry_type = 'mesh'
        visualize_cameras(all_c2w, all_focal,points, RTs, [0, 0, 1], sphere_radius, 
                        camera_size=camera_size)