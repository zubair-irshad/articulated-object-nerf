import sapien.core as sapien
import numpy as np
from PIL import Image, ImageColor
import open3d as o3d
from sapien.utils.viewer import Viewer
from transforms3d.euler import mat2euler
import math
import random
import matplotlib.pyplot as plt
from pathlib import Path as P
import json
from tqdm import tqdm
# camera position coordinate: https://sapien.ucsd.edu/docs/2.2/tutorial/basic/hello_world.html, check the viewer section

# openGL coordinate definition: https://medium.com/@christophkrautz/what-are-the-coordinates-225f1ec0dd78

# from camera position coordinate to openGL: x = -y, y = z, z = -x

conversion_matrix = np.array([
    [0, -1, 0], 
    [0, 0, 1], 
    [-1, 0, 0]
])

def min_max_depth(depth):
    max_depth = depth.max()
    min_depth = depth[depth>0].min()
    return min_depth, max_depth

def model_rot_cvt_trans(camera):
    model_mat = camera.get_model_matrix()
    model_trans = model_mat[:3, -1:]

    # from forward(x), left(y) and up(z), to right(x), up(y), backwards(z)
    cvt_matrix_3x3 = np.array([
        [0, -1, 0],  # left(y) -> right(x)
        [0, 0, 1], # up(z) -> up(y)
        [-1, 0, 0] # forward(x) -> backward(z)
    ])
    new_trans = np.dot(cvt_matrix_3x3, model_trans)
    model_mat[:3, -1:] = new_trans
    return model_mat

def calculate_pose_openGL(translation):
    """
    recalculate the rotation matrix for camera extrinsic, camera is facing the origin
    input
        @param translation: object position given in viwer coordinate, row vector
        
    """
    trans_gl = np.dot(conversion_matrix, translation.T) # permute
    forward = -trans_gl / np.linalg.norm(trans_gl)
    right = np.cross([0, 1, 0], forward)
    right = right / np.linalg.norm(right)
    up = np.cross(forward, right)
    mat44 = np.eye(4)
    mat44[:3, :3] = np.stack([right, up, forward], axis=1)
    mat44[:3, 3] = trans_gl
    return mat44

def custom_openGL(camera):
    model_mat = camera.pose.to_transformation_matrix()
    model_trans = model_mat[:3, -1:]
    return calculate_pose_openGL(model_trans.reshape(-1))

def random_point_in_sphere(radius, theta_range=[0, 2*math.pi], phi_range=[0, math.pi]):
    # Generate random spherical coordinates
    theta_low, theta_high = theta_range
    phi_low, phi_high = phi_range
    
    theta = random.uniform(theta_low, theta_high)       # Azimuthal angle
    phi = random.uniform(phi_low, phi_high)            # Polar angle
    r = random.uniform(radius-0.5, radius+0.5)      # Radial distance
    
    # Convert spherical coordinates to Cartesian coordinates
    x = r * math.sin(phi) * math.cos(theta)
    y = r * math.sin(phi) * math.sin(theta)
    z = r * math.cos(phi)
    
    return x, y, z

def point_in_sphere(r, theta, phi):
    x = r * math.sin(phi) * math.cos(theta)
    y = r * math.sin(phi) * math.sin(theta)
    z = r * math.cos(phi)
    
    return x, y, z

def get_depth(camera):
    position = camera.get_float_texture('Position')  # [H, W, 4]
    # Depth
    depth = -position[..., 2]
    depth_image = (depth * 1000.0).astype(np.uint16)
    depth_pil = Image.fromarray(depth_image)
    return depth_pil

def get_joint_type(asset):
    joints = asset.get_joints()
    j_type = []
    for joint in joints:
        if joint.get_dof() != 0:
            j_type += [joint.type[0]] * joint.get_dof()
    return j_type

def calculate_cam_ext(point):
    cam_pos = np.array(point)
    # def update_cam_pose(cam_pos):
    forward = -cam_pos / np.linalg.norm(cam_pos)
    left = np.cross([0, 0, 1], forward)
    left = left / np.linalg.norm(left)
    up = np.cross(forward, left)
    mat44 = np.eye(4)
    mat44[:3, :3] = np.stack([forward, left, up], axis=1)
    mat44[:3, 3] = cam_pos
    return mat44

def render_img(point, save_path, camera_mount_actor, scene, camera, asset, q_pos=None, pose_fn=None, save=True):
    mat44 = calculate_cam_ext(point)
    if camera_mount_actor is None:
        camera.set_pose(sapien.Pose.from_transformation_matrix(mat44))
    else:
        camera_mount_actor.set_pose(sapien.Pose.from_transformation_matrix(mat44))
    if q_pos is not None:
        asset.set_qpos(q_pos)

    scene.step()  # make everything set
    scene.update_render()
    camera.take_picture()

    rgba = camera.get_float_texture('Color')  # [H, W, 4]
        
    rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
    

    seg_labels = camera.get_uint32_texture('Segmentation')  # [H, W, 4]
    mask = seg_labels.sum(axis=-1)
    mask[mask>0] = 1
    rgba_img[:, :, -1] = rgba_img[:, :, -1] * mask
    
    rgba_pil = Image.fromarray(rgba_img, 'RGBA')
    colormap = sorted(set(ImageColor.colormap.values()))
    color_palette = np.array([ImageColor.getrgb(color) for color in colormap],
                                dtype=np.uint8)
    label0_image = seg_labels[..., 0].astype(np.uint8)  # mesh-level
    label1_image = seg_labels[..., 1].astype(np.uint8)  # actor-level
    label0_pil = Image.fromarray(color_palette[label0_image])
    label1_pil = Image.fromarray(color_palette[label1_image])
    label2_pil = Image.fromarray(label1_image)
    camera_pose = camera.get_pose()
    qpos = asset.get_qpos()
    if pose_fn is not None:
        save_pose = pose_fn(camera).tolist()
    else:
        save_pose = camera.get_extrinsic_matrix().tolist()
    model_mat = camera.get_model_matrix()
    cv_ext = camera.get_extrinsic_matrix()
    meta_dict={
        "pose": save_pose,
        "ext_pose": camera.get_extrinsic_matrix().tolist(),
        "model_mat": model_mat.tolist(),
        "qpos": qpos.tolist(),
        "joint_type": get_joint_type(asset),
        "cam_param": camera.get_intrinsic_matrix().tolist()}
        
    depth_pil = get_depth(camera)
    min_d, max_d = min_max_depth(np.array(depth_pil))
    if save:
        depth_pil.save(str(save_path/'depth.png'))
        label0_pil.save(str(save_path/'label0.png'))
        label1_pil.save(str(save_path/'label1.png'))
        label2_pil.save(str(save_path/'label_actor.png'))
        rgba_pil.save(str(save_path/'color.png'))
        json_fname = str(save_path/'meta.json')
        with open(json_fname, 'w') as f:
            json.dump(meta_dict, f)
    ret_dict = {
        'rgba': rgba_pil,
        'depth': depth_pil,
        'label_0': label0_pil,
        'label_1': label1_pil,
        'label_actor': label2_pil,
        'meta': meta_dict,
        'min_d': min_d,
        'max_d': max_d,
        'mat44': mat44
    }
    return ret_dict

def gen_articulated_object_nerf_s1(num_pos_img, radius_, split, camera, asset, scene, object_path, camera_mount_actor=None, theta_range = [0*math.pi, 2*math.pi], phi_range = [0*math.pi, 1*math.pi], render_pose_file_dir = None):
    save_base_path = object_path / split
    save_base_path.mkdir(exist_ok=True)
    save_rgb_path = save_base_path / 'rgb'
    save_rgb_path.mkdir(exist_ok=True)
    save_depth_path = save_base_path / 'depth'
    save_depth_path.mkdir(exist_ok=True)
    
    render_pose_dict = {}
    # j_types = get_joint_type(asset)
    transform_json = {
        "focal": camera.fy
    }
    frame_dict = dict()
    max_d = 0
    min_d = np.inf
    for i in tqdm(range(num_pos_img)):
        instance_save_path = None
        point = random_point_in_sphere(radius=radius_, theta_range=theta_range, phi_range=phi_range)
        # point = points[i]
        ret_dict = render_img(point, instance_save_path, camera_mount_actor, scene, camera, asset, pose_fn=custom_openGL, save=False)
        frame_id = 'r_'+str(i)
        c2w = camera.get_model_matrix()
        frame_dict[frame_id] = c2w.tolist()
        
        render_pose = ret_dict['mat44']
        render_pose_dict[frame_id] = render_pose.tolist()
        
        rgb_fname = save_rgb_path / (frame_id + '.png')
        rgba_pil = ret_dict['rgba']
        rgba_pil.save(str(rgb_fname))   
        
        depth_fname = save_depth_path / ('depth' + str(i) + '.png')
        depth_pil = ret_dict['depth']
        depth_pil.save(str(depth_fname))
        
        if ret_dict['max_d'] > max_d:
            max_d = ret_dict['max_d'] 
        if ret_dict['min_d'] < min_d:
            min_d = ret_dict['min_d']  
    print('min_d = ', min_d)
    print('max_d = ', max_d)

    transform_json['frames'] = frame_dict
    transform_fname = str(save_base_path / 'transforms.json')
    if render_pose_file_dir is not None:
        P(render_pose_file_dir).mkdir(parents=True, exist_ok=True)
        render_pose_fname = P(render_pose_file_dir) / (split + '.json')
        with open(render_pose_fname, 'w') as f:
            json.dump(render_pose_dict, f)
            
    with open(transform_fname, 'w') as f:
        json.dump(transform_json, f)
    pass

def generate_img_with_pose(pose_dir, split, camera, asset, scene, object_path, camera_mount_actor=None):
    save_base_path = object_path / split
    save_base_path.mkdir(exist_ok=True)
    save_rgb_path = save_base_path / 'rgb'
    save_rgb_path.mkdir(exist_ok=True)
    save_depth_path = save_base_path / 'depth'
    save_depth_path.mkdir(exist_ok=True)
    render_pose_dict = {}
    # j_types = get_joint_type(asset)
    transform_json = {
        "focal": camera.fy
    }
    frame_dict = dict()
    max_d = 0
    min_d = np.inf
    # load camera pose
    pose_fname = P(pose_dir) / (split + '.json')
    
    print('generating images from saved pose file: ', pose_fname)
    render_pose = json.load(open(str(pose_fname)))
    for frame_id in tqdm(render_pose.keys()):
        img_pose = np.array(render_pose[frame_id])
        
        ret_dict = render_img_with_pose(img_pose, None, camera_mount_actor, scene, camera, asset, save=False)
        rgb_fname = save_rgb_path / (frame_id + '.png')
        rgba_pil = ret_dict['rgba']
        rgba_pil.save(str(rgb_fname)) 
        c2w = camera.get_model_matrix()
        frame_dict[frame_id] = c2w.tolist()
        depth_fname = save_depth_path / ('depth' + frame_id[2:] + '.png')
        depth_pil = ret_dict['depth']
        depth_pil.save(str(depth_fname))
        if ret_dict['max_d'] > max_d:
            max_d = ret_dict['max_d'] 
        if ret_dict['min_d'] < min_d:
            min_d = ret_dict['min_d']  
    print('min_d = ', min_d)
    print('max_d = ', max_d)
    transform_json['frames'] = frame_dict
    transform_fname = str(save_base_path / 'transforms.json')
            
    with open(transform_fname, 'w') as f:
        json.dump(transform_json, f)
    
    pass

def render_img_with_pose(pose, save_path, camera_mount_actor, scene, camera, asset, q_pos=None, pose_fn=None, save=True):
    mat44 = pose
    if camera_mount_actor is None:
        camera.set_pose(sapien.Pose.from_transformation_matrix(mat44))
    else:
        camera_mount_actor.set_pose(sapien.Pose.from_transformation_matrix(mat44))
    if q_pos is not None:
        asset.set_qpos(q_pos)

    scene.step()  # make everything set
    scene.update_render()
    camera.take_picture()

    rgba = camera.get_float_texture('Color')  # [H, W, 4]
        
    rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
    

    seg_labels = camera.get_uint32_texture('Segmentation')  # [H, W, 4]
    mask = seg_labels.sum(axis=-1)
    mask[mask>0] = 1
    rgba_img[:, :, -1] = rgba_img[:, :, -1] * mask
    
    rgba_pil = Image.fromarray(rgba_img, 'RGBA')
    colormap = sorted(set(ImageColor.colormap.values()))
    color_palette = np.array([ImageColor.getrgb(color) for color in colormap],
                                dtype=np.uint8)
    label0_image = seg_labels[..., 0].astype(np.uint8)  # mesh-level
    label1_image = seg_labels[..., 1].astype(np.uint8)  # actor-level
    label0_pil = Image.fromarray(color_palette[label0_image])
    label1_pil = Image.fromarray(color_palette[label1_image])
    label2_pil = Image.fromarray(label1_image)
    camera_pose = camera.get_pose()
    qpos = asset.get_qpos()
    if pose_fn is not None:
        save_pose = pose_fn(camera).tolist()
    else:
        save_pose = camera.get_extrinsic_matrix().tolist()
    model_mat = camera.get_model_matrix()
    cv_ext = camera.get_extrinsic_matrix()
    meta_dict={
        "pose": save_pose,
        "ext_pose": camera.get_extrinsic_matrix().tolist(),
        "model_mat": model_mat.tolist(),
        "qpos": qpos.tolist(),
        "joint_type": get_joint_type(asset),
        "cam_param": camera.get_intrinsic_matrix().tolist()}
        
    depth_pil = get_depth(camera)
    min_d, max_d = min_max_depth(np.array(depth_pil))
    if save:
        depth_pil.save(str(save_path/'depth.png'))
        label0_pil.save(str(save_path/'label0.png'))
        label1_pil.save(str(save_path/'label1.png'))
        label2_pil.save(str(save_path/'label_actor.png'))
        rgba_pil.save(str(save_path/'color.png'))
        json_fname = str(save_path/'meta.json')
        with open(json_fname, 'w') as f:
            json.dump(meta_dict, f)
    ret_dict = {
        'rgba': rgba_pil,
        'depth': depth_pil,
        'label_0': label0_pil,
        'label_1': label1_pil,
        'label_actor': label2_pil,
        'meta': meta_dict,
        'min_d': min_d,
        'max_d': max_d,
        'mat44': mat44
    }
    return ret_dict