import open3d as o3d
import json
import numpy as np
import json
import pytransform3d.visualizer as pv


def get_camera_frustum(img_size, focal, C2W, frustum_length=1, color=[0., 1., 0.]):
    W, H = img_size
    hfov = np.rad2deg(np.arctan(W / 2. / focal) * 2.)
    vfov = np.rad2deg(np.arctan(H / 2. / focal) * 2.)
    half_w = frustum_length * np.tan(np.deg2rad(hfov / 2.))
    half_h = frustum_length * np.tan(np.deg2rad(vfov / 2.))
    frustum_points = np.array([[0., 0., 0.],                          # frustum origin
                               [-half_w, -half_h, frustum_length],    # top-left image corner
                               [half_w, -half_h, frustum_length],     # top-right image corner
                               [half_w, half_h, frustum_length],      # bottom-right image corner
                               [-half_w, half_h, frustum_length]])    # bottom-left image corner
    frustum_lines = np.array([[0, i] for i in range(1, 5)] + [[i, (i+1)] for i in range(1, 4)] + [[4, 1]])
    frustum_colors = np.tile(np.array(color).reshape((1, 3)), (frustum_lines.shape[0], 1))
    frustum_points = np.dot(np.hstack((frustum_points, np.ones_like(frustum_points[:, 0:1]))), C2W.T)
    frustum_points = frustum_points[:, :3] / frustum_points[:, 3:4]
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

def visualize_poses(all_c2w, fov, obj_location, sphere_radius=1, bbox_dimension= None):
    T = np.eye(4)
    T[:3,3] = obj_location
    things_to_draw = []
    focal = (640 / 2) / np.tan((fov / 2) / (180 / np.pi))
    frustums = []
    fig = pv.figure()
    for C2W in all_c2w:
        img_size = (640, 640)
        frustums.append(get_camera_frustum(img_size, focal, C2W, frustum_length=0.5, color=[0,1,0]))

    # print("frustums", len(frustums))
    if bbox_dimension is not None:
        if isinstance(bbox_dimension, dict):
            all_R = bbox_dimension['R']
            all_T = bbox_dimension['T']
            all_s = bbox_dimension['s']
            
            for Rot,Tran,sca in zip(all_R, all_T, all_s):
                bbox = o3d.geometry.OrientedBoundingBox(center = Tran, R = Rot, extent=sca)
                things_to_draw.append(bbox)


    cameras = frustums2lineset(frustums)
    things_to_draw.append(cameras)

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius, resolution=5)
    sphere = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=np.array([0., 0., 0.]))
    sphere.paint_uniform_color((1, 0, 0))
    sphere.transform(T)
    coordinate_frame.transform(T)
    things_to_draw.append(sphere)
    things_to_draw.append(coordinate_frame)
    for geometry in things_to_draw:
        fig.add_geometry(geometry)
    for C2W in all_c2w:
        fig.plot_transform(A2B=C2W, s=0.5, strict_check=False)
    fig.show()

def convert_pose_PD_to_NeRF(C2W):

    flip_axes = np.array([[1,0,0,0],
                         [0,0,-1,0],
                         [0,1,0,0],
                         [0,0,0,1]])
    C2W = np.matmul(C2W, flip_axes)
    return C2W


def read_poses(pose_path):
    pose_file = os.path.join(pose_path)
    with open(pose_file, "r") as read_content:
        data = json.load(read_content)
    obj_location = data['obj_location']
    focal = data['focal']
    fov = data['fov']
    img_wh = data['img_size']
    all_c2w = []
    all_image_names = []
    for k,v in data['transform'].items():
        all_c2w.append(np.array(v))
        all_image_names.append(k)
    
    bbox_dimensions = data['bbox_dimensions']
    all_translations = data['obj_translations']
    all_rotations = data["obj_rotations"]
    return all_c2w, focal, fov, img_wh, bbox_dimensions, obj_location, all_rotations, all_translations



def read_poses_hemispherical(pose_dir_train):
    pose_file_train = os.path.join(pose_dir_train)
    with open(pose_file_train, "r") as read_content:
        data = json.load(read_content)

    focal = data['focal']
    fov = data['fov']
    img_wh = data['img_size']
    obj_location = np.array(data["obj_location"])
    all_c2w = []
    all_image_names = []
    for k,v in data['transform'].items():
        c2w = np.array(v)
        # all_c2w.append(convert_pose_PD_to_NeRF(c2w))
        all_c2w.append(c2w)
        all_image_names.append(k)

    all_boxes = []
    for k,v in data['bbox_dimensions'].items():
            bbox = np.array(v)
            bbox_dimension =  [(bbox[1,0]-bbox[0,0]), (bbox[1,1]-bbox[0,1]), (bbox[1,2]-bbox[0,2])]
            all_boxes.append(np.array(bbox_dimension))
    
    all_translations = (np.array(data['obj_translations']))
    all_rotations = data["obj_rotations"]

    bbox_dimension_modified = {'R': all_rotations, 'T': all_translations, 's': all_boxes}
    return all_c2w, focal, fov, img_wh, bbox_dimension_modified, obj_location, all_rotations, all_translations, all_boxes


if __name__ == '__main__':
    import os
    import json

    hemispherical = True
    if hemispherical:
        pose_path = '/home/zubairirshad/pd_poses/hemispherical.json'
    
    all_c2w, focal, fov, img_size, bbox_dimensions, obj_location, all_R, all_T, all_bboxes = read_poses_hemispherical(pose_path)

    bbox_dimension_modified = {'R': all_R, 'T': all_T, 's': all_bboxes}

    visualize_poses(all_c2w, fov=fov, sphere_radius = 1.0, obj_location = obj_location, bbox_dimension= bbox_dimension_modified)