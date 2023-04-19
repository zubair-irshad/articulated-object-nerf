from curses import resize_term
import open3d as o3d
import json
import numpy as np
import json


import numpy as np
import os
import json
import sys
sys.path.append('/home/zubairirshad/nerf_pl')
from visualize_nerf.transform_utils import *
from visualize_nerf.utils import *
from visualize_nerf.viz_utils import *

TINY_NUMBER = 1e-6      # float32 only has 7 decimal digits precision
def angular_dist_between_2_vectors(vec1, vec2):
    vec1_unit = vec1 / (np.linalg.norm(vec1, axis=1, keepdims=True) + TINY_NUMBER)
    vec2_unit = vec2 / (np.linalg.norm(vec2, axis=1, keepdims=True) + TINY_NUMBER)
    angular_dists = np.arccos(np.clip(np.sum(vec1_unit*vec2_unit, axis=-1), -1.0, 1.0))
    return angular_dists

def batched_angular_dist_rot_matrix(R1, R2):
    '''
    calculate the angular distance between two rotation matrices (batched)
    :param R1: the first rotation matrix [N, 3, 3]
    :param R2: the second rotation matrix [N, 3, 3]
    :return: angular distance in radiance [N, ]
    '''
    assert R1.shape[-1] == 3 and R2.shape[-1] == 3 and R1.shape[-2] == 3 and R2.shape[-2] == 3
    return np.arccos(np.clip((np.trace(np.matmul(R2.transpose(0, 2, 1), R1), axis1=1, axis2=2) - 1) / 2.,
                             a_min=-1 + TINY_NUMBER, a_max=1 - TINY_NUMBER))

def get_nearest_pose_ids(tar_pose, ref_poses, num_select=4, tar_id=-1, angular_dist_method='vector',
                         scene_center=(0, 0, 0)):
    '''
    Args:
        tar_pose: target pose [3, 3]
        ref_poses: reference poses [N, 3, 3]
        num_select: the number of nearest views to select
    Returns: the selected indices
    '''
    num_cams = len(ref_poses)
    num_select = min(num_select, num_cams-1)
    batched_tar_pose = tar_pose[None, ...].repeat(num_cams, 0)

    if angular_dist_method == 'matrix':
        dists = batched_angular_dist_rot_matrix(batched_tar_pose[:, :3, :3], ref_poses[:, :3, :3])
    elif angular_dist_method == 'vector':
        tar_cam_locs = batched_tar_pose[:, :3, 3]
        ref_cam_locs = ref_poses[:, :3, 3]
        scene_center = np.array(scene_center)[None, ...]
        tar_vectors = tar_cam_locs - scene_center
        ref_vectors = ref_cam_locs - scene_center
        dists = angular_dist_between_2_vectors(tar_vectors, ref_vectors)
    elif angular_dist_method == 'dist':
        tar_cam_locs = batched_tar_pose[:, :3, 3]
        ref_cam_locs = ref_poses[:, :3, 3]
        dists = np.linalg.norm(tar_cam_locs - ref_cam_locs, axis=1)
    else:
        raise Exception('unknown angular distance calculation method!')

    if tar_id >= 0:
        assert tar_id < num_cams
        dists[tar_id] = 1e3  # make sure not to select the target id itself

    sorted_ids = np.argsort(dists)
    selected_ids = sorted_ids[:num_select]
    # print(angular_dists[selected_ids] * 180 / np.pi)
    return selected_ids



if __name__ == '__main__':

    relative_rot_trans_dict = {}

    new = True
    scene_path = '/home/zubairirshad/pd-api-py/single_scene_23/SF_6thAndMission_medium23'
    if new:
        base_dir_train = os.path.join(scene_path, 'train')
        base_dir_val = os.path.join(scene_path, 'val')

        img_files_train = os.listdir(os.path.join(base_dir_train, 'rgb'))
        img_files_train.sort()
        all_c2w_train, all_c2w_val, qgmagma, fov, img_size, bbox_dimension_modified, RTs, pose_scale_factor, obj_location = read_poses(pose_dir_train = os.path.join(base_dir_train, 'pose'), img_files_train = img_files_train)
        all_c2w =  np.concatenate((all_c2w_train, all_c2w_val), axis=0)
        id_render = np.random.randint(0, 98)+100
        
        tar_pose = all_c2w[id_render]
        selected_ids = get_nearest_pose_ids(tar_pose, all_c2w_train)
        print("id_render", id_render)
        print("selected_ids", selected_ids[1:])

        # all_c2w =  np.concatenate((all_c2w_train, all_c2w_val), axis=0)

        # print("len(all_c2w)", len(all_c2w), len(all_c2w_val))

        # num_cameras = len(all_c2w_train)
        # rel_translations_train = np.zeros((num_cameras, num_cameras, 3))
        # rel_rotations_train = np.zeros((num_cameras, num_cameras))
        # for i in range(num_cameras):
        #     for j in range(i+1, num_cameras):
        #         rel_translations_train[i, j] = all_c2w_train[j][:3, 3] - all_c2w_train[i][:3, 3]
        #         rel_translations_train[j, i] = -rel_translations_train[i, j]
                
        #         R12 = all_c2w_train[i] @ all_c2w_train[j]
        #         R_error = np.arccos(np.clip((np.trace(R12)-1)/2, -1.0, 1.0)) * 180 / np.pi
        #         rel_rotations_train[i, j] = R_error
        #         rel_rotations_train[j, i] = rel_rotations_train[i, j]

        # num_cameras = len(all_c2w)
        # rel_translations_val = np.zeros((num_cameras, num_cameras, 3))
        # rel_rotations_val = np.zeros((num_cameras, num_cameras))
        # for i in range(num_cameras):
        #     for j in range(i+1, num_cameras):

        #         rel_translations_val[i, j] = all_c2w[j][:3, 3] - all_c2w[i][:3, 3]
        #         rel_translations_val[j, i] = -rel_translations_val[i, j]
                
        #         R12 = all_c2w[i] @ all_c2w[j]
        #         R_error = np.arccos(np.clip((np.trace(R12)-1)/2, -1.0, 1.0)) * 180 / np.pi
        #         rel_rotations_val[i, j] = R_error
        #         rel_rotations_val[j, i] = rel_rotations_val[i, j]

        # relative_rot_trans_dict["rel_rotations_train"] = rel_rotations_train
        # relative_rot_trans_dict["rel_translations_train"] = rel_translations_train
        # relative_rot_trans_dict["rel_translations_val"] = rel_translations_val
        # relative_rot_trans_dict["rel_rotations_val"] = rel_rotations_val

        # import pickle

        # with open("relative_rot_trans_dict.pkl", "wb") as f:
        #     pickle.dump(relative_rot_trans_dict, f)
        # print("rel_translations", rel_translations)
        # print("==========================\n\n\n")


        # print("rel_translations", rel_rotations)
        # print("==========================\n\n\n")

        # print("rel_translations")


        # print("rel_rotations 31 89", rel_rotations[31,89])


        # src = 49
        # dest = 30
        # print("rel_rotations",src, dest, ":", rel_rotations_val[src,dest])

        # print("rel_translations",src, dest, ":", np.linalg.norm(rel_translations_val[src, dest]))

        # # tran_error <0.4
        # # rot_error <90

        # src_view_num = [49, 38, 44]

        # import time
        # start_time = time.time()
        # dest_view_nums = []
        # for dest_camera in range(len(all_c2w)):
        #     if dest_camera not in src_view_num:
        #         for src_camera in src_view_num:
        #             if np.linalg.norm(rel_translations_val[src_camera, dest_camera]) < 0.4 and rel_rotations_val[src_camera, dest_camera] < 90:
        #                 # The destination camera satisfies the conditions
        #                 # Select it as a key frame and break out of the loop
        #                 dest_view_nums.append(dest_camera)
        #                 break
        #     else:
        #         continue

        # print("time naive", time.time()-start_time)

        # print("dest_view_nums",dest_view_nums)


        # def select_key_frames(num_cameras, source_cameras, rel_translations, rel_rotations, tmax, Rmax):
        #     # Create boolean masks for source cameras and valid destination cameras
        #     is_source_camera = np.zeros(num_cameras, dtype=bool)
        #     is_source_camera[source_cameras] = True
        #     is_valid_dest_camera = ~is_source_camera
            
        #     # Create boolean masks for pairs of cameras that satisfy the conditions
        #     trans_mask = np.linalg.norm(rel_translations, axis=-1) < tmax
        #     rot_mask = np.abs(rel_rotations) < Rmax
            
        #     # Use boolean masks to select valid pairs of cameras
        #     valid_pairs = np.logical_and(np.logical_and(trans_mask, rot_mask), np.outer(is_source_camera, is_valid_dest_camera))
            
        #     # Find the indices of valid destination cameras
        #     key_frames = np.where(np.any(valid_pairs, axis=0))[0].tolist()
            
        #     # Return the list of key frames
        #     return key_frames
        
        # start_time = time.time()
        # dest_view_nums = select_key_frames(len(all_c2w), src_view_num, rel_translations_val, rel_rotations_val, 0.4, 90)

        # print("time naive", time.time()-start_time)

        # print("dest_view_nums",dest_view_nums)
            
