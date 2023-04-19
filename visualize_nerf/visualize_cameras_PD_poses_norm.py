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

if __name__ == '__main__':

    new = True
    scene_path = '/home/zubairirshad/pd-api-py/PD_v3_eval/test_novelobj/SF_6thAndMission_medium0'
    if new:

        base_dir_train = os.path.join(scene_path, 'train')
        img_files_train = os.listdir(os.path.join(base_dir_train, 'rgb'))
        img_files_train.sort()

        base_dir_val = os.path.join(scene_path, 'val')
        img_files_test = os.listdir(os.path.join(base_dir_val, 'rgb'))
        img_files_test.sort()

        

        all_c2w, all_c2w_val, all_c2w_test, qgmagma, fov, img_size, \
           bbox_dimension_modified, RTs, pose_scale_factor, obj_location = read_poses(pose_dir_train = os.path.join(base_dir_train, 'pose'), 
                                                                                      img_files_train = img_files_train, 
                                                                                      pose_dir_test=os.path.join(base_dir_val, 'pose'), 
                                                                                      img_files_test=img_files_test)
        
        # all_c2w, all_c2w_val, qgmagma, fov, img_size, \
        #    bbox_dimension_modified, RTs, pose_scale_factor, obj_location = read_poses(pose_dir_train = os.path.join(base_dir_train, 'pose'), 
        #                                                                               img_files_train = img_files_train)
        # all_c2w_test = None
        visualize_poses(all_c2w, all_c2w_val, all_c2w_test, fov=fov, sphere_radius = 1.0, obj_location = [0,0,0], bbox_dimension= bbox_dimension_modified, RTs=RTs)