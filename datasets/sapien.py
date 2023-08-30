import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T

from .ray_utils import *

class SapienDataset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(320, 240), model_type = None, white_back = None, eval_inference= None):
        self.root_dir = root_dir
        self.split = split
        self.img_wh = img_wh
        self.define_transforms()

        self.read_meta()
        self.white_back = white_back

        w,h = self.img_wh
        if eval_inference is not None:
            num = len(self.img_files_val)
            self.image_sizes = np.array([[h, w] for i in range(num)])
        else:
            self.image_sizes = np.array([[h, w] for i in range(1)])

    def read_meta(self):

        base_dir = self.root_dir
        instance_dir = 'laptop'
        instance_id = '10211'
        degree_id = '80_degree'
        
        if self.split == 'train':
            # base_dir_train = os.path.join(base_dir, instance_dir, instance_id, degree_id, 'train')
            base_dir_train = os.path.join(base_dir, 'train')
            img_files_train = os.listdir(os.path.join(base_dir_train, 'rgb'))
            pose_path_train = os.path.join(base_dir_train, 'transforms.json')
            self.meta = json.load(open(pose_path_train))
        
        elif self.split =='val':

            # self.base_dir_val = os.path.join(base_dir, instance_dir, instance_id, degree_id, 'val')
            self.base_dir_val = os.path.join(base_dir, 'val')
            self.img_files_val = os.listdir(os.path.join(self.base_dir_val, 'rgb'))
            sorted_indices = np.argsort([int(filename.split('_')[1].split('.')[0]) for filename in self.img_files_val])
            self.img_files_val = [self.img_files_val[i] for i in sorted_indices]
            pose_path_val = os.path.join(self.base_dir_val, 'transforms.json')
            self.meta = json.load(open(pose_path_val))
        else:
            # self.base_dir_val = os.path.join(base_dir, instance_dir, instance_id, degree_id, 'test')
            self.base_dir_val = os.path.join(base_dir, 'test')
            self.img_files_val = os.listdir(os.path.join(self.base_dir_val, 'rgb'))
            sorted_indices = np.argsort([int(filename.split('_')[1].split('.')[0]) for filename in self.img_files_val])
            self.img_files_val = [self.img_files_val[i] for i in sorted_indices]
            pose_path_val = os.path.join(self.base_dir_val, 'transforms.json')
            self.meta = json.load(open(pose_path_val))
            
        w, h = self.img_wh

        cam_x = self.meta.get('camera_angle_x', False)
        if cam_x:
            self.focal = 0.5*h/np.tan(0.5*self.meta['camera_angle_x'])
            self.focal *= self.img_wh[0]/320 # modify focal length to match size self.img_wh
        else:
            self.focal = self.meta.get('focal', None)
            if self.focal is None:
                raise ValueError('focal length not found in transforms.json')

        # bounds, common for all scenes
        self.near = 2.0
        self.far = 6.0
        # pose_scale_factor =  0.2512323810155881
        #obtained after pose_scale_factor = 1. / np.max(np.abs(all_c2w_train[:, :3, 3]))

        self.bounds = np.array([self.near, self.far])
        
        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(h, w, self.focal) # (h, w, 3)
            
        if self.split == 'train': # create buffer of all rays and rgb data
            self.poses = []
            self.all_rays = []
            self.all_rays_d = []
            self.all_rgbs = []

            for img_file in img_files_train:
                pose = np.array(self.meta['frames'][img_file.split('.')[0]])
                self.poses += [pose]
                # c2w = torch.FloatTensor(pose)

                image_path = os.path.join(base_dir_train, 'rgb', img_file)
                img = Image.open(image_path)
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img) # (4, h, w)
                img = img.view(4, -1).permute(1, 0) # (h*w, 4) RGBA
                img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB
                self.all_rgbs += [img]
                c2w = torch.FloatTensor(pose)[:3, :4]
                rays_o, view_dirs, rays_d, radii = get_rays(self.directions, c2w, output_view_dirs=True, output_radii=True)
                #rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)
                
                self.all_rays += [torch.cat([rays_o, rays_d, 
                                             self.near*torch.ones_like(rays_o[:, :1]),
                                             self.far*torch.ones_like(rays_o[:, :1])],
                                             1)] # (h*w, 8)
                self.all_rays_d+=[view_dirs]

            self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_rays_d = torch.cat(self.all_rays_d, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w, 3)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        if self.split == 'val':
            return 1 # only validate 8 images (to support <=8 gpus)
        else:
            return len(self.img_files_val) # return for testset

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            sample = {'rays_o': self.all_rays[idx][:3],
                      'rays_d': self.all_rays_d[idx],
                      'viewdirs': self.all_rays[idx][3:6],
                      'target' : self.all_rgbs[idx]}

        else: # create data for each image separately
            img_file = self.img_files_val[idx]
            c2w = np.array(self.meta['frames'][img_file.split('.')[0]])
            c2w = torch.FloatTensor(c2w)[:3, :4]
            img = Image.open(os.path.join(self.base_dir_val, 'rgb', img_file))
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img) # (4, H, W)
            valid_mask = (img[-1]>0).flatten() # (H*W) valid color area
            img = img.view(4, -1).permute(1, 0) # (H*W, 4) RGBA
            img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB

            # rays_o, rays_d = get_rays(self.directions, c2w)
            rays_o, view_dirs, rays_d, radii = get_rays(self.directions, c2w, output_view_dirs=True, output_radii=True)

            rays = torch.cat([rays_o, rays_d, 
                              self.near*torch.ones_like(rays_o[:, :1]),
                              self.far*torch.ones_like(rays_o[:, :1])],
                              1) # (H*W, 8)

            sample = {'rays_o' :rays[:,:3],
                    'rays_d' : view_dirs,
                    'viewdirs' : rays[:,3:6],
                    'instance_mask': valid_mask,             
                    'target': img}

        return sample