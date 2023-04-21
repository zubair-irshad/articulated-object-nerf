import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T
import random
from .ray_utils import *

idx_to_deg = {
    "train": {0: 0, 1: 10, 2: 20, 3: 30, 4: 40, 5: 50, 6: 60, 7: 70, 8: 80, 9: 90},
    "val": {0: 5, 1: 15, 2: 25, 3: 35, 4: 45, 5: 55, 6: 65, 7: 75, 8: 85},
}


def get_bbox_from_mask(inst_mask):
    # bounding box
    horizontal_indicies = np.where(np.any(inst_mask, axis=0))[0]
    vertical_indicies = np.where(np.any(inst_mask, axis=1))[0]
    x1, x2 = horizontal_indicies[[0, -1]]
    y1, y2 = vertical_indicies[[0, -1]]
    # x2 and y2 should not be part of the box. Increment by 1.
    x2 += 1
    y2 += 1
    return x1, x2, y1, y2


class SapienDatasetMulti(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        img_wh=(320, 240),
        model_type=None,
        white_back=None,
        eval_inference=None,
    ):
        self.root_dir = root_dir
        self.split = split
        self.img_wh = img_wh
        self.define_transforms()
        self.ids = np.sort([f.name for f in os.scandir(self.root_dir)])
        self.samples_per_epoch = 4000
        self.white_back = white_back
        # bounds, common for all scenes
        self.near = 2.0
        self.far = 6.0
        # self.img_transform = T.Compose([T.Resize((128, 128)), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.img_transform = T.Compose([T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        w, h = self.img_wh
        if eval_inference is not None:
            # eval_num = int(self.eval_inference[0])
            # num =  100 - eval_num
            num = 19
            self.image_sizes = np.array([[h, w] for i in range(num)])
        else:
            self.image_sizes = np.array([[h, w] for i in range(1)])

    def load_image_and_seg(self, img_path, seg_path):
        img = Image.open(img_path).convert("RGB")
        w, h = self.img_wh
        img = img.resize((w, h), Image.LANCZOS)
        # img = self.transform(img) # (4, h, w)
        # img = img.view(4, -1).permute(1, 0) # (h*w, 4) RGBA
        # img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB
        seg_mask = Image.open(seg_path)
        seg_mask = seg_mask.resize((w, h), Image.LANCZOS)
        seg_mask = np.array(seg_mask)
        seg_mask = seg_mask > 0

        return img, seg_mask

    def get_cropped_img_seg(self, img, instance_mask):
        w, h = self.img_wh
        x1, x2, y1, y2 = get_bbox_from_mask(instance_mask)

        if self.white_back:
            rgb_masked = np.ones((h, w, 3), dtype=np.uint16) * 255
        else:
            rgb_masked = np.zeros((h, w, 3), dtype=np.uint16) * 255
        # black background
        instance_mask_repeat = np.repeat(instance_mask[..., None], 3, axis=2)
        rgb_masked[instance_mask_repeat] = np.array(img)[instance_mask_repeat]
        img = rgb_masked[y1:y2, x1:x2]
        instance_mask = instance_mask[y1:y2, x1:x2]

        box_2d = (x1, x2, y1, y2)

        return img, instance_mask, box_2d

    def get_masked_img_seg(self, img, instance_mask):
        w, h = self.img_wh
        if self.white_back:
            rgb_masked = np.ones((h, w, 3), dtype=np.uint16) * 255
        else:
            rgb_masked = np.zeros((h, w, 3), dtype=np.uint16) * 255
        # black background
        instance_mask_repeat = np.repeat(instance_mask[..., None], 3, axis=2)
        rgb_masked[instance_mask_repeat] = np.array(img)[instance_mask_repeat]
        img = rgb_masked

        return img, instance_mask

    def get_cropped_rays(self, rays_o, view_dirs, rays_d, box_2d):
        w, h = self.img_wh
        x1, x2, y1, y2 = box_2d
        rays_o = rays_o.reshape(h, w, 3)[y1:y2, x1:x2].contiguous().view(-1, 3)
        rays_d = rays_d.reshape(h, w, 3)[y1:y2, x1:x2].contiguous().view(-1, 3)
        view_dirs = view_dirs.reshape(h, w, 3)[y1:y2, x1:x2].contiguous().view(-1, 3)

        return rays_o, view_dirs, rays_d

    def get_ray_batch(
        self, cam_rays, cam_view_dirs, cam_rays_d, img, instance_mask, ray_batch_size
    ):
        instance_mask = T.ToTensor()(instance_mask)
        img = Image.fromarray(np.uint8(img))
        img = T.ToTensor()(img)

        cam_rays = torch.FloatTensor(cam_rays)
        cam_view_dirs = torch.FloatTensor(cam_view_dirs)
        cam_rays_d = torch.FloatTensor(cam_rays_d)
        rays = cam_rays.view(-1, cam_rays.shape[-1])
        rays_d = cam_rays_d.view(-1, cam_rays_d.shape[-1])
        view_dirs = cam_view_dirs.view(-1, cam_view_dirs.shape[-1])

        if self.split == "train":
            # instance_mask_t = instance_mask.permute(1,2,0).flatten(0,1).squeeze(-1)
            # #equal probability of foreground and background
            # N_fg = int(ray_batch_size * 0.8)
            # N_bg = ray_batch_size - N_fg
            # b_fg_inds = torch.nonzero(instance_mask_t == 1)
            # b_bg_inds = torch.nonzero(instance_mask_t == 0)
            # b_fg_inds = b_fg_inds[torch.randperm(b_fg_inds.shape[0])[:N_fg]]
            # b_bg_inds = b_bg_inds[torch.randperm(b_bg_inds.shape[0])[:N_bg]]
            # pix_inds = torch.cat([b_fg_inds, b_bg_inds], 0).squeeze(-1)
            _, H, W = img.shape
            pix_inds = torch.randint(0, H * W, (ray_batch_size,))
            src_img = self.img_transform(img)
            msk_gt = instance_mask.permute(1, 2, 0).flatten(0, 1)[pix_inds, ...]
            rgbs = img.permute(1, 2, 0).flatten(0, 1)[pix_inds, ...]
            rays = rays[pix_inds]
            rays_d = rays_d[pix_inds]
            view_dirs = view_dirs[pix_inds]

        else:
            src_img = self.img_transform(img)
            msk_gt = instance_mask.permute(1, 2, 0).flatten(0, 1)
            rgbs = img.permute(1, 2, 0).flatten(0, 1)

        return rays, rays_d, view_dirs, src_img, rgbs, msk_gt

    def read_data(self, instance_id, degree_id, image_id):
        base_dir = self.root_dir

        if self.split == "train":
            base_dir = os.path.join(base_dir, instance_id, "train", degree_id)
            img_files = os.listdir(os.path.join(base_dir, "rgb"))
            pose_path = os.path.join(base_dir, "transforms.json")
            poses = json.load(open(pose_path))

        elif self.split == "val":
            # base_dir = os.path.join(base_dir, instance_id, 'val', degree_id)
            base_dir = os.path.join(base_dir, instance_id, "train", degree_id)
            img_files = os.listdir(os.path.join(base_dir, "rgb"))
            sorted_indices = np.argsort(
                [int(filename.split("_")[1].split(".")[0]) for filename in img_files]
            )
            img_files = [img_files[i] for i in sorted_indices]
            pose_path = os.path.join(base_dir, "transforms.json")
            poses = json.load(open(pose_path))
        else:
            base_dir = os.path.join(base_dir, instance_id, "train", degree_id)
            img_files = os.listdir(os.path.join(base_dir, "rgb"))
            sorted_indices = np.argsort(
                [int(filename.split("_")[1].split(".")[0]) for filename in img_files]
            )
            img_files = [img_files[i] for i in sorted_indices]
            pose_path_val = os.path.join(base_dir, "transforms.json")
            poses = json.load(open(pose_path_val))

        w, h = self.img_wh

        focal = 0.5 * h / np.tan(0.5 * poses["camera_angle_x"])
        # 320 is the original rendered image width
        focal *= self.img_wh[0] / 320  # modify focal length to match size self.img_wh
        # pose_scale_factor =  0.2512323810155881
        # obtained after pose_scale_factor = 1. / np.max(np.abs(all_c2w_train[:, :3, 3]))

        # ray directions for all pixels, same for all images (same H, W, focal)
        directions = get_ray_directions(h, w, focal)  # (h, w, 3)

        img_file = img_files[image_id]
        c2w = np.array(poses["frames"][img_file.split(".")[0]])
        c2w = torch.FloatTensor(c2w)[:3, :4]

        img, seg = self.load_image_and_seg(
            img_path=os.path.join(base_dir, "rgb", img_file),
            seg_path=os.path.join(base_dir, "seg", img_file),
        )
        # img, seg, box_2d = self.get_cropped_img_seg(img, seg)
        img, seg = self.get_masked_img_seg(img, seg)

        rays_o, view_dirs, rays_d, _ = get_rays(
            directions, c2w, output_view_dirs=True, output_radii=True
        )
        # rays_o, view_dirs, rays_d = self.get_cropped_rays(rays_o, view_dirs, rays_d, box_2d)

        return rays_o, view_dirs, rays_d, img, seg

    def get_test_rays(self, instance_id, image_id):
        base_dir = os.path.join(base_dir, instance_id, "train", "0_degree")
        img_files = os.listdir(os.path.join(base_dir, "rgb"))
        sorted_indices = np.argsort(
            [int(filename.split("_")[1].split(".")[0]) for filename in img_files]
        )
        img_files = [img_files[i] for i in sorted_indices]
        pose_path_val = os.path.join(base_dir, "transforms.json")
        poses = json.load(open(pose_path_val))

        w, h = self.img_wh

        focal = 0.5 * h / np.tan(0.5 * poses["camera_angle_x"])
        # 320 is the original rendered image width
        focal *= self.img_wh[0] / 320  # modify focal length to match size self.img_wh
        # pose_scale_factor =  0.2512323810155881
        # obtained after pose_scale_factor = 1. / np.max(np.abs(all_c2w_train[:, :3, 3]))

        # ray directions for all pixels, same for all images (same H, W, focal)
        directions = get_ray_directions(h, w, focal)  # (h, w, 3)

        img_file = img_files[image_id]
        c2w = np.array(poses["frames"][img_file.split(".")[0]])
        c2w = torch.FloatTensor(c2w)[:3, :4]

        img, seg = self.load_image_and_seg(
            img_path=os.path.join(base_dir, "rgb", img_file),
            seg_path=os.path.join(base_dir, "seg", img_file),
        )
        # img, seg, box_2d = self.get_cropped_img_seg(img, seg)
        img, seg = self.get_masked_img_seg(img, seg)

        rays_o, view_dirs, rays_d, _ = get_rays(
            directions, c2w, output_view_dirs=True, output_radii=True
        )
        # rays_o, view_dirs, rays_d = self.get_cropped_rays(rays_o, view_dirs, rays_d, box_2d)

        return rays_o, view_dirs, rays_d, img, seg

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == "train":
            return self.samples_per_epoch
        if self.split == "val":
            return 1  # only validate 8 images (to support <=8 gpus)
        else:
            return 19
        # return len(self.meta['frames'])

    def __getitem__(self, idx):
        if self.split == "train":  # use data in the buffers
            train_idx = random.randint(0, len(self.ids) - 1)
            instance_dir = self.ids[train_idx]
            deg_paths = np.sort(
                [
                    f.name
                    for f in os.scandir(
                        os.path.join(self.root_dir, instance_dir, "train")
                    )
                ]
            )
            sorted_indices = np.argsort(
                [int(filename.split("_")[0]) for filename in deg_paths]
            )
            deg_paths = [deg_paths[i] for i in sorted_indices]

            deg_idx = random.randint(0, len(deg_paths) - 1)
            degree_dir = deg_paths[deg_idx]
            image_id = np.random.randint(0, 59)

            cam_rays, cam_view_dirs, cam_rays_d, img, seg = self.read_data(
                instance_dir, degree_dir, image_id
            )
            rays, rays_d, view_dirs, src_img, rgbs, mask = self.get_ray_batch(
                cam_rays, cam_view_dirs, cam_rays_d, img, seg, ray_batch_size=4096
            )

            sample = {}
            sample["rays_o"] = rays
            sample["rays_d"] = rays_d
            sample["viewdirs"] = view_dirs
            sample["src_imgs"] = src_img
            sample["target"] = rgbs
            sample["instance_mask"] = mask
            sample["deg"] = np.deg2rad(idx_to_deg["train"][deg_idx]).astype(np.float32)
            sample["instance_id"] = train_idx
            sample["articulation_id"] = deg_idx

        elif self.split == "val":  # create data for each image separately
            val_idx = random.randint(0, len(self.ids) - 1)
            instance_dir = self.ids[val_idx]
            deg_paths = [
                f.name
                for f in os.scandir(os.path.join(self.root_dir, instance_dir, "train"))
            ]

            sorted_indices = np.argsort(
                [int(filename.split("_")[0]) for filename in deg_paths]
            )
            deg_paths = [deg_paths[i] for i in sorted_indices]

            deg_idx = random.randint(0, len(deg_paths) - 1)
            degree_dir = deg_paths[deg_idx]
            image_id = np.random.randint(0, 59)

            cam_rays, cam_view_dirs, cam_rays_d, img, seg = self.read_data(
                instance_dir, degree_dir, image_id
            )
            h, w, _ = img.shape
            rays, rays_d, view_dirs, src_img, rgbs, mask = self.get_ray_batch(
                cam_rays, cam_view_dirs, cam_rays_d, img, seg, ray_batch_size=None
            )

            sample = {}
            sample["rays_o"] = rays
            sample["rays_d"] = rays_d
            sample["viewdirs"] = view_dirs
            sample["src_imgs"] = src_img
            sample["target"] = rgbs
            sample["instance_mask"] = mask
            sample["deg"] = np.deg2rad(idx_to_deg["train"][deg_idx]).astype(np.float32)
            sample["img_wh"] = np.array((w, h))
            sample["instance_id"] = val_idx
            sample["articulation_id"] = deg_idx

        else:
            val_idx = random.randint(0, len(self.ids) - 1)
            instance_dir = self.ids[val_idx]
            deg_paths = [
                f.name
                for f in os.scandir(os.path.join(self.root_dir, instance_dir, "train"))
            ]

            sorted_indices = np.argsort(
                [int(filename.split("_")[0]) for filename in deg_paths]
            )
            deg_paths = [deg_paths[i] for i in sorted_indices]

            # deg_idx = random.randint(0, len(deg_paths) - 1)
            deg_idx = idx
            degree_dir = deg_paths[deg_idx]

            # image_id = np.random.randint(0, 59)
            image_id = 0
            cam_rays, cam_view_dirs, cam_rays_d, img, seg = self.get_test_rays(
                instance_dir, image_id
            )
            # cam_rays, cam_view_dirs, cam_rays_d, img, seg = self.read_data(
            #     instance_dir, degree_dir, image_id
            # )
            h, w, _ = img.shape
            rays, rays_d, view_dirs, src_img, rgbs, mask = self.get_ray_batch(
                cam_rays, cam_view_dirs, cam_rays_d, img, seg, ray_batch_size=None
            )

            sample = {}
            sample["rays_o"] = rays
            sample["rays_d"] = rays_d
            sample["viewdirs"] = view_dirs
            sample["src_imgs"] = src_img
            sample["target"] = rgbs
            sample["instance_mask"] = mask
            sample["deg"] = np.deg2rad(idx_to_deg["train"][deg_idx]).astype(np.float32)
            sample["img_wh"] = np.array((w, h))
            sample["instance_id"] = val_idx
            sample["articulation_id"] = deg_idx
        return sample
