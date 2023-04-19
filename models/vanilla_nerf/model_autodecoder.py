# ------------------------------------------------------------------------------------
# NeRF-Factory
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Modified from NeRF (https://github.com/bmild/nerf)
# Copyright (c) 2020 Google LLC. All Rights Reserved.
# ------------------------------------------------------------------------------------

import os
from random import random
from typing import *
from datasets import dataset_dict
from torch.utils.data import DataLoader
from einops import rearrange, reduce, repeat
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.distributed as dist
from collections import defaultdict
from models.utils import store_image, write_stats, get_obj_rgbs_from_segmap
from models.resnet_encoder import ImgEncoder_MultiHead_Art

import models.vanilla_nerf.helper as helper
from utils.train_helper import *
from models.vanilla_nerf.util import *
from models.interface import LitModel
import wandb
import random
from models.code_library import *

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(0)
random.seed(0)


class JointStateDecoder(nn.Module):
    def __init__(self):
        super(JointStateDecoder, self).__init__()
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 32)
        # self.fc_joint_type = nn.Linear(32, 1)
        self.fc_joint_state = nn.Linear(32, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        # joint_type = torch.sigmoid(self.fc_joint_type(x))
        joint_state = self.fc_joint_state(x)
        # return joint_type, joint_state
        return joint_state


class NeRFMLP(nn.Module):
    def __init__(
        self,
        min_deg_point,
        max_deg_point,
        deg_view,
        netdepth: int = 8,
        netwidth: int = 256,
        netdepth_deformation=4,
        netwidth_deformation: int = 128,
        netdepth_condition: int = 4,
        netwidth_condition: int = 128,
        shape_latent_dim=128,
        appearance_latent_dim=128,
        articulation_latent_dim=32,
        skip_layer: int = 4,
        input_ch: int = 3,
        input_ch_view: int = 3,
        num_rgb_channels: int = 3,
        num_density_channels: int = 1,
        deformation_mlp: bool = True,
        enc_after: bool = True,
        embed_deg: bool = False,
    ):
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super(NeRFMLP, self).__init__()

        self.net_activation = nn.ReLU()
        view_pos_size = (deg_view * 2 + 1) * input_ch_view
        self.enc_after = enc_after
        self.embed_deg = embed_deg
        self.deformation_mlp = deformation_mlp
        if deformation_mlp:
            if self.enc_after:
                pos_size_deformation = (
                    input_ch + shape_latent_dim + articulation_latent_dim
                )
            else:
                pos_size_deformation = (
                    ((max_deg_point - min_deg_point) * 2 + 1) * input_ch
                    + shape_latent_dim
                    + articulation_latent_dim
                )

            init_layer_deformation = nn.Linear(
                pos_size_deformation, netwidth_deformation
            )
            init.xavier_uniform_(init_layer_deformation.weight)
            deformations_linear = [init_layer_deformation]

            for idx in range(netdepth_deformation - 1):
                module_deformation = nn.Linear(
                    netwidth_deformation, netwidth_deformation
                )
                init.xavier_uniform_(module_deformation.weight)
                deformations_linear.append(module_deformation)
            self.deformations_linear = nn.ModuleList(deformations_linear)

            if self.enc_after:
                self.deformation_layer = nn.Linear(netwidth_deformation, 3)
            else:
                self.deformation_layer = nn.Linear(netwidth_deformation, 63)
            init.xavier_uniform_(self.deformation_layer.weight)

            pos_size = (
                (max_deg_point - min_deg_point) * 2 + 1
            ) * input_ch + shape_latent_dim
        else:
            pos_size = (
                ((max_deg_point - min_deg_point) * 2 + 1) * input_ch
                + shape_latent_dim
                + articulation_latent_dim
            )

        init_layer = nn.Linear(pos_size, netwidth)
        init.xavier_uniform_(init_layer.weight)
        pts_linear = [init_layer]

        for idx in range(netdepth - 1):
            if idx % skip_layer == 0 and idx > 0:
                module = nn.Linear(netwidth + pos_size, netwidth)
            else:
                module = nn.Linear(netwidth, netwidth)
            init.xavier_uniform_(module.weight)
            pts_linear.append(module)

        self.pts_linears = nn.ModuleList(pts_linear)

        views_linear = [
            nn.Linear(
                netwidth + view_pos_size + appearance_latent_dim, netwidth_condition
            )
        ]
        for idx in range(netdepth_condition - 1):
            layer = nn.Linear(netwidth_condition, netwidth_condition)
            init.xavier_uniform_(layer.weight)
            views_linear.append(layer)

        self.views_linear = nn.ModuleList(views_linear)

        self.bottleneck_layer = nn.Linear(netwidth, netwidth)
        self.density_layer = nn.Linear(netwidth, num_density_channels)
        self.rgb_layer = nn.Linear(netwidth_condition, num_rgb_channels)

        init.xavier_uniform_(self.bottleneck_layer.weight)
        init.xavier_uniform_(self.density_layer.weight)
        init.xavier_uniform_(self.rgb_layer.weight)

    def forward(self, pos, condition, latents):
        embedding_instance_shape = latents["density"]  # B,128
        embedding_instance_appearance = latents["color"]  # B,128

        if self.embed_deg:
            embedding_instance_articulation = latents["artifuclaiton_deg"]  # B,32
        else:
            embedding_instance_articulation = latents["articulation"]  # B,32

        B, num_samples, feat_dim = pos.shape

        pos = pos.reshape(-1, feat_dim)

        BN = B * num_samples

        embedding_instance_shape = repeat(
            embedding_instance_shape, "n1 c -> (n1 n2) c", n2=BN
        )
        embedding_instance_appearance = repeat(
            embedding_instance_appearance, "n1 c -> (n1 n2) c", n2=BN
        )
        embedding_instance_articulation = repeat(
            embedding_instance_articulation, "n1 c -> (n1 n2) c", n2=BN
        )

        x = torch.cat(
            [pos, embedding_instance_shape, embedding_instance_articulation], -1
        )

        if self.deformation_mlp:
            for idx in range(self.netdepth_deformation):
                x = self.deformations_linear[idx](x)
                x = self.net_activation(x)

            x = self.deformation_layer(x) + pos

            if self.enc_after:
                x = helper.pos_enc(
                    x,
                    self.min_deg_point,
                    self.max_deg_point,
                )
            x = torch.cat([x, embedding_instance_shape], -1)

        inputs = x
        for idx in range(self.netdepth):
            x = self.pts_linears[idx](x)
            x = self.net_activation(x)
            if idx % self.skip_layer == 0 and idx > 0:
                x = torch.cat([x, inputs], dim=-1)

        raw_density = self.density_layer(x).reshape(
            -1, num_samples, self.num_density_channels
        )

        bottleneck = self.bottleneck_layer(x)
        condition_tile = torch.tile(condition[:, None, :], (1, num_samples, 1)).reshape(
            -1, condition.shape[-1]
        )
        x = torch.cat(
            [bottleneck, condition_tile, embedding_instance_appearance], dim=-1
        )
        for idx in range(self.netdepth_condition):
            x = self.views_linear[idx](x)
            x = self.net_activation(x)

        raw_rgb = self.rgb_layer(x).reshape(-1, num_samples, self.num_rgb_channels)

        return raw_rgb, raw_density


class NeRF_AE_Art(nn.Module):
    def __init__(
        self,
        num_levels: int = 2,
        min_deg_point: int = 0,
        max_deg_point: int = 10,
        deg_view: int = 4,
        num_coarse_samples: int = 128,
        num_fine_samples: int = 256,
        use_viewdirs: bool = True,
        noise_std: float = 0.0,
        lindisp: bool = False,
        rgb_padding: float = 0.001,
        density_bias: float = -1.0,
        enc_after=True,
        embed_deg=False,
    ):
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super(NeRF_AE_Art, self).__init__()

        self.rgb_activation = nn.Sigmoid()
        # self.sigma_activation = nn.ReLU()
        self.sigma_activation = nn.Softplus()
        self.coarse_mlp = NeRFMLP(min_deg_point, max_deg_point, deg_view)
        self.fine_mlp = NeRFMLP(min_deg_point, max_deg_point, deg_view)
        # self.joint_state_decoder = JointStateDecoder()
        # self.encoder = ImgEncoder_MultiHead_Art()
        # if embed_deg:
        #     self.deg_embedding = nn.Embedding(91, 32)  # 91 because 0 to 90 inclusive

    def encode(self, images):
        return self.encoder(images)

    def forward(self, rays, randomized, white_bkgd, near, far, latents, train=True):
        ret = []
        for i_level in range(self.num_levels):
            if i_level == 0:
                t_vals, samples = helper.sample_along_rays(
                    rays_o=rays["rays_o"],
                    rays_d=rays["rays_d"],
                    num_samples=self.num_coarse_samples,
                    near=near,
                    far=far,
                    randomized=randomized,
                    lindisp=self.lindisp,
                )
                mlp = self.coarse_mlp

            else:
                t_mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
                t_vals, samples = helper.sample_pdf(
                    bins=t_mids,
                    weights=weights[..., 1:-1],
                    origins=rays["rays_o"],
                    directions=rays["rays_d"],
                    t_vals=t_vals,
                    num_samples=self.num_fine_samples,
                    randomized=randomized,
                )
                mlp = self.fine_mlp

            if self.enc_after:
                samples_enc = samples
            else:
                samples_enc = helper.pos_enc(
                    samples,
                    self.min_deg_point,
                    self.max_deg_point,
                )

            viewdirs_enc = helper.pos_enc(rays["viewdirs"], 0, self.deg_view)
            raw_rgb, raw_sigma = mlp(samples_enc, viewdirs_enc, latents)

            if self.noise_std > 0 and randomized:
                raw_sigma = raw_sigma + torch.rand_like(raw_sigma) * self.noise_std

            rgb = self.rgb_activation(raw_rgb)
            rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding
            sigma = self.sigma_activation(raw_sigma + self.density_bias)
            # sigma = self.sigma_activation(raw_sigma)

            comp_rgb, acc, weights, depth = helper.volumetric_rendering(
                rgb,
                sigma,
                t_vals,
                rays["rays_d"],
                white_bkgd=white_bkgd,
            )

            # ret.append((comp_rgb, acc))
            ret.append((comp_rgb, acc, depth))

        return ret


class LitNeRF_AutoDecoder(LitModel):
    def __init__(
        self,
        hparams,
        lr_init: float = 5.0e-4,
        lr_final: float = 5.0e-6,
        lr_delay_steps: int = 2500,
        lr_delay_mult: float = 0.01,
        randomized: bool = True,
    ):
        for name, value in vars().items():
            if name not in ["self", "__class__", "hparams"]:
                print(name, value)
                setattr(self, name, value)
        self.hparams.update(vars(hparams))
        super(LitNeRF_AutoDecoder, self).__init__()
        self.model = NeRF_AE_Art()
        self.code_library = CodeLibraryArticulated(self.hparams)

    def setup(self, stage: Optional[str] = None) -> None:
        dataset = dataset_dict[self.hparams.dataset_name]

        kwargs_train = {
            "root_dir": self.hparams.root_dir,
            "img_wh": tuple(self.hparams.img_wh),
            "white_back": self.hparams.white_back,
            "model_type": "vailla_nerf",
        }
        kwargs_val = {
            "root_dir": self.hparams.root_dir,
            "img_wh": tuple(self.hparams.img_wh),
            "white_back": self.hparams.white_back,
            "model_type": "vanilla_nerf",
        }

        if self.hparams.run_eval:
            kwargs_test = {
                "root_dir": self.hparams.root_dir,
                "img_wh": tuple(self.hparams.img_wh),
                "white_back": self.hparams.white_back,
                "model_type": "vanilla_nerf",
                "eval_inference": self.hparams.render_name,
            }
            self.test_dataset = dataset(split="test_val", **kwargs_test)
            self.near = self.test_dataset.near
            self.far = self.test_dataset.far
            self.white_bkgd = self.test_dataset.white_back

        else:
            self.train_dataset = dataset(split="train", **kwargs_train)
            self.val_dataset = dataset(split="val", **kwargs_val)
            self.near = self.train_dataset.near
            self.far = self.train_dataset.far
            self.white_bkgd = self.train_dataset.white_back

    def training_step(self, batch, batch_idx):
        for k, v in batch.items():
            if k == "deg" or k == "instance_id" or k == "articulation_id":
                continue
            batch[k] = v.squeeze(0)

        # latents = self.model.encode(batch["src_imgs"].unsqueeze(0))

        latents = self.code_library(batch)
        # regress joint state values here:
        # embedding_instance_articulation = latents["articulation"]  # B,32
        # # for batch size of 1, we need to unsqueeze the batch dimension
        # pred_state = self.model.joint_state_decoder(
        #     embedding_instance_articulation.squeeze(0)
        # )
        # state_deg = torch.round(torch.rad2deg(batch["deg"])).long()
        # embedding_deg_articulation = self.model.deg_embedding(state_deg)
        # latents["artifuclaiton_deg"] = embedding_deg_articulation

        white_bkgd = self.white_bkgd
        # white_bkgd = False
        rendered_results = self.model(
            batch, self.randomized, white_bkgd, self.near, self.far, latents
        )
        rgb_coarse = rendered_results[0][0]
        rgb_fine = rendered_results[1][0]
        target = batch["target"]

        mask = batch["instance_mask"].view(-1, 1).repeat(1, 3)

        loss0 = helper.img2mse(rgb_coarse[mask], target[mask])
        loss1 = helper.img2mse(rgb_fine[mask], target[mask])

        # loss0 = helper.img2mse(rgb_coarse, target)
        # loss1 = helper.img2mse(rgb_fine, target)
        loss = loss1 + loss0

        # crtierion_state = nn.MSELoss()
        # # loss_state = crtierion_state(rendered_results[1][2].float(), batch["deg"])
        # loss_state = crtierion_state(pred_state.float(), batch["deg"])
        # self.log("train/loss_state", loss_state, on_step=True)
        # loss += loss_state

        # opacity loss

        opacity_loss = self.opacity_loss_CE(
            rendered_results, batch["instance_mask"].view(-1)
        )

        # opacity_loss = self.opacity_loss(
        #         rendered_results, batch["instance_mask"].view(-1)
        #     )
        # opacity_loss = self.opacity_loss_autorf(
        #         rendered_results, batch["instance_mask"].view(-1)
        #     )
        self.log("train/opacity_loss", opacity_loss, on_step=True)
        loss += opacity_loss

        shape_code = latents["density"]
        appearance_code = latents["color"]
        articulation_code = latents["articulation"]

        reg_loss = (
            torch.mean(torch.norm(shape_code, dim=0))
            + torch.mean(torch.norm(appearance_code, dim=0))
            + torch.mean(torch.norm(articulation_code, dim=0))
        )
        reg_loss = 1e-4 * reg_loss
        loss += reg_loss

        psnr0 = helper.mse2psnr(loss0)
        psnr1 = helper.mse2psnr(loss1)

        self.log("train/psnr1", psnr1, on_step=True, prog_bar=True, logger=True)
        self.log("train/psnr0", psnr0, on_step=True, prog_bar=True, logger=True)
        self.log("train/loss", loss, on_step=True)
        self.log("train/loss/reg", reg_loss, on_step=True)
        self.log("train/lr", helper.get_learning_rate(self.optimizers()))

        return loss

    def render_rays(self, batch, latents):
        B = batch["rays_o"].shape[0]
        ret = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            batch_chunk = dict()
            for k, v in batch.items():
                if k == "img_wh" or k == "src_imgs":
                    continue
                if k == "radii":
                    batch_chunk[k] = v[:, i : i + self.hparams.chunk]
                else:
                    batch_chunk[k] = v[i : i + self.hparams.chunk]

            white_bkgd = self.white_bkgd
            # white_bkgd = True
            rendered_results_chunk = self.model(
                batch_chunk, False, white_bkgd, self.near, self.far, latents
            )
            # here 1 denotes fine
            ret["comp_rgb"] += [rendered_results_chunk[1][0]]
            ret["acc"] += [rendered_results_chunk[1][1]]
            ret["depth"] += [rendered_results_chunk[1][2]]
            # for k, v in rendered_results_chunk[1].items():
            #     ret[k] += [v]
        for k, v in ret.items():
            ret[k] = torch.cat(v, 0)

        psnr_ = self.psnr_legacy(ret["comp_rgb"], batch["target"]).mean()
        self.log("val/psnr", psnr_.item(), on_step=True, prog_bar=True, logger=True)

        mask = batch["instance_mask"].view(-1, 1).repeat(1, 3)
        psnr_obj = self.psnr_legacy(ret["comp_rgb"][mask], batch["target"][mask]).mean()
        self.log("val/psnr_obj", psnr_obj.item(), on_epoch=True, sync_dist=True)
        return ret

    def render_rays_test(self, batch, batch_idx):
        B = batch["rays_o"].shape[0]
        ret = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            batch_chunk = dict()
            for k, v in batch.items():
                if k == "radii":
                    batch_chunk[k] = v[:, i : i + self.hparams.chunk]
                else:
                    batch_chunk[k] = v[i : i + self.hparams.chunk]
            rendered_results_chunk = self.model(
                batch_chunk, False, self.white_bkgd, self.near, self.far
            )
            # here 1 denotes fine
            ret["comp_rgb"] += [rendered_results_chunk[1][0]]
            ret["acc"] += [rendered_results_chunk[1][1]]
            # for k, v in rendered_results_chunk[1].items():
            #     ret[k] += [v]

        for k, v in ret.items():
            ret[k] = torch.cat(v, 0)
        test_output = {}
        test_output["target"] = batch["target"]
        test_output["instance_mask"] = batch["instance_mask"]
        test_output["rgb"] = ret["comp_rgb"]
        return test_output

    def on_validation_start(self):
        self.random_batch = np.random.randint(1, size=1)[0]

    def validation_step(self, batch, batch_idx):
        for k, v in batch.items():
            if k == "deg" or k == "instance_id" or k == "articulation_id":
                continue
            batch[k] = v.squeeze(0)

        for k, v in batch.items():
            print(k, v.shape)

        # latents = self.model.encode(batch["src_imgs"].unsqueeze(0))

        latents = self.code_library(batch)

        # # regress joint state values here:
        # embedding_instance_articulation = latents["articulation"]  # B,32
        # # for batch size of 1, we need to unsqueeze the batch dimension
        # pred_state = self.model.joint_state_decoder(
        #     embedding_instance_articulation.squeeze(0)
        # )

        # # input gt state to artculation since it throws an error in the start of training where errors are large
        # state_deg = torch.round(torch.rad2deg(batch["deg"])).long()
        # # state_deg = torch.round(torch.rad2deg(pred_state)).long()
        # embedding_deg_articulation = self.model.deg_embedding(state_deg)
        # latents["artifuclaiton_deg"] = embedding_deg_articulation

        W, H = batch["img_wh"]
        # W,H = batch["img_wh"][0], batch["img_wh"][1]
        ret = self.render_rays(batch, latents)
        rank = dist.get_rank()
        # rank =0
        if rank == 0:
            if batch_idx == self.random_batch:
                grid_img = visualize_val_rgb_opa_depth((W, H), batch, ret)
                self.logger.experiment.log({"val/GT_pred rgb": wandb.Image(grid_img)})

        return ret

    def test_step(self, batch, batch_idx):
        for k, v in batch.items():
            if k == "deg":
                continue
            batch[k] = v.squeeze()
            if k == "radii":
                batch[k] = v.unsqueeze(-1)
            if k == "near_obj" or k == "far_obj":
                batch[k] = batch[k].unsqueeze(-1)
        return self.render_rays_test(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(), lr=self.lr_init, betas=(0.9, 0.999)
        )

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu,
        using_native_amp,
        using_lbfgs,
    ):
        step = self.trainer.global_step
        max_steps = self.hparams.run_max_steps

        if self.lr_delay_steps > 0:
            delay_rate = self.lr_delay_mult + (1 - self.lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / self.lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0

        t = np.clip(step / max_steps, 0, 1)
        scaled_lr = np.exp(np.log(self.lr_init) * (1 - t) + np.log(self.lr_final) * t)
        new_lr = delay_rate * scaled_lr

        for pg in optimizer.param_groups:
            pg["lr"] = new_lr

        optimizer.step(closure=optimizer_closure)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=16,
            batch_size=1,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=2,
            batch_size=1,  # validate one image (H*W rays) at a time
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            num_workers=1,
            batch_size=1,
            pin_memory=True,
        )

    def test_epoch_end(self, outputs):
        all_image_sizes = self.test_dataset.image_sizes

        rgbs = self.alter_gather_cat(outputs, "rgb", all_image_sizes)
        instance_masks = self.alter_gather_cat(
            outputs, "instance_mask", all_image_sizes
        )
        targets = self.alter_gather_cat(outputs, "target", all_image_sizes)

        psnr = self.psnr(rgbs, targets, None, None, None)
        ssim = self.ssim(rgbs, targets, None, None, None)
        lpips = self.lpips(rgbs, targets, None, None, None)

        all_obj_rgbs, all_target_rgbs = get_obj_rgbs_from_segmap(
            instance_masks, rgbs, targets
        )

        psnr_obj = self.psnr(all_obj_rgbs, all_target_rgbs, None, None, None)
        print("psnr obj", psnr_obj)

        self.log("test/psnr", psnr["test"], on_epoch=True)
        self.log("test/ssim", ssim["test"], on_epoch=True)
        self.log("test/lpips", lpips["test"], on_epoch=True)
        print("psnr, ssim, lpips", psnr, ssim, lpips)
        self.log("test/psnr_obj", psnr_obj["test"], on_epoch=True)

        if self.trainer.is_global_zero:
            image_dir = os.path.join(
                "ckpts", self.hparams.exp_name, self.hparams.render_name
            )
            os.makedirs(image_dir, exist_ok=True)
            store_image(image_dir, rgbs, "image")

            result_path = os.path.join("ckpts", self.hparams.exp_name, "results.json")
            write_stats(result_path, psnr, ssim, lpips, psnr_obj)

        return psnr, ssim, lpips

    def opacity_loss(self, rendered_results, instance_mask):
        criterion = nn.MSELoss(reduction="none")
        loss = (
            criterion(
                torch.clamp(rendered_results[0][1], 0, 1),
                instance_mask.float(),
            )
        ).mean()
        loss += (
            criterion(
                torch.clamp(rendered_results[1][1], 0, 1),
                instance_mask.float(),
            )
        ).mean()
        return loss

    def opacity_loss_CE(self, rendered_results, instance_mask):
        opacity_lambda = 0.05
        criterion = nn.BCEWithLogitsLoss()
        loss = (
            criterion(
                rendered_results[0][1].float(),
                instance_mask.float(),
            )
        ).mean()
        loss += (
            criterion(
                rendered_results[1][1].float(),
                instance_mask.float(),
            )
        ).mean()
        #
        return loss * opacity_lambda
        # return loss

    def opacity_loss_autorf(self, rendered_results, instance_mask):
        pred_op_course = rendered_results[0][1]
        pred_op_fine = rendered_results[1][1]
        loss = 0
        bg_mask = instance_mask == 0
        bg_ratio = bg_mask.sum() / bg_mask.numel()
        if bg_mask.sum() > 0:
            bg_loss_course = (
                pred_op_course[bg_mask].mean() * bg_ratio
            )  # l2 error to bg -> depth =0
            bg_loss_fine = (
                pred_op_fine[bg_mask].mean() * bg_ratio
            )  # l2 error to bg -> depth =0
            loss += bg_loss_course
            loss += bg_loss_fine

        # ensure fg is occupied
        fg_mask = instance_mask == 1
        fg_ratio = fg_mask.sum() / fg_mask.numel()
        if fg_mask.sum() > 0:
            fg_loss_course = (
                fg_ratio * (1 - pred_op_course[fg_mask]).mean()
            )  # l2 error on sum of fg weights => 1
            fg_loss_fine = (
                fg_ratio * (1 - pred_op_course[fg_mask]).mean()
            )  # l2 error on sum of fg weights => 1
            loss += fg_loss_course
            loss += fg_loss_fine
        return loss

    def mse_error(self, predictions, targets):
        squared_errors = (predictions - targets) ** 2
        mean_squared_error = torch.mean(squared_errors)
        return mean_squared_error
