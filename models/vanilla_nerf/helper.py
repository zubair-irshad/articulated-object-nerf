# ------------------------------------------------------------------------------------
# NeRF-Factory
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Modified from NeRF (https://github.com/bmild/nerf)
# Copyright (c) 2020 Google LLC. All Rights Reserved.
# ------------------------------------------------------------------------------------


import numpy as np
import torch
import torch.nn.functional as F


def img2mse(x, y):
    return torch.mean((x - y) ** 2)


def mse2psnr(x):
    return -10.0 * torch.log(x) / np.log(10)


def cast_rays(t_vals, origins, directions):
    return origins[..., None, :] + t_vals[..., None] * directions[..., None, :]


def get_ray_limits(rays_o: torch.Tensor, rays_d: torch.Tensor, box_side_length=2):
    batch_near, batch_far = get_ray_limits_box(
        rays_o, rays_d, box_side_length=box_side_length
    )
    is_ray_valid = batch_far > batch_near
    if torch.any(is_ray_valid).item():
        batch_near[~is_ray_valid] = batch_near[is_ray_valid].min()
        batch_far[~is_ray_valid] = batch_far[is_ray_valid].max()
    batch_near[batch_near < 0] = 0
    batch_far[batch_far < 0] = 0
    return batch_near, batch_far


def get_ray_limits_box(rays_o: torch.Tensor, rays_d: torch.Tensor, box_side_length):
    """
    Author: Petr Kellnhofer
    Intersects rays with the [-1, 1] NDC volume.
    Returns min and max distance of entry.
    Returns -1 for no intersection.
    https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
    """
    o_shape = rays_o.shape
    rays_o = rays_o.detach().reshape(-1, 3)
    rays_d = rays_d.detach().reshape(-1, 3)
    bb_min = [
        -1 * (box_side_length / 2),
        -1 * (box_side_length / 2),
        -1 * (box_side_length / 2),
    ]
    bb_max = [
        1 * (box_side_length / 2),
        1 * (box_side_length / 2),
        1 * (box_side_length / 2),
    ]
    bounds = torch.tensor([bb_min, bb_max], dtype=rays_o.dtype, device=rays_o.device)
    is_valid = torch.ones(rays_o.shape[:-1], dtype=bool, device=rays_o.device)
    # Precompute inverse for stability.
    invdir = 1 / rays_d
    sign = (invdir < 0).long()
    # Intersect with YZ plane.
    tmin = (bounds.index_select(0, sign[..., 0])[..., 0] - rays_o[..., 0]) * invdir[
        ..., 0
    ]
    tmax = (bounds.index_select(0, 1 - sign[..., 0])[..., 0] - rays_o[..., 0]) * invdir[
        ..., 0
    ]
    # Intersect with XZ plane.
    tymin = (bounds.index_select(0, sign[..., 1])[..., 1] - rays_o[..., 1]) * invdir[
        ..., 1
    ]
    tymax = (
        bounds.index_select(0, 1 - sign[..., 1])[..., 1] - rays_o[..., 1]
    ) * invdir[..., 1]
    # Resolve parallel rays.
    is_valid[torch.logical_or(tmin > tymax, tymin > tmax)] = False
    # Use the shortest intersection.
    tmin = torch.max(tmin, tymin)
    tmax = torch.min(tmax, tymax)
    # Intersect with XY plane.
    tzmin = (bounds.index_select(0, sign[..., 2])[..., 2] - rays_o[..., 2]) * invdir[
        ..., 2
    ]
    tzmax = (
        bounds.index_select(0, 1 - sign[..., 2])[..., 2] - rays_o[..., 2]
    ) * invdir[..., 2]
    # Resolve parallel rays.
    is_valid[torch.logical_or(tmin > tzmax, tzmin > tmax)] = False
    # Use the shortest intersection.
    tmin = torch.max(tmin, tzmin)
    tmax = torch.min(tmax, tzmax)
    # Mark invalid.
    tmin[torch.logical_not(is_valid)] = -1
    tmax[torch.logical_not(is_valid)] = -2
    return tmin.reshape(*o_shape[:-1], 1), tmax.reshape(*o_shape[:-1], 1)



def sample_along_rays(
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


def pos_enc(x, min_deg, max_deg):
    scales = torch.tensor([2**i for i in range(min_deg, max_deg)]).type_as(x)
    xb = torch.reshape((x[..., None, :] * scales[:, None]), list(x.shape[:-1]) + [-1])
    four_feat = torch.sin(torch.cat([xb, xb + 0.5 * np.pi], dim=-1))
    return torch.cat([x] + [four_feat], dim=-1)


def get_parameters(models):
    """Get all model parameters recursively."""
    parameters = []
    if isinstance(models, list):
        for model in models:
            parameters += get_parameters(model)
    elif isinstance(models, dict):
        for model in models.values():
            parameters += get_parameters(model)
    else:  # models is actually a single pytorch model
        parameters += list(models.parameters())
    return parameters


def volumetric_rendering(rgb, density, t_vals, dirs, white_bkgd, nocs=None):
    eps = 1e-10

    dists = torch.cat(
        [
            t_vals[..., 1:] - t_vals[..., :-1],
            torch.ones(t_vals[..., :1].shape, device=t_vals.device) * 1e10,
        ],
        dim=-1,
    )
    dists = dists * torch.norm(dirs[..., None, :], dim=-1)
    alpha = 1.0 - torch.exp(-density[..., 0] * dists)
    accum_prod = torch.cat(
        [
            torch.ones_like(alpha[..., :1]),
            torch.cumprod(1.0 - alpha[..., :-1] + eps, dim=-1),
        ],
        dim=-1,
    )

    weights = alpha * accum_prod

    comp_rgb = (weights[..., None] * rgb).sum(dim=-2)
    depth = (weights * t_vals).sum(dim=-1)

    depth = torch.nan_to_num(depth, float("inf"))
    depth = torch.clamp(depth, torch.min(depth), torch.max(depth))

    acc = weights.sum(dim=-1)
    inv_eps = 1 / eps

    if white_bkgd:
        comp_rgb = comp_rgb + (1.0 - acc[..., None])

    if nocs is not None:
        comp_nocs = (weights[..., None] * nocs).sum(dim=-2)
        return comp_rgb, acc, weights, comp_nocs
    else:
        return comp_rgb, acc, weights, depth


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def sorted_piecewise_constant_pdf(
    bins, weights, num_samples, randomized, float_min_eps=2**-32
):
    eps = 1e-5
    weight_sum = weights.sum(dim=-1, keepdims=True)
    padding = torch.fmax(torch.zeros_like(weight_sum), eps - weight_sum)
    weights = weights + padding / weights.shape[-1]
    weight_sum = weight_sum + padding

    pdf = weights / weight_sum
    cdf = torch.fmin(
        torch.ones_like(pdf[..., :-1]), torch.cumsum(pdf[..., :-1], dim=-1)
    )
    cdf = torch.cat(
        [
            torch.zeros(list(cdf.shape[:-1]) + [1], device=weights.device),
            cdf,
            torch.ones(list(cdf.shape[:-1]) + [1], device=weights.device),
        ],
        dim=-1,
    )

    s = 1 / num_samples
    if randomized:
        u = torch.rand(list(cdf.shape[:-1]) + [num_samples], device=cdf.device)
    else:
        u = torch.linspace(0.0, 1.0 - float_min_eps, num_samples, device=cdf.device)
        u = torch.broadcast_to(u, list(cdf.shape[:-1]) + [num_samples])

    mask = u[..., None, :] >= cdf[..., :, None]

    bin0 = (mask * bins[..., None] + ~mask * bins[..., :1, None]).max(dim=-2)[0]
    bin1 = (~mask * bins[..., None] + mask * bins[..., -1:, None]).min(dim=-2)[0]
    # Debug Here
    cdf0 = (mask * cdf[..., None] + ~mask * cdf[..., :1, None]).max(dim=-2)[0]
    cdf1 = (~mask * cdf[..., None] + mask * cdf[..., -1:, None]).min(dim=-2)[0]

    t = torch.clip(torch.nan_to_num((u - cdf0) / (cdf1 - cdf0), 0), 0, 1)
    samples = bin0 + t * (bin1 - bin0)

    return samples


def sample_pdf(bins, weights, origins, directions, t_vals, num_samples, randomized):
    t_samples = sorted_piecewise_constant_pdf(
        bins, weights, num_samples, randomized
    ).detach()
    t_vals = torch.sort(torch.cat([t_vals, t_samples], dim=-1), dim=-1).values
    coords = cast_rays(t_vals, origins, directions)
    return t_vals, coords
