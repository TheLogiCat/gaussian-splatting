#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def _render_semantic_sanity(viewpoint_camera, pc: GaussianModel, pipe, means3D, means2D, opacity):
    sem = pc.get_sem
    if sem.shape[0] == 0:
        H = int(viewpoint_camera.image_height)
        W = int(viewpoint_camera.image_width)
        D = int(getattr(pc, "sem_dim", 32))
        return (
            torch.zeros((H, W, D), device=means3D.device, dtype=means3D.dtype),
            torch.zeros((H, W, D), device=means3D.device, dtype=means3D.dtype),
            torch.zeros((H, W), device=means3D.device, dtype=means3D.dtype),
        )

    H = int(viewpoint_camera.image_height)
    W = int(viewpoint_camera.image_width)
    D = sem.shape[1]
    sem_numer = torch.zeros((H, W, D), device=means3D.device, dtype=means3D.dtype)
    w_denom = torch.zeros((H, W), device=means3D.device, dtype=means3D.dtype)
    sem_numer_flat = sem_numer.view(-1, D)
    w_denom_flat = w_denom.view(-1)

    xs = means2D[:, 0]
    ys = means2D[:, 1]
    z = means3D[:, 2].abs() + 1e-6
    w = (torch.sigmoid(opacity).squeeze(-1) / z).clamp_min(0.0)
    valid = (xs >= 0) & (xs <= (W - 1)) & (ys >= 0) & (ys <= (H - 1)) & torch.isfinite(w)
    if valid.any():
        xv = xs[valid]
        yv = ys[valid]
        wv = w[valid]
        sv = sem[valid]

        x0 = torch.floor(xv).long().clamp(0, W - 1)
        y0 = torch.floor(yv).long().clamp(0, H - 1)
        x1 = (x0 + 1).clamp(0, W - 1)
        y1 = (y0 + 1).clamp(0, H - 1)
        wx = (xv - x0.float()).clamp(0.0, 1.0)
        wy = (yv - y0.float()).clamp(0.0, 1.0)

        bilinear = [
            (x0, y0, (1.0 - wx) * (1.0 - wy)),
            (x1, y0, wx * (1.0 - wy)),
            (x0, y1, (1.0 - wx) * wy),
            (x1, y1, wx * wy),
        ]
        for xx, yy, ww in bilinear:
            contrib_w = ww * wv
            pix_id = yy * W + xx
            w_denom_flat.scatter_add_(0, pix_id, contrib_w)
            sem_numer_flat.index_add_(0, pix_id, contrib_w[:, None] * sv)

    sem_map = sem_numer / (w_denom[..., None] + getattr(pipe, "semantic_eps", 1e-6))
    return sem_map, sem_numer, w_denom

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False, render_semantics=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    sem_map = None
    sem_numer = None
    w_denom = None
    sem_features = pc.get_sem if render_semantics else None
    if separate_sh:
        raster_outputs = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp,
            sem_features = sem_features)
    else:
        raster_outputs = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp,
            sem_features = sem_features)

    if len(raster_outputs) == 3:
        rendered_image, radii, depth_image = raster_outputs
    else:
        rendered_image, radii, depth_image, sem_numer_chw, w_denom_hw = raster_outputs
        sem_numer = sem_numer_chw.permute(1, 2, 0).contiguous()
        w_denom = w_denom_hw.contiguous()
        sem_map = sem_numer / (w_denom[..., None] + getattr(pipe, "semantic_eps", 1e-6))
        
    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        "depth" : depth_image
        }

    if render_semantics:
        if sem_map is None:
            sem_map, sem_numer, w_denom = _render_semantic_sanity(viewpoint_camera, pc, pipe, means3D, means2D, opacity)
        out["sem_map"] = sem_map
        out["sem_numer"] = sem_numer
        out["w_denom"] = w_denom
    
    return out
