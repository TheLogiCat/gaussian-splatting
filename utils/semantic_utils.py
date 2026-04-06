import os
import json
import torch
import torch.nn.functional as F
from PIL import Image


def _to_chw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got shape {tuple(x.shape)}")
    if x.shape[0] <= 64:
        return x
    return x.permute(2, 0, 1).contiguous()


def load_teacher_feature_map(teacher_dir: str, image_name: str, device: torch.device, sem_dim: int, height: int, width: int) -> torch.Tensor:
    if teacher_dir is None or teacher_dir == "":
        raise FileNotFoundError("semantic_teacher_dir is empty")
    base = os.path.join(teacher_dir, f"{image_name}.pt")
    if not os.path.exists(base):
        raise FileNotFoundError(f"Teacher feature file not found: {base}")
    feat = torch.load(base, map_location="cpu")
    if isinstance(feat, dict):
        if "feature_map" in feat:
            feat = feat["feature_map"]
        elif "feat" in feat:
            feat = feat["feat"]
    feat = feat.float()
    feat = _to_chw(feat)
    if feat.shape[0] >= sem_dim:
        feat = feat[:sem_dim]
    else:
        pad = torch.zeros((sem_dim - feat.shape[0], feat.shape[1], feat.shape[2]), dtype=feat.dtype)
        feat = torch.cat([feat, pad], dim=0)
    feat = F.interpolate(feat[None], size=(height, width), mode="bilinear", align_corners=False)[0]
    return feat.permute(1, 2, 0).contiguous().to(device)


def semantic_cosine_loss(sem_map: torch.Tensor, teacher_map: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
    sem_map = F.normalize(sem_map, dim=-1)
    teacher_map = F.normalize(teacher_map, dim=-1)
    cos = (sem_map * teacher_map).sum(dim=-1)
    loss_map = 1.0 - cos
    if valid_mask is not None:
        if valid_mask.any():
            return loss_map[valid_mask].mean()
        return loss_map.mean() * 0.0
    return loss_map.mean()


def pca_to_rgb(feature_map: torch.Tensor) -> torch.Tensor:
    # feature_map: [H, W, D]
    H, W, D = feature_map.shape
    x = feature_map.reshape(-1, D)
    x = x - x.mean(dim=0, keepdim=True)
    cov = x.t().matmul(x) / max(x.shape[0] - 1, 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    top = eigvecs[:, -3:]
    proj = x.matmul(top).reshape(H, W, 3)
    mins = proj.amin(dim=(0, 1), keepdim=True)
    maxs = proj.amax(dim=(0, 1), keepdim=True)
    proj = (proj - mins) / (maxs - mins + 1e-6)
    return proj.clamp(0.0, 1.0)


def write_rgb_tensor(path: str, rgb_hwc: torch.Tensor):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = (rgb_hwc.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype("uint8")
    Image.fromarray(arr).save(path)


def write_gray_tensor(path: str, gray_hw: torch.Tensor):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    x = gray_hw.detach().cpu()
    x = (x - x.min()) / (x.max() - x.min() + 1e-6)
    arr = (x.numpy() * 255.0).astype("uint8")
    Image.fromarray(arr).save(path)


def save_semantic_debug(out_dir: str, iteration: int, image_name: str, sem_map: torch.Tensor, w_denom: torch.Tensor):
    sem_rgb = pca_to_rgb(sem_map)
    sem_path = os.path.join(out_dir, f"{iteration:06d}_{image_name}_sem_pca.png")
    den_path = os.path.join(out_dir, f"{iteration:06d}_{image_name}_w_denom.png")
    write_rgb_tensor(sem_path, sem_rgb)
    write_gray_tensor(den_path, w_denom)


def save_teacher_manifest(path: str, payload: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
