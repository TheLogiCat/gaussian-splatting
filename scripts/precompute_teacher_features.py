#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def list_images(images_dir: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    return sorted([p for p in images_dir.iterdir() if p.suffix.lower() in exts])


def load_image(path: Path):
    img = Image.open(path).convert("RGB")
    t = transforms.ToTensor()(img)
    return t


def infer_dinov2(model, image_tensor, device, out_h, out_w):
    x = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        feats = model.forward_features(x)
        if isinstance(feats, dict):
            if "x_norm_patchtokens" in feats:
                tokens = feats["x_norm_patchtokens"]
            elif "x_prenorm" in feats:
                tokens = feats["x_prenorm"][:, 1:, :]
            else:
                raise RuntimeError("Unsupported DINOv2 forward_features output keys")
        else:
            raise RuntimeError("Expected dict output from DINOv2 forward_features")
    B, N, C = tokens.shape
    side = int(N ** 0.5)
    fmap = tokens.reshape(B, side, side, C).permute(0, 3, 1, 2).contiguous()
    fmap = F.interpolate(fmap, size=(out_h, out_w), mode="bilinear", align_corners=False)[0]
    return fmap.cpu()


def infer_random_feature(image_tensor, out_dim):
    C, H, W = image_tensor.shape
    x = image_tensor.mean(dim=0, keepdim=True).repeat(out_dim, 1, 1)
    x = x + 0.01 * torch.randn_like(x)
    return x


def main():
    parser = argparse.ArgumentParser("Precompute teacher features for Gaussian Splatting training images")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory containing training images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write .pt feature maps")
    parser.add_argument("--feature_dim", type=int, default=32, help="Output feature dim used in training")
    parser.add_argument("--teacher", type=str, default="dinov2_vitb14", help="torch.hub model name")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fallback_random", action="store_true", help="Use random mock features if model loading fails")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images = list_images(images_dir)
    if not images:
        raise RuntimeError(f"No images found in {images_dir}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = None
    use_random = False
    try:
        model = torch.hub.load("facebookresearch/dinov2", args.teacher)
        model.eval().to(device)
    except Exception:
        if not args.fallback_random:
            raise
        use_random = True

    for p in images:
        img = load_image(p)
        _, H, W = img.shape
        if use_random:
            feat = infer_random_feature(img, args.feature_dim)
        else:
            feat = infer_dinov2(model, img, device, H, W)
            if feat.shape[0] > args.feature_dim:
                feat = feat[:args.feature_dim]
            elif feat.shape[0] < args.feature_dim:
                pad = torch.zeros((args.feature_dim - feat.shape[0], H, W), dtype=feat.dtype)
                feat = torch.cat([feat, pad], dim=0)
        torch.save({"feature_map": feat, "image_name": p.name, "source": "dinov2" if not use_random else "random"}, output_dir / f"{p.name}.pt")

    manifest = {
        "num_images": len(images),
        "feature_dim": args.feature_dim,
        "teacher": args.teacher if not use_random else "random",
    }
    with open(output_dir / "manifest.json", "w") as f:
        import json
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
