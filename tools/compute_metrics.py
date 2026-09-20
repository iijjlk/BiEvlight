#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

RELEASE_ROOT = Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in sys.path:
    sys.path.insert(0, str(RELEASE_ROOT))

from egllie.losses.image_loss import SSIM


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def natural_key(path: Path):
    return [int(token) if token.isdigit() else token.lower()
            for token in re.split(r"(\d+)", path.name)]


def list_images(folder: Path):
    files = [p for p in folder.iterdir()
             if p.is_file() and p.suffix.lower() in IMG_EXTS]
    return sorted(files, key=natural_key)


def load_rgb01(path: Path):
    arr = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def psnr(pred, gt, eps=1e-12):
    mse = torch.mean((pred - gt) ** 2)
    return 100.0 if mse.item() < eps else float(-10.0 * torch.log10(mse))


def psnr_star(pred, gt):
    batch = pred.shape[0]
    pred_mean = pred.reshape(batch, -1).mean(dim=1, keepdim=True).clamp_min(1e-12)
    gt_mean = gt.reshape(batch, -1).mean(dim=1, keepdim=True)
    ratio = (gt_mean / pred_mean).view(batch, 1, 1, 1)
    pred_adj = torch.clamp(pred * ratio, 0, 1)
    return psnr(pred_adj, gt)


def evaluate(pred_dir, gt_dir, device="cuda", resize_pred=False):
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)
    if not pred_dir.is_dir():
        raise FileNotFoundError(f"Prediction folder not found: {pred_dir}")
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"GT folder not found: {gt_dir}")

    pred_files = list_images(pred_dir)
    gt_files = list_images(gt_dir)
    count = min(len(pred_files), len(gt_files))
    if count == 0:
        raise RuntimeError("No images found in one or both folders.")
    if len(pred_files) != len(gt_files):
        print(f"[WARN] counts differ: pred={len(pred_files)} gt={len(gt_files)}; evaluating first {count} pairs.")

    device = torch.device(device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    ssim_metric = SSIM(value_range=1.0, window_size=11, size_average=True).to(device)

    psnr_values, psnr_star_values, ssim_values = [], [], []
    for idx, (pred_path, gt_path) in enumerate(zip(pred_files[:count], gt_files[:count]), 1):
        pred_img = Image.open(pred_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")
        if pred_img.size != gt_img.size:
            if not resize_pred:
                raise RuntimeError(
                    f"Shape mismatch at pair #{idx}: {pred_path.name} {pred_img.size} vs {gt_path.name} {gt_img.size}. "
                    "Use --resize_pred to resize predictions to GT size."
                )
            pred_img = pred_img.resize(gt_img.size, Image.BICUBIC)

        pred = torch.from_numpy(np.array(pred_img, dtype=np.float32) / 255.0).permute(2, 0, 1).contiguous()
        gt = torch.from_numpy(np.array(gt_img, dtype=np.float32) / 255.0).permute(2, 0, 1).contiguous()
        pred_b = pred.unsqueeze(0).to(device)
        gt_b = gt.unsqueeze(0).to(device)

        with torch.no_grad():
            psnr_values.append(psnr(pred_b, gt_b))
            psnr_star_values.append(psnr_star(pred_b, gt_b))
            ssim_values.append(float(ssim_metric(pred_b, gt_b).detach().mean().cpu()))

    print("===========================================")
    print(f"Pred dir        : {pred_dir}")
    print(f"GT dir          : {gt_dir}")
    print(f"Pairs evaluated : {count}")
    print(f"Mean PSNR       : {float(np.mean(psnr_values)):.4f} dB")
    print(f"Mean PSNR*      : {float(np.mean(psnr_star_values)):.4f} dB")
    print(f"Mean SSIM       : {float(np.mean(ssim_values)):.6f}")
    print("===========================================")


def main():
    parser = argparse.ArgumentParser("Evaluate PSNR, PSNR*, and SSIM for paired image folders.")
    parser.add_argument("--pred_dir", required=True, help="Folder with predicted/enhanced images.")
    parser.add_argument("--gt_dir", required=True, help="Folder with ground-truth images.")
    parser.add_argument("--device", default="cuda", help="cuda or cpu.")
    parser.add_argument("--resize_pred", action="store_true", help="Resize prediction images to GT size when needed.")
    args = parser.parse_args()
    evaluate(args.pred_dir, args.gt_dir, args.device, args.resize_pred)


if __name__ == "__main__":
    main()
