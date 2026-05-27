#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import json
import ast
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import importlib.util
# ====================== 1. 参数解析 ======================
def parse_args():
    repo_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_py_path", type=str, default=str(repo_dir / "SAM3UNetCNNDifussion.py"))
    parser.add_argument("--checkpoint_path", type=str, default="/workspace/weights/SAM3UNetDiffusion2/best.pt")
    parser.add_argument("--sam3_ckpt", type=str, default="", help="Optional SAM3 backbone checkpoint")
    parser.add_argument("--vae_checkpoint", type=str, default="", help="Optional VAE/latent checkpoint")
    parser.add_argument("--data_root", type=str, default="/workspace/MAF")
    parser.add_argument("--maf_config", type=str, default="/workspace/MAF_Seg/config/maf_fujian.py")
    parser.add_argument("--save_dir", type=str, default="/workspace/weights/SAM3UNetDiffusion2/test_results")
    parser.add_argument("--image_size", type=int, default=336)
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--eval_stage", type=str, default="stage2", choices=["stage1", "stage2"])
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--ddim_t_start", type=int, default=200)
    parser.add_argument("--ddim_steps", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.0)
    return parser.parse_args()


# ====================== 2. 动态加载模型 ======================
def _load_model_cls(py_path):
    p = Path(py_path).resolve()
    sys.path.insert(0, str(p.parent))
    spec = importlib.util.spec_from_file_location(p.stem, str(p))
   
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SAM3UNetCNNDifussion

# ====================== 3. 元数据加载 ======================
def load_maf_fujian_meta(config_path):
    out = {}
    p = Path(config_path)
    if not p.exists(): return out
    src = p.read_text(encoding="utf-8")
    tree = ast.parse(src)
    wanted = {"GT_LABEL_MAPPING", "CLASS_COLORS", "CLASSES"}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in wanted:
                    out[t.id] = ast.literal_eval(node.value)
    return out

# ====================== 4. 测试数据集类 ======================
class MAFSegTestDataset(Dataset):
    def __init__(self, root, split="test", image_size=336, label_mapping=None):
        self.root = Path(root)
        self.image_size = image_size
        self.img_dir = self.root / split / "image"
        self.mask_dir = self.root / split / "mask"
        self.images = sorted(list(self.img_dir.glob("*.png")) + list(self.img_dir.glob("*.jpg")))
        self.mapping = label_mapping or {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}
        
        self.transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img_p = self.images[idx]
        mask_p = self.mask_dir / f"{img_p.stem}.png"
        img = np.array(Image.open(img_p).convert("RGB"))
        
        # 加载并映射 Mask
        m = np.array(Image.open(mask_p).convert("L"))
        mask = np.full_like(m, 255)
        for k, v in self.mapping.items():
            mask[m == k] = v
            
        augmented = self.transform(image=img, mask=mask)
        return augmented["image"], augmented["mask"].long(), img_p.name

# ====================== 5. 主测试逻辑 ======================
@torch.no_grad()
def run_test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "vis").mkdir(exist_ok=True)

    # 1. 加载配置
    meta = load_maf_fujian_meta(args.maf_config)
    gt_map = meta.get("GT_LABEL_MAPPING", None)
    colors = meta.get("CLASS_COLORS", [[0, 0, 0]] * args.num_classes)

    # 2. 初始化模型并加载纯权重
    model_cls = _load_model_cls(args.model_py_path)
    model = model_cls(
        checkpoint_path=args.sam3_ckpt or None,
        img_size=args.image_size,
        num_classes=args.num_classes,
        vae_checkpoint=args.vae_checkpoint or None,
    ).to(device)
    if not Path(args.checkpoint_path).exists():
        print(f"Error: Checkpoint {args.checkpoint_path} not found!")
        return
    
    # 适配纯权重加载
    state_dict = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Successfully loaded weights from {args.checkpoint_path}")

    # 3. 预先构建 DDPM 噪声调度表 (与训练一致)
    T = args.diffusion_steps
    betas = torch.linspace(1e-4, 2e-2, T, device=device)

    # 4. 数据准备
    dataset = MAFSegTestDataset(args.data_root, split="test", image_size=args.image_size, label_mapping=gt_map)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # 5. 指标容器 (coarse + diffusion-refined)
    conf_coarse = torch.zeros((args.num_classes, args.num_classes), dtype=torch.float64, device=device)
    conf_refined = torch.zeros((args.num_classes, args.num_classes), dtype=torch.float64, device=device)

    if args.eval_stage == "stage2":
        print("Starting Inference (coarse + DDPM refinement)...")
    else:
        print("Starting Inference (coarse only)...")
    for imgs, masks, filenames in tqdm(loader):
        imgs, masks = imgs.to(device), masks.to(device)
        
        # ── Coarse prediction (discriminative branch) ──────────────────────
        outputs = model(imgs)
        coarse_out = outputs[0]          # (coarse_logits, boundary_logits)
        pred_coarse = coarse_out.argmax(dim=1)

        # ── DDPM refined prediction ─────────────────────────────────────────
        pred_refined = None
        if args.eval_stage == "stage2":
            t_start = min(args.ddim_t_start, T - 1)
            refined_mask = model.refine_sample(imgs, t_start=t_start, num_steps=args.ddim_steps, T=T, betas=betas, eta=args.eta)
            pred_refined = refined_mask.argmax(dim=1)

        # ── Update confusion matrices ────────────────────────────────────────
        valid = masks != 255
        
        idx_c = masks[valid] * args.num_classes + pred_coarse[valid]
        conf_coarse += torch.bincount(idx_c, minlength=args.num_classes * args.num_classes).reshape(args.num_classes, args.num_classes)

        if pred_refined is not None:
            idx_r = masks[valid] * args.num_classes + pred_refined[valid]
            conf_refined += torch.bincount(idx_r, minlength=args.num_classes * args.num_classes).reshape(args.num_classes, args.num_classes)

        # ── Visualisation: original | GT | coarse | refined ──────────────────
        fn = filenames[0]
        img_np = (imgs[0].cpu().numpy().transpose(1, 2, 0) * np.array([0.229, 0.224, 0.225]) +
                  np.array([0.485, 0.456, 0.406])) * 255
        img_np = img_np.clip(0, 255).astype(np.uint8)

        pred_coarse_np = pred_coarse[0].cpu().numpy()
        pred_refined_np = pred_refined[0].cpu().numpy() if pred_refined is not None else None
        gt_np = masks[0].cpu().numpy()

        vis_gt      = np.zeros_like(img_np)
        vis_coarse  = np.zeros_like(img_np)
        vis_refined = np.zeros_like(img_np)
        for c in range(args.num_classes):
            vis_gt[gt_np == c]             = colors[c]
            vis_coarse[pred_coarse_np == c] = colors[c]
            if pred_refined_np is not None:
                vis_refined[pred_refined_np == c] = colors[c]

        # Concatenate: original | GT | coarse pred | refined pred
        if pred_refined_np is not None:
            combined = np.concatenate([img_np, vis_gt, vis_coarse, vis_refined], axis=1)
        else:
            combined = np.concatenate([img_np, vis_gt, vis_coarse], axis=1)
        Image.fromarray(combined).save(save_dir / "vis" / f"res_{fn}")

    # 6. 指标计算
    def _metrics(conf):
        intersection = conf.diag()
        union = conf.sum(1) + conf.sum(0) - intersection + 1e-6
        iou = intersection / union
        miou = iou.mean().item()
        freq = conf.sum(1) / (conf.sum() + 1e-6)
        fwiou = (freq * iou).sum().item()
        pixel_acc = (intersection.sum() / (conf.sum() + 1e-6)).item()
        return iou, miou, fwiou, pixel_acc

    iou_c, miou_c, fwiou_c, acc_c = _metrics(conf_coarse)
    if args.eval_stage == "stage2":
        iou_r, miou_r, fwiou_r, acc_r = _metrics(conf_refined)

    # 7. 结果汇总
    results = {
        "coarse": {
            "mIoU": round(miou_c, 4),
            "FWIoU": round(fwiou_c, 4),
            "Pixel_Acc": round(acc_c, 4),
        }
    }
    if args.eval_stage == "stage2":
        results["refined"] = {
            "mIoU": round(miou_r, 4),
            "FWIoU": round(fwiou_r, 4),
            "Pixel_Acc": round(acc_r, 4),
        }
    
    print("\n" + "="*40)
    print("Test Metrics Report:")
    print(json.dumps(results, indent=4))
    print("-" * 40)
    if args.eval_stage == "stage2":
        print(f"{'Class':<8} {'Coarse IoU':>12} {'Refined IoU':>12}")
        for i in range(args.num_classes):
            print(f"Class {i:<2}  {iou_c[i].item():>12.4f} {iou_r[i].item():>12.4f}")
    else:
        print(f"{'Class':<8} {'Coarse IoU':>12}")
        for i in range(args.num_classes):
            print(f"Class {i:<2}  {iou_c[i].item():>12.4f}")
    print("=" * 40)

    # 保存结果到 JSON
    with open(save_dir / "test_metrics.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    run_test(parse_args())