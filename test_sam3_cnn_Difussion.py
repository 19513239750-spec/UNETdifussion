#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import json
import ast
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
# ====================== 1. 脚本硬编码配置 ======================
# 模型文件路径
MODEL_PY_PATH = "/workspace/sam3_Unet/sam3_unet/sam3/SAM3UNetCNNDifussion.py"
# 权重文件路径 (之前训练保存的 best.pt 或 final.pt)
CHECKPOINT_PATH = "/workspace/weights/SAM3UNetDiffusion2/best.pt"
# 数据集根目录
DATA_ROOT = "/workspace/MAF"
# 类别映射配置文件
MAF_CONFIG_PATH = "/workspace/MAF_Seg/config/maf_fujian.py"
# 结果保存目录
SAVE_DIR = "/workspace/weights/SAM3UNetDiffusion2/test_results"

IMAGE_SIZE = 336
NUM_CLASSES = 7

# DDPM 反向采样步数 (减小可加速推理，增大可提升精度)
DDPM_NUM_STEPS = 20

# ====================== 2. 动态加载模型 ======================
def _load_model_cls(py_path):
    p = Path(py_path).resolve()
    sys.path.insert(0, str(p.parent))
    spec = importlib.util.spec_from_file_location(p.stem, str(p))
   
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SAM3UNetCNNDifussion

SAM3UNetCNNDifussion = _load_model_cls(MODEL_PY_PATH)

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
def run_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "vis").mkdir(exist_ok=True)

    # 1. 加载配置
    meta = load_maf_fujian_meta(MAF_CONFIG_PATH)
    gt_map = meta.get("GT_LABEL_MAPPING", None)
    colors = meta.get("CLASS_COLORS", [[0,0,0]]*NUM_CLASSES)

    # 2. 初始化模型并加载纯权重
    model = SAM3UNetCNNDifussion(img_size=IMAGE_SIZE, num_classes=NUM_CLASSES).to(device)
    if not Path(CHECKPOINT_PATH).exists():
        print(f"Error: Checkpoint {CHECKPOINT_PATH} not found!")
        return
    
    # 适配纯权重加载
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Successfully loaded weights from {CHECKPOINT_PATH}")

    # 3. 预先构建 DDPM 噪声调度表 (与训练一致)
    T = 1000
    betas = torch.linspace(1e-4, 2e-2, T, device=device)

    # 4. 数据准备
    dataset = MAFSegTestDataset(DATA_ROOT, split="test", image_size=IMAGE_SIZE, label_mapping=gt_map)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # 5. 指标容器 (coarse + diffusion-refined)
    conf_coarse  = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.float64, device=device)
    conf_refined = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.float64, device=device)

    print("Starting Inference (coarse + DDPM refinement)...")
    for imgs, masks, filenames in tqdm(loader):
        imgs, masks = imgs.to(device), masks.to(device)
        
        # ── Coarse prediction (discriminative branch) ──────────────────────
        outputs = model(imgs)
        coarse_out = outputs[0]          # (coarse_logits, boundary_logits)
        pred_coarse = coarse_out.argmax(dim=1)

        # ── DDPM refined prediction ─────────────────────────────────────────
        refined_mask = model.refine_sample(imgs, t_start=200, num_steps=5, T=T, betas=betas)
        pred_refined = refined_mask.argmax(dim=1)

        # ── Update confusion matrices ────────────────────────────────────────
        valid = masks != 255

        idx_c = masks[valid] * NUM_CLASSES + pred_coarse[valid]
        conf_coarse += torch.bincount(idx_c, minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)

        idx_r = masks[valid] * NUM_CLASSES + pred_refined[valid]
        conf_refined += torch.bincount(idx_r, minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)

        # ── Visualisation: original | GT | coarse | refined ──────────────────
        fn = filenames[0]
        img_np = (imgs[0].cpu().numpy().transpose(1, 2, 0) * np.array([0.229, 0.224, 0.225]) +
                  np.array([0.485, 0.456, 0.406])) * 255
        img_np = img_np.clip(0, 255).astype(np.uint8)

        pred_coarse_np  = pred_coarse[0].cpu().numpy()
        pred_refined_np = pred_refined[0].cpu().numpy()
        gt_np = masks[0].cpu().numpy()

        vis_gt      = np.zeros_like(img_np)
        vis_coarse  = np.zeros_like(img_np)
        vis_refined = np.zeros_like(img_np)
        for c in range(NUM_CLASSES):
            vis_gt[gt_np == c]             = colors[c]
            vis_coarse[pred_coarse_np == c]  = colors[c]
            vis_refined[pred_refined_np == c] = colors[c]

        # Concatenate: original | GT | coarse pred | refined pred
        combined = np.concatenate([img_np, vis_gt, vis_coarse, vis_refined], axis=1)
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

    iou_c,  miou_c,  fwiou_c,  acc_c  = _metrics(conf_coarse)
    iou_r,  miou_r,  fwiou_r,  acc_r  = _metrics(conf_refined)

    # 7. 结果汇总
    results = {
        "coarse": {
            "mIoU": round(miou_c, 4),
            "FWIoU": round(fwiou_c, 4),
            "Pixel_Acc": round(acc_c, 4)
        },
        "refined": {
            "mIoU": round(miou_r, 4),
            "FWIoU": round(fwiou_r, 4),
            "Pixel_Acc": round(acc_r, 4)
        }
    }
    
    print("\n" + "="*40)
    print("Test Metrics Report:")
    print(json.dumps(results, indent=4))
    print("-" * 40)
    print(f"{'Class':<8} {'Coarse IoU':>12} {'Refined IoU':>12}")
    for i in range(NUM_CLASSES):
        print(f"Class {i:<2}  {iou_c[i].item():>12.4f} {iou_r[i].item():>12.4f}")
    print("="*40)

    # 保存结果到 JSON
    with open(save_dir / "test_metrics.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    run_test()