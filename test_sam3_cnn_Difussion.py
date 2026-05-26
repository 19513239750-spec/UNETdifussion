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

    # 3. 数据准备
    dataset = MAFSegTestDataset(DATA_ROOT, split="test", image_size=IMAGE_SIZE, label_mapping=gt_map)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # 4. 指标容器
    conf = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.float64, device=device)

    print("Starting Inference...")
    for imgs, masks, filenames in tqdm(loader):
        imgs, masks = imgs.to(device), masks.to(device)
        
        # 前向传播 (取分割分支输出)
        output = model(imgs)
        logits = output[0] if isinstance(output, tuple) else output
        pred = logits.argmax(dim=1)

        # 更新混淆矩阵 (忽略 255)
        valid = masks != 255
        idx = masks[valid] * NUM_CLASSES + pred[valid]
        conf += torch.bincount(idx, minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)

        # 可视化对比保存
        fn = filenames[0]
        # 反归一化原图
        img_np = (imgs[0].cpu().numpy().transpose(1, 2, 0) * np.array([0.229, 0.224, 0.225]) + 
                  np.array([0.485, 0.456, 0.406])) * 255
        img_np = img_np.clip(0, 255).astype(np.uint8)
        
        # 转换预测和真值为彩色图
        pred_np = pred[0].cpu().numpy()
        gt_np = masks[0].cpu().numpy()
        vis_pred = np.zeros_like(img_np)
        vis_gt = np.zeros_like(img_np)
        for c in range(NUM_CLASSES):
            vis_pred[pred_np == c] = colors[c]
            vis_gt[gt_np == c] = colors[c]
            
        # 拼接图片: 原图 | 真值 | 预测
        combined = np.concatenate([img_np, vis_gt, vis_pred], axis=1)
        Image.fromarray(combined).save(save_dir / "vis" / f"res_{fn}")

    # 5. 指标计算
    intersection = conf.diag()
    union = conf.sum(1) + conf.sum(0) - intersection + 1e-6
    iou = intersection / union
    
    miou = iou.mean().item()
    
    # 计算 FWIoU (Frequency Weighted IoU)
    freq = conf.sum(1) / (conf.sum() + 1e-6)
    fwiou = (freq * iou).sum().item()
    
    pixel_acc = (intersection.sum() / (conf.sum() + 1e-6)).item()

    # 6. 结果汇总
    results = {
        "mIoU": round(miou, 4),
        "FWIoU": round(fwiou, 4),
        "Pixel_Acc": round(pixel_acc, 4)
    }
    
    print("\n" + "="*30)
    print("Test Metrics Report:")
    print(json.dumps(results, indent=4))
    print("-" * 30)
    for i in range(NUM_CLASSES):
        print(f"Class {i:<2} IoU: {iou[i].item():.4f}")
    print("="*30)

    # 保存结果到 JSON
    with open(save_dir / "test_metrics.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    run_test()