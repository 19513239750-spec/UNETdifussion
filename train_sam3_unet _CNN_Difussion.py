#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 假设您已经定义好了这个 Loss 类
from sam3_unet.loss.difussionloss import DiffusionLoss



T = 1000
betas = torch.linspace(1e-4, 2e-2, T)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).cuda()
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).cuda()
# ====================== 绘图库 ======================
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 8)


def _load_sam3_unet_cnn_cls():
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "SAM3UNetCNNDifussion.py",
        script_dir / "sam3_unet" / "sam3" / "SAM3UNetCNNDifussion.py",
    ]
    for p in candidates:
        if p.exists():
            spec = importlib.util.spec_from_file_location("SAM3UNetCNNDifussion", str(p))
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod.SAM3UNetCNNDifussion
    raise ModuleNotFoundError("Cannot find SAM3UNetCNNDifussion.py")

SAM3UNetCNNDifussion = _load_sam3_unet_cnn_cls()

def load_maf_fujian_meta(config_path: str) -> Dict[str, object]:
    out: Dict[str, object] = {}
    p = Path(config_path)
    if not p.exists(): return out
    src = p.read_text(encoding="utf-8")
    tree = ast.parse(src)
    wanted = {"GT_LABEL_MAPPING", "CLASS_COLORS", "CLASSES"}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in wanted:
                    try: out[t.id] = ast.literal_eval(node.value)
                    except: pass
    return out

# ====================== 改进的数据集类 ======================
class MAFSegDataset(Dataset):
    def __init__(self, root, split="train", image_size=336, num_classes=7, label_mapping=None):
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.num_classes = num_classes
        self.img_dir = self.root / split / "image"
        self.mask_dir = self.root / split / "mask"
        self.images = sorted(list(self.img_dir.glob("*.png")) + list(self.img_dir.glob("*.jpg")))
        self.default_mapping = label_mapping or {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}

       
        if split == "train":
            self.transform = A.Compose([
    # 1. scale 必须在 0 到 1 之间 (例如 0.5 到 1.0)
    # 2. 使用 size 参数或者确保符合新版 API
    A.RandomResizedCrop(
        size=(image_size, image_size), 
        scale=(0.5, 1.0), 
        p=1.0
    ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.5),
                A.OneOf([
                    A.GridDistortion(p=0.3),
                    A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.3),
                ], p=0.2),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

    def _load_mask(self, p: Path) -> np.ndarray:
        m = np.array(Image.open(p).convert("L"))
        # 简单映射逻辑
        out = np.full_like(m, 255)
        for k, v in self.default_mapping.items():
            out[m == k] = v
        return out

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img_p = self.images[idx]
        mask_p = self.mask_dir / f"{img_p.stem}.png"
        img = np.array(Image.open(img_p).convert("RGB"))
        mask = self._load_mask(mask_p)
        
        augmented = self.transform(image=img, mask=mask)
        return augmented["image"], augmented["mask"].long()

# ====================== 改进的评价函数 (支持迭代细化) ======================
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> Dict[str, float]:
    model.eval()
    conf = torch.zeros((num_classes, num_classes), dtype=torch.float64, device=device)
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        output = model(imgs) 
        logits = output[0] if isinstance(output, tuple) else output
        
        pred = logits.argmax(dim=1)
        valid = masks != 255
        idx = masks[valid] * num_classes + pred[valid]
        conf += torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)

    # 计算 IoU
    iou = conf.diag() / (conf.sum(1) + conf.sum(0) - conf.diag() + 1e-6)
    miou = float(iou.mean().item())
    
    # --- 新增 FWIoU 计算 ---
    freq = conf.sum(1) / (conf.sum() + 1e-6)
    fwiou = float((freq * iou).sum().item())
    
    # Pixel Accuracy
    pixel_acc = float((conf.diag().sum() / (conf.sum() + 1e-6)).item())
    
    return {"mIoU": miou, "FWIoU": fwiou, "pixel_acc": pixel_acc}


def plot_curves(epochs, losses, mious, fwious, accs, save_path):
    fig, axes = plt.subplots(4, 1, figsize=(10, 15))
    axes[0].plot(epochs, losses, 'r-'); axes[0].set_title("Train Loss")
    axes[1].plot(epochs, mious, 'g-'); axes[1].set_title("Val mIoU")
    axes[2].plot(epochs, fwious, 'm-'); axes[2].set_title("Val FWIoU")
    axes[3].plot(epochs, accs, 'b-'); axes[3].set_title("Val Pixel Acc")
    for ax in axes: ax.grid(True)
    plt.tight_layout()
    plt.savefig(save_path); plt.close()
# ====================== 主训练函数 ======================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/workspace/MAF")
    parser.add_argument("--sam3_ckpt", type=str, default="/workspace/sam3/sam3.pt")
    parser.add_argument("--save_dir", type=str, default="/workspace/weights/SAM3UNetDiffusion2")
    parser.add_argument("--image_size", type=int, default=336)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--maf_config", type=str, default="/workspace/MAF_Seg/config/maf_fujian.py")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)

    meta = load_maf_fujian_meta(args.maf_config)
    gt_map = meta.get("GT_LABEL_MAPPING", None)

    train_loader = DataLoader(MAFSegDataset(args.data_root, "train", args.image_size, label_mapping=gt_map), 
                              batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(MAFSegDataset(args.data_root, "test", args.image_size, label_mapping=gt_map), 
                            batch_size=2, shuffle=False, num_workers=4)

    model = SAM3UNetCNNDifussion(checkpoint_path=args.sam3_ckpt, img_size=args.image_size, num_classes=7).to(device)

    # 分层学习率分配
    params = []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        lr = args.lr
        if "sam3_vit" in n: lr *= 0.1
        elif "eba" in n: lr *= 2.0
        elif "diffusion_head" in n: lr *= 1.5 # 为扩散头设置较高学习率
        params.append({"params": [p], "lr": lr})

    optimizer = torch.optim.AdamW(params, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 初始化改进后的损失函数
    criterion = DiffusionLoss(ignore_index=255).to(device)
    scaler = torch.cuda.amp.GradScaler()

    best_miou = -1.0
    history = {"epoch": [], "loss": [], "miou": [], "fwiou":[], "acc":[]}

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        # --- 修改训练循环中的前向部分 ---
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                B = imgs.shape[0]
        
        # 1. 构造 7 通道的 GT One-Hot 掩码 [B, 7, H, W]
        # 处理 255 标签，防止 one_hot 越界（将其暂时指向背景 0，计算 Loss 时会自动忽略）
                target_for_onehot = masks.clone()
                target_for_onehot[target_for_onehot == 255] = 0 
        
        # 生成 one_hot 并转为 [B, 7, H, W]
                gt_mask = F.one_hot(target_for_onehot, num_classes=7).permute(0, 3, 1, 2).float()
        
        # 关键：归一化到 [-1, 1] 范围，这是扩散模型的标准分布范围
                gt_mask = gt_mask * 2.0 - 1.0 

        # 2. 时间步
                t = torch.randint(0, T, (B,), device=device).float()

        # 3. 扩散加噪 (7通道噪声)
                noise = torch.randn_like(gt_mask)
        
        # 提取系数并对齐维度 [B, 1, 1, 1]
                s_alpha = sqrt_alphas_cumprod[t.long()].view(B, 1, 1, 1)
                s_noise = sqrt_one_minus_alphas_cumprod[t.long()].view(B, 1, 1, 1)
        
        # 生成带噪掩码
                noisy_mask = s_alpha * gt_mask + s_noise * noise

        # 4. 模型前向 (此时 noisy_mask 是 7 通道)
                logits, noise_pred = model(imgs, noisy_mask, t)

        # 5. 计算损失
        # logits: [B, 7, H, W] -> 与 masks(类别索引) 算交叉熵
        # noise_pred: [B, 7, H, W] -> 与 noise 算 MSE
                loss = criterion(logits, masks, noise_pred, noise)

    # 反向传播 ...
    # 反向传播
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        pbar.set_postfix(loss=f"{running_loss/(pbar.n+1):.4f}")

        # 评估与记录
        metrics = evaluate(model, val_loader, device, 7)
        avg_loss = running_loss / len(train_loader)
        scheduler.step()

        history["epoch"].append(epoch)
        history["loss"].append(avg_loss)
        history["miou"].append(metrics["mIoU"])
        history["fwiou"].append(metrics["FWIoU"])
        history["acc"].append(metrics["pixel_acc"])


        if metrics["mIoU"] > best_miou:
            best_miou = metrics["mIoU"]
            torch.save(model.state_dict(), save_dir / "best.pt")
            print(f"新最佳模型已保存: mIoU = {best_miou:.4f}, FWIoU = {metrics['FWIoU']:.4f}")

        plot_curves(history["epoch"], history["loss"], history["miou"], history["fwiou"], history["acc"], save_dir / "curves.png")
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | mIoU: {metrics['mIoU']:.4f} | FWIoU: {metrics['FWIoU']:.4f}")

    # 训练结束后保存一次
        torch.save(model.state_dict(), save_dir / "final_model.pt")
        print(f"训练完成，模型已保存至 {save_dir / 'final_model.pt'}")

if __name__ == "__main__":
    torch.manual_seed(42)
    main()