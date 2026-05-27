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

try:
    from sam3_unet.loss.difussionloss import FocalLoss
except ModuleNotFoundError:
    from difussionloss import FocalLoss

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


def compute_boundary_gt(mask: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
    """
    Compute 4-neighbour boundary ground truth from a semantic mask.

    Parameters
    ----------
    mask : torch.Tensor  [B, H, W]  long tensor with class labels 0..6 and ignore_index
    ignore_index : int   pixels with this value are excluded from boundary detection

    Returns
    -------
    torch.Tensor  [B, 1, H, W]  float, 1 at boundary pixels, 0 elsewhere
    """
    valid = (mask != ignore_index)
    # Replace ignored pixels with -1 so they don't trigger false boundaries
    m = mask.clone()
    m[~valid] = -1

    boundary = torch.zeros_like(m, dtype=torch.float32)

    # Horizontal neighbours
    diff_h = (m[:, :, 1:] != m[:, :, :-1]) & valid[:, :, 1:] & valid[:, :, :-1]
    boundary[:, :, 1:]  = boundary[:, :, 1:].masked_fill(diff_h, 1.0)
    boundary[:, :, :-1] = boundary[:, :, :-1].masked_fill(diff_h, 1.0)

    # Vertical neighbours
    diff_v = (m[:, 1:, :] != m[:, :-1, :]) & valid[:, 1:, :] & valid[:, :-1, :]
    boundary[:, 1:, :]  = boundary[:, 1:, :].masked_fill(diff_v, 1.0)
    boundary[:, :-1, :] = boundary[:, :-1, :].masked_fill(diff_v, 1.0)

    return boundary.unsqueeze(1)  # [B, 1, H, W]


class Stage1SegLoss(nn.Module):
    """CE + Dice + Focal for backbone coarse segmentation training."""
    def __init__(self, num_classes=7, ignore_index=255, ce_weight=1.0, dice_weight=1.0, focal_weight=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.focal_loss = FocalLoss(alpha=0.5, gamma=2.0, ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, ignore_index=self.ignore_index)
        focal = self.focal_loss(logits, targets)

        probs = F.softmax(logits.float(), dim=1)
        valid_mask = (targets != self.ignore_index).unsqueeze(1)
        targets_safe = torch.where(targets == self.ignore_index, torch.zeros_like(targets), targets)
        onehot = F.one_hot(targets_safe, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        probs = probs * valid_mask
        onehot = onehot * valid_mask
        inter = torch.sum(probs * onehot, dim=(2, 3))
        card = torch.sum(probs + onehot, dim=(2, 3))
        dice = 1.0 - ((2.0 * inter + 1e-5) / (card + 1e-5)).mean()
        return self.ce_weight * ce + self.dice_weight * dice + self.focal_weight * focal


def build_diffusion_schedule(total_steps: int, device: torch.device):
    betas = torch.linspace(1e-4, 2e-2, total_steps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return {
        "T": total_steps,
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
    }


def sample_timesteps(batch_size: int, total_steps: int, device: torch.device):
    return torch.randint(0, total_steps, (batch_size,), device=device)


def add_mask_noise(gt_mask: torch.Tensor, t: torch.Tensor, schedule: Dict[str, torch.Tensor]):
    noise = torch.randn_like(gt_mask)
    s_alpha = schedule["sqrt_alphas_cumprod"][t.long()].view(gt_mask.shape[0], 1, 1, 1)
    s_noise = schedule["sqrt_one_minus_alphas_cumprod"][t.long()].view(gt_mask.shape[0], 1, 1, 1)
    noisy_mask = s_alpha * gt_mask + s_noise * noise
    return noisy_mask, noise


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
    # stage1: train backbone coarse segmentation with CE + Dice + Focal
    # stage2: freeze backbone and train conditional diffusion refinement with noise MSE
    parser.add_argument("--data_root", type=str, default="/workspace/MAF")
    parser.add_argument("--sam3_ckpt", type=str, default="/workspace/sam3/sam3.pt")
    parser.add_argument("--init_weights", type=str, default="", help="Optional checkpoint to resume stage training")
    parser.add_argument("--save_dir", type=str, default="/workspace/weights/SAM3UNetDiffusion2")
    parser.add_argument("--image_size", type=int, default=336)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train_stage", type=str, default="stage1", choices=["stage1", "stage2"])
    parser.add_argument("--diffusion_steps", type=int, default=1000)
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
    if args.init_weights:
        ckpt = torch.load(args.init_weights, map_location=device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        model.load_state_dict(ckpt, strict=False)
        print(f"Loaded initial weights from {args.init_weights}")

    model.configure_training_stage(args.train_stage)
    print(f"Training stage: {args.train_stage}")

    # 分层学习率分配
    params = []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        lr = args.lr
        if args.train_stage == "stage1":
            if "sam3_vit" in n: lr *= 0.1
            elif "eba" in n: lr *= 2.0
        else:
            if "diffusion_head" in n or "boundary_head" in n: lr *= 1.5
        params.append({"params": [p], "lr": lr})

    optimizer = torch.optim.AdamW(params, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    stage1_criterion = Stage1SegLoss(num_classes=7, ignore_index=255).to(device)
    stage2_criterion = nn.MSELoss().to(device)
    diffusion_schedule = build_diffusion_schedule(args.diffusion_steps, device) if args.train_stage == "stage2" else None
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_miou = -1.0
    history = {"epoch": [], "loss": [], "miou": [], "fwiou":[], "acc":[]}

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                if args.train_stage == "stage1":
                    coarse_logits, _ = model(imgs)
                    loss = stage1_criterion(coarse_logits, masks)
                else:
                    B = imgs.shape[0]
                    target_for_onehot = masks.clone()
                    target_for_onehot[target_for_onehot == 255] = 0
                    gt_mask = F.one_hot(target_for_onehot, num_classes=7).permute(0, 3, 1, 2).float()
                    gt_mask = gt_mask * 2.0 - 1.0
                    t = sample_timesteps(B, diffusion_schedule["T"], device).float()
                    noisy_mask, true_noise = add_mask_noise(gt_mask, t, diffusion_schedule)
                    _, _, noise_pred = model(imgs, noisy_mask, t)
                    loss = stage2_criterion(noise_pred, true_noise)

            # 反向传播
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
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