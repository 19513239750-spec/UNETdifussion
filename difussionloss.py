import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss 实现，用于处理类别不平衡和难分类样本。
    """
    def __init__(self, alpha=0.5, gamma=2.0, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        # 计算原始的 CrossEntropy
        ce_loss = F.cross_entropy(logits, targets, ignore_index=self.ignore_index, reduction='none')
        
        # pt 是模型预测正确类别的概率
        pt = torch.exp(-ce_loss)
        
        # Focal Loss 公式: FL = alpha * (1 - pt)^gamma * CE
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # 返回平均损失
        return focal_loss.mean()

class DiffusionLoss(nn.Module):
    def __init__(self, 
                 focal_weight=1.0, 
                 dice_weight=1.0, 
                 boundary_weight=1.5, 
                 diffusion_weight=1.5,
                 repa_weight=0.5,
                 ignore_index=255):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.diffusion_weight = diffusion_weight
        self.repa_weight = repa_weight
        self.ignore_index = ignore_index
        
        self.focal_loss = FocalLoss(alpha=0.5, gamma=2.0, ignore_index=ignore_index)
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, 
                coarse_logits: torch.Tensor,
                targets: torch.Tensor,
                boundary_logits: torch.Tensor = None,
                boundary_gt: torch.Tensor = None,
                noise_pred: torch.Tensor = None,
                true_noise: torch.Tensor = None,
                repa_feat: torch.Tensor = None,
                teacher_feat: torch.Tensor = None
                ) -> torch.Tensor:
        """
        Parameters
        ----------
        coarse_logits   : [B, num_classes, H, W]  coarse segmentation logits
        targets         : [B, H, W]  integer labels 0..6, ignore_index=255
        boundary_logits : [B, 1, H, W]  boundary prediction logits (optional)
        boundary_gt     : [B, 1, H, W]  binary boundary ground truth (optional)
        noise_pred      : [B, num_classes, H, W]  predicted noise (optional)
        true_noise      : [B, num_classes, H, W]  ground-truth noise (optional)
        """
        coarse_logits = coarse_logits.float()

        # ── 1. Coarse segmentation loss (Focal + Dice) ──────────────────────
        loss_focal = self.focal_loss(coarse_logits, targets)

        num_classes = coarse_logits.shape[1]
        probs = F.softmax(coarse_logits, dim=1)
        valid_mask = (targets != self.ignore_index).unsqueeze(1)

        targets_safe = torch.where(targets == self.ignore_index, torch.zeros_like(targets), targets)
        targets_onehot = F.one_hot(targets_safe, num_classes=num_classes).permute(0, 3, 1, 2).float()

        probs_masked = probs * valid_mask
        targets_onehot_masked = targets_onehot * valid_mask

        loss_dice = self._compute_dice(probs_masked, targets_onehot_masked)

        # ── 2. Boundary loss ────────────────────────────────────────────────
        if boundary_logits is not None and boundary_gt is not None:
            # Explicit binary boundary supervision (BCE)
            bl = boundary_logits.float()
            if bl.shape != boundary_gt.shape:
                bl = F.interpolate(bl, size=boundary_gt.shape[2:], mode='bilinear', align_corners=False)
            loss_boundary = self.bce_loss(bl, boundary_gt.float())
        else:
            # Fallback: implicit boundary separation loss from probs
            loss_boundary = self._compute_separation_boundary_loss(probs_masked, targets_onehot_masked)

        # ── 3. Diffusion noise prediction loss (MSE) ─────────────────────────
        loss_diffusion = torch.tensor(0.0, device=coarse_logits.device)
        if noise_pred is not None and true_noise is not None:
            np_ = noise_pred.float()
            if np_.shape != true_noise.shape:
                np_ = F.interpolate(np_, size=true_noise.shape[2:], mode='bilinear', align_corners=False)
            loss_diffusion = self.mse_loss(np_, true_noise.float())

        # ── REPA feature alignment loss (cosine) ─────────────────────────────
        loss_repa = torch.tensor(0.0, device=coarse_logits.device)
        if repa_feat is not None and teacher_feat is not None:
            teacher_resized = F.interpolate(teacher_feat.float(), size=repa_feat.shape[2:], mode='bilinear', align_corners=False)
            rf = repa_feat.float()
            tf = teacher_resized.float()
            cos_sim = (rf * tf).sum(dim=1) / (rf.norm(dim=1) * tf.norm(dim=1) + 1e-8)
            loss_repa = -cos_sim.mean()
            # cos_sim = F.cosine_similarity(repa_feat.float(), teacher_resized.float(), dim=1)
            # loss_repa = -cos_sim.mean()

        # ── Total loss ────────────────────────────────────────────────────────
        total_loss = (self.focal_weight * loss_focal +
                      self.dice_weight * loss_dice +
                      self.boundary_weight * loss_boundary +
                      self.diffusion_weight * loss_diffusion +
                      self.repa_weight * loss_repa)

        return total_loss

    def _compute_separation_boundary_loss(self, probs, targets_onehot, eps=1e-5):
        k_size = 5 
        targets_max = F.max_pool2d(targets_onehot, kernel_size=k_size, stride=1, padding=k_size//2)
        targets_min = -F.max_pool2d(-targets_onehot, kernel_size=k_size, stride=1, padding=k_size//2)
        boundary_targets = targets_max - targets_min 

        probs_max = F.max_pool2d(probs, kernel_size=k_size, stride=1, padding=k_size//2)
        probs_min = -F.max_pool2d(-probs, kernel_size=k_size, stride=1, padding=k_size//2)
        boundary_preds = probs_max - probs_min

        intersection = torch.sum(boundary_preds * boundary_targets, dim=(2, 3))
        fp = torch.sum(boundary_preds * (1 - boundary_targets), dim=(2, 3)) 
        fn = torch.sum((1 - boundary_preds) * boundary_targets, dim=(2, 3))
        
        boundary_score = (intersection + eps) / (intersection + 0.7 * fn + 0.3 * fp + eps)
        return 1.0 - boundary_score.mean()

    def _compute_dice(self, probs, targets_onehot, eps=1e-5):
        intersection = torch.sum(probs * targets_onehot, dim=(2, 3))
        cardinality = torch.sum(probs + targets_onehot, dim=(2, 3))
        dice_score = (2.0 * intersection + eps) / (cardinality + eps)
        return 1.0 - dice_score.mean()
