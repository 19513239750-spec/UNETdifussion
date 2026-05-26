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
                 ignore_index=255):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.diffusion_weight = diffusion_weight
        self.ignore_index = ignore_index
        
        # 现在 FocalLoss 已经定义，不会再报错
        self.focal_loss = FocalLoss(alpha=0.5, gamma=2.0, ignore_index=ignore_index)
        self.mse_loss = nn.MSELoss() 

    def forward(self, 
                logits: torch.Tensor, 
                targets: torch.Tensor, 
                noise_pred: torch.Tensor = None, 
                true_noise: torch.Tensor = None  
                ) -> torch.Tensor:
        
        logits = logits.float() 
        
        # 1. 基础分割损失 (Focal + Dice)
        loss_focal = self.focal_loss(logits, targets)
        
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        valid_mask = (targets != self.ignore_index).unsqueeze(1)
        
        targets_safe = torch.where(targets == self.ignore_index, torch.zeros_like(targets), targets)
        targets_onehot = F.one_hot(targets_safe, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        probs = probs * valid_mask
        targets_onehot = targets_onehot * valid_mask

        loss_dice = self._compute_dice(probs, targets_onehot)

        # 2. 增强型边界分离损失
        loss_boundary = self._compute_separation_boundary_loss(probs, targets_onehot)

        # 3. Diffusion 噪声损失
        loss_diffusion = 0
        if noise_pred is not None and true_noise is not None:
            # 确保尺寸一致
            if noise_pred.shape != true_noise.shape:
                noise_pred = F.interpolate(noise_pred, size=true_noise.shape[2:], mode='bilinear', align_corners=False)
            loss_diffusion = self.mse_loss(noise_pred, true_noise)

        # 综合损失
        total_loss = (self.focal_weight * loss_focal + 
                      self.dice_weight * loss_dice + 
                      self.boundary_weight * loss_boundary +
                      self.diffusion_weight * loss_diffusion)
        
        return total_loss
    

    # def _compute_separation_boundary_loss(self, probs, targets_onehot, eps=1e-5):
    #     """
    #     改进：重点惩罚 False Positive (FP)，即本该是间隙却连在一起的部分。
    #     """
    #     k_size = 5 # 对于极小的浮阀，可以尝试 k_size=3；对于大池塘，k_size=5 较好
        
    #     # 1. 提取真值边界
    #     targets_max = F.max_pool2d(targets_onehot, kernel_size=k_size, stride=1, padding=k_size//2)
    #     targets_min = -F.max_pool2d(-targets_onehot, kernel_size=k_size, stride=1, padding=k_size//2)
    #     boundary_targets = targets_max - targets_min 

    #     # 2. 提取预测边界
    #     probs_max = F.max_pool2d(probs, kernel_size=k_size, stride=1, padding=k_size//2)
    #     probs_min = -F.max_pool2d(-probs, kernel_size=k_size, stride=1, padding=k_size//2)
    #     boundary_preds = probs_max - probs_min

    #     # 3. 计算交集和错位
    #     intersection = torch.sum(boundary_preds * boundary_targets, dim=(2, 3))
        
    #     # FP: 预测为边界但实际不是（即模型在间隙处乱涂，导致黏连）
    #     fp = torch.sum(boundary_preds * (1 - boundary_targets), dim=(2, 3)) 
    #     # FN: 实际是边界但没预测出来（即边界丢失）
    #     fn = torch.sum((1 - boundary_preds) * boundary_targets, dim=(2, 3))
        
    #     # --- 核心调整点 ---
    #     # 原始：0.7 * fn + 0.3 * fp (侧重找回边界)
    #     # 建议：0.3 * fn + 0.8 * fp (侧重严惩黏连/多余预测)
    #     # 提高 FP 的权重，会让模型在面对“到底连不连”时，倾向于“断开”以降低 FP。
    #     boundary_score = (intersection + eps) / (intersection + 0.3 * fn + 0.8 * fp + eps)
        
    #     return 1.0 - boundary_score.mean()


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