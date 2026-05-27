import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_   
from sam3.model.vitdet import ViT




########################################unetformer相关模块########################################
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6()
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
###############################################################

class WF1(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WF1, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = F.interpolate(x, size=(res.shape[2], res.shape[3]), mode="bilinear", align_corners=False)
        res = self.pre_conv(res)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * res + fuse_weights[1] * x
        x = self.post_conv(x)
        return x


from einops import rearrange
class GlobalLocalAttention(nn.Module):
    def __init__(self,
                 dim=256,
                 num_heads=16,
                 qkv_bias=False,
                 window_size=8,
                 relative_pos_embedding=True
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size

        self.qkv = Conv(dim, 3*dim, kernel_size=1, bias=qkv_bias)
        self.local1 = ConvBN(dim, dim, kernel_size=3)
        self.local2 = ConvBN(dim, dim, kernel_size=1)
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

        self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1,  padding=(window_size//2 - 1, 0))
        self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1, padding=(0, window_size//2 - 1))

        self.relative_pos_embedding = relative_pos_embedding


        self.reduce = nn.Sequential(
            nn.Conv2d(dim, dim//4, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(dim//4),
            nn.ReLU(inplace=True),
        )

        if self.relative_pos_embedding:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.ws - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps), mode='constant')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='constant')
        return x

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='constant')#reflect
        return x

    def forward(self, x):
        B, C, H, W = x.shape


        local = self.local2(x) + self.local1(x)

        x = self.pad(x, self.ws)
        B, C, Hp, Wp = x.shape
        qkv = self.qkv(x)

        q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                            d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, qkv=3, ws1=self.ws, ws2=self.ws)
     
        dots = (q @ k.transpose(-2, -1)) * self.scale  #######torch.Size([64, 8]) torch.Size([8, 64]) -> torch.Size([64, 64])

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            #print(relative_position_bias.unsqueeze(0).shape)
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        attn = attn @ v 


        attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
                         d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, ws1=self.ws, ws2=self.ws)

        attn = attn[:, :, :H, :W]
        out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + \
              self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))

        out = out + local
        out = self.pad_out(out)
        out = self.proj(out)
        out = out[:, :, :H, :W]

        return out



class LightBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, 1),
            nn.BatchNorm2d(in_channels//4, 1),
            nn.GELU()
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(in_channels//2, out_channels, 1),
            nn.BatchNorm2d(out_channels, 1),
            nn.GELU()
        )
        self.dw1 = nn.Sequential(
            nn.Conv2d(in_channels//8, in_channels//8, kernel_size=3, stride=1, padding=1, groups=in_channels//8),
            nn.BatchNorm2d(in_channels//8),
            nn.GELU()
        )
        self.dw2 = nn.Sequential(
            nn.Conv2d(in_channels//8, in_channels//8, kernel_size=3, stride=1, padding=1, groups=in_channels//8),
            nn.BatchNorm2d(in_channels//8),
            nn.GELU()
        )

    def forward(self, x):
        x = self.conv_in(x)
        x1, x2 = torch.split(x, x.shape[1]//2, 1)
        x3 = self.dw1(x2)
        x4 = self.dw2(x3)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.conv_out(x)
        return x
    
    
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        self.conv = LightBlock(in_channels, out_channels)

    def forward(self, x1, x2=None):
        if x2 is not None:
            diffY = x1.size()[2] - x2.size()[2]
            diffX = x1.size()[3] - x2.size()[3]
            x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
            x = torch.cat([x1, x2], dim=1)
        else:
            x = x1
        x = self.up(x)
        return self.conv(x)


class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )
        self.init_weights()

    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net
    
    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.prompt_learn.apply(_init_weights)


# class MaskDiffusionHead(nn.Module):
#     def __init__(self, feature_dim=128, hidden_dim=64):
#         super().__init__()
#         # 时间步嵌入 (Time Embedding)
#         self.time_mlp = nn.Sequential(
#             nn.Linear(1, hidden_dim),
#             nn.GELU(),
#             nn.Linear(hidden_dim, hidden_dim),
#         )
        
#         # 掩码处理层 (处理 1 通道的噪声掩码)
#         self.mask_in = nn.Conv2d(1, hidden_dim, kernel_size=3, padding=1)
        
#         # 特征融合层 (融合来自 SAM3+CNN 的 128维强特征)
#         self.feat_cond = nn.Conv2d(feature_dim, hidden_dim, kernel_size=1)
        
#         # 去噪残差块
#         self.res_blocks = nn.ModuleList([
#             ConvBNReLU(hidden_dim * 2, hidden_dim), # 拼合 mask 和 condition
#             ConvBNReLU(hidden_dim, hidden_dim),
#             nn.Conv2d(hidden_dim, 1, kernel_size=1) # 输出预测的噪声
#         ])

#     def forward(self, x_t, t, cond_feat):
#         """
#         x_t: 当前步的掩码 [B, 1, H, W]
#         t: 时间步 [B, 1]
#         cond_feat: SAM3UNet 输出的 visual_features [B, 128, H, W]
#         """
#         # 1. 时间嵌入
       
#     # 1. 将 t 转换为 float，并增加特征维度从 [B] 变为 [B, 1]
#     # 2. 调用 .to(x_t.dtype) 确保它能自动匹配 autocast 环境下的 float16 或 float32
#         t = t.view(-1, 1).to(dtype=x_t.dtype)
    
#     # 现在 t 是浮点数且维度正确，可以进入 linear 层了
#         t_embed = self.time_mlp(t).view(-1, 64, 1, 1) 
    
        
#         # 2. 特征对齐
#         m_feat = self.mask_in(x_t) + t_embed
#         c_feat = self.feat_cond(cond_feat)
#         c_feat = F.interpolate(c_feat, size=x_t.shape[2:], mode='bilinear')
        
#         # 3. 拼接去噪
#         combined = torch.cat([m_feat, c_feat], dim=1)
#         out = self.res_blocks[0](combined)
#         out = self.res_blocks[1](out)
#         noise_pred = self.res_blocks[2](out)
        
#         return noise_pred
class AdaGN_ResidualBlock(nn.Module):
    def __init__(self, channels, cond_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        # 用视觉特征生成 scale 和 shift 参数
        self.cond_embed = nn.Linear(cond_dim, channels * 2) 
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x, cond_feat):
        # cond_feat: [B, cond_dim] (可以通过对视觉特征做全局平均池化得到)
        emb = self.cond_embed(cond_feat).unsqueeze(-1).unsqueeze(-1)
        scale, shift = torch.chunk(emb, 2, dim=1)
        
        res = x
        x = self.norm1(x)
        x = x * (1 + scale) + shift # 核心调制公式
        x = self.conv1(self.act(x))
        x = self.conv2(self.act(x))
        return x + res

class MaskEmbedding(nn.Module):
    """Embed discrete mask logits/one-hot into continuous features for latent encoding."""
    def __init__(self, in_channels: int, embed_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LatentEncoder(nn.Module):
    """Lightweight VAE-style encoder (deterministic) for latent diffusion."""
    def __init__(self, in_channels: int, latent_channels: int = 4, base_channels: int = 64, num_down: int = 2):
        super().__init__()
        layers = [nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)]
        ch = base_channels
        for _ in range(num_down):
            layers += [
                nn.GELU(),
                nn.Conv2d(ch, ch * 2, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(8, ch * 2)
            ]
            ch *= 2
        layers += [nn.GELU(), nn.Conv2d(ch, latent_channels, kernel_size=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LatentDecoder(nn.Module):
    """Lightweight decoder to map latent mask features back to logits."""
    def __init__(self, latent_channels: int = 4, out_channels: int = 7, base_channels: int = 64, num_up: int = 2):
        super().__init__()
        ch = base_channels * (2 ** num_up)
        layers = [nn.Conv2d(latent_channels, ch, kernel_size=1)]
        for _ in range(num_up):
            layers += [
                nn.GELU(),
                nn.ConvTranspose2d(ch, ch // 2, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(8, ch // 2)
            ]
            ch //= 2
        layers += [nn.GELU(), nn.Conv2d(ch, out_channels, kernel_size=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class MaskDiffusionHead(nn.Module):
    def __init__(self, latent_channels=4, hidden_dim=128):
        super().__init__()
        # Conditioning: t(1) + image latent + coarse latent + boundary map (1)
        cond_dim = 1 + latent_channels + latent_channels + 1
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.mask_in = nn.Conv2d(latent_channels, hidden_dim, kernel_size=3, padding=1)
        self.res_block = AdaGN_ResidualBlock(hidden_dim, hidden_dim)
        self.out_conv = nn.Conv2d(hidden_dim, latent_channels, kernel_size=1)

    def forward(self, x_t, t, image_latent, coarse_latent, boundary_logits):
        """
        x_t:            [B, latent_channels, H, W]  noisy latent mask
        t:              [B]                      diffusion timestep
        image_latent:   [B, latent_channels, H, W]  image latent condition
        coarse_latent:  [B, latent_channels, H, W]  coarse mask latent condition
        boundary_logits:[B, 1, H, W]             boundary prior
        """
        image_global = F.adaptive_avg_pool2d(image_latent, 1).flatten(1)
        coarse_global = F.adaptive_avg_pool2d(coarse_latent, 1).flatten(1)
        boundary_global = F.adaptive_avg_pool2d(boundary_logits, 1).flatten(1)
        t_input = torch.cat(
            [t.view(-1, 1).to(x_t.dtype), image_global, coarse_global, boundary_global], dim=1
        )
        c_emb = self.cond_mlp(t_input)
        x = self.mask_in(x_t)
        x = self.res_block(x, c_emb)
        return self.out_conv(x)


class BoundaryHead(nn.Module):
    """Lightweight head to predict a binary boundary map from visual features."""
    def __init__(self, in_channels=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1)
        )

    def forward(self, x):
        return self.conv(x)

class EBAModule(nn.Module):
    """
    Edge Boundary Attention (EBA) Feature Fusion Module
    专门用于编码器阶段：利用 CNN 提取的高频边缘注意力，引导与 ViT 语义特征的融合。
    """
    def __init__(self, channels=1024):
        super().__init__()
        # 1. 从 CNN 特征中提取边缘/高频注意力分布
        self.edge_extractor = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.GELU(),
            nn.Conv2d(channels // 4, 1, kernel_size=1) # 输出 1 通道的空间注意力图
        )
        self.gate = nn.Sigmoid()

        # 2. 跨模态特征融合卷积
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

    def forward(self, vit_feat: torch.Tensor, cnn_feat: torch.Tensor) -> torch.Tensor:
        # 1. 从局部细节丰富的 CNN 特征中生成边缘注意力图 (Edge Attention Map)
        edge_logits = self.edge_extractor(cnn_feat)
        edge_attn = self.gate(edge_logits) # shape: [B, 1, H, W]

        # 2. 边缘引导解耦 (核心创新点)
        # 逻辑：对于 ViT 特征，抑制其边界处的模糊语义 (1 - edge_attn)，保留纯净的主体
        #      对于 CNN 特征，强化其边界处的高清细节 (edge_attn)
        vit_refined = vit_feat * (1.0 - edge_attn)
        cnn_refined = cnn_feat * edge_attn

        # 3. 拼接并降维融合，输出与原通道数一致的增强特征
        fused = torch.cat([vit_refined, cnn_refined], dim=1)
        out_feat = self.fuse_conv(fused)
        
        return out_feat



import torch
import torch.nn.functional as F
import torch.nn as nn 


class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num:int, 
                 group_num:int = 16, 
                 eps:float = 1e-10
                 ):
        super(GroupBatchnorm2d,self).__init__()
        assert c_num    >= group_num
        self.group_num  = group_num
        self.weight     = nn.Parameter( torch.randn(c_num, 1, 1)    )
        self.bias       = nn.Parameter( torch.zeros(c_num, 1, 1)    )
        self.eps        = eps
    def forward(self, x):
        N, C, H, W  = x.size()
        x           = x.view(   N, self.group_num, -1   )
        mean        = x.mean(   dim = 2, keepdim = True )
        std         = x.std (   dim = 2, keepdim = True )
        x           = (x - mean) / (std+self.eps)
        x           = x.view(N, C, H, W)
        return x * self.weight + self.bias


class SRU(nn.Module):
    def __init__(self,
                 oup_channels:int, 
                 group_num:int = 16,
                 gate_treshold:float = 0.5,
                 torch_gn:bool = True
                 ):
        super().__init__()
        
        self.gn             = nn.GroupNorm( num_channels = oup_channels, num_groups = group_num ) if torch_gn else GroupBatchnorm2d(c_num = oup_channels, group_num = group_num)
        self.gate_treshold  = gate_treshold
        self.sigomid        = nn.Sigmoid()

    def forward(self,x):
        gn_x        = self.gn(x)
        w_gamma     = self.gn.weight/sum(self.gn.weight)
        w_gamma     = w_gamma.view(1,-1,1,1)
        reweigts    = self.sigomid( gn_x * w_gamma )
        # Gate
        w1          = torch.where(reweigts > self.gate_treshold, torch.ones_like(reweigts), reweigts) # 大于门限值的设为1，否则保留原值
        w2          = torch.where(reweigts > self.gate_treshold, torch.zeros_like(reweigts), reweigts) # 大于门限值的设为0，否则保留原值
        x_1         = w1 * x
        x_2         = w2 * x
        y           = self.reconstruct(x_1,x_2)
        return y
    
    def reconstruct(self,x_1,x_2):
        x_11,x_12 = torch.split(x_1, x_1.size(1)//2, dim=1)
        x_21,x_22 = torch.split(x_2, x_2.size(1)//2, dim=1)
        return torch.cat([ x_11+x_22, x_12+x_21 ],dim=1)


class CRU(nn.Module):
    '''
    alpha: 0<alpha<1
    '''
    def __init__(self, 
                 op_channel:int,
                 alpha:float = 1/2,
                 squeeze_radio:int = 2 ,
                 group_size:int = 2,
                 group_kernel_size:int = 3,
                 ):
        super().__init__()
        self.up_channel     = up_channel   =   int(alpha*op_channel)
        self.low_channel    = low_channel  =   op_channel-up_channel
        self.squeeze1       = nn.Conv2d(up_channel,up_channel//squeeze_radio,kernel_size=1,bias=False)
        self.squeeze2       = nn.Conv2d(low_channel,low_channel//squeeze_radio,kernel_size=1,bias=False)
        #up
        self.GWC            = nn.Conv2d(up_channel//squeeze_radio, op_channel,kernel_size=group_kernel_size, stride=1,padding=group_kernel_size//2, groups = group_size)
        self.PWC1           = nn.Conv2d(up_channel//squeeze_radio, op_channel,kernel_size=1, bias=False)
        #low
        self.PWC2           = nn.Conv2d(low_channel//squeeze_radio, op_channel-low_channel//squeeze_radio,kernel_size=1, bias=False)
        self.advavg         = nn.AdaptiveAvgPool2d(1)

    def forward(self,x):
        # Split
        up,low  = torch.split(x,[self.up_channel,self.low_channel],dim=1)
        up,low  = self.squeeze1(up),self.squeeze2(low)
        # Transform
        Y1      = self.GWC(up) + self.PWC1(up)
        Y2      = torch.cat( [self.PWC2(low), low], dim= 1 )
        # Fuse
        out     = torch.cat( [Y1,Y2], dim= 1 )
        out     = F.softmax( self.advavg(out), dim=1 ) * out
        out1,out2 = torch.split(out,out.size(1)//2,dim=1)
        return out1+out2


class SCConv(nn.Module):
    def __init__(self,
                op_channel:int,
                group_num:int = 4,
                gate_treshold:float = 0.5,
                alpha:float = 1/2,
                squeeze_radio:int = 2 ,
                group_size:int = 2,
                group_kernel_size:int = 3,
                 ):
        super().__init__()
        self.SRU = SRU( op_channel, 
                       group_num            = group_num,  
                       gate_treshold        = gate_treshold )
        self.CRU = CRU( op_channel, 
                       alpha                = alpha, 
                       squeeze_radio        = squeeze_radio ,
                       group_size           = group_size ,
                       group_kernel_size    = group_kernel_size )
    
    def forward(self,x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x



class SCBasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(SCBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        # --- 核心修改：将标准卷积替换为 SCConv ---
        self.scconv = SCConv(planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # 应用 SCConv
        out = self.scconv(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class CNNFeatureBranch(nn.Module):
    def __init__(self, in_chans: int = 3, embed_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        self.dim = embed_dim

        # 替换为 SCBasicBlock
        self.stage1 = nn.Sequential(
            SCBasicBlock(64, self.dim, stride=2),
            SCBasicBlock(self.dim, self.dim, stride=1)
        )
        self.stage2 = SCBasicBlock(self.dim, self.dim, stride=2)
        self.stage3 = SCBasicBlock(self.dim, self.dim, stride=2)
        self.stage4 = SCBasicBlock(self.dim, self.dim, stride=2)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        c1 = self.stage1(x)  # 1/4
        c2 = self.stage2(c1) # 1/8
        c3 = self.stage3(c2) # 1/16
        c4 = self.stage4(c3) # 1/32
        return c1, c2, c3, c4
def _create_vit_backbone(img_size):
    """Create ViT backbone for visual feature extraction."""
    return ViT(
      #   img_size=1008,
        img_size=img_size,
        pretrain_img_size=336,
        patch_size=14,
        embed_dim=1024,
        depth=32,
        num_heads=16,
        mlp_ratio=4.625,
        norm_layer="LayerNorm",
        drop_path_rate=0.1,
        qkv_bias=True,
        use_abs_pos=True,
        tile_abs_pos=True,
        global_att_blocks=(7, 15, 23, 31),
        rel_pos_blocks=(),
        use_rope=True,
        use_interp_rope=True,
        window_size=24,
        pretrain_use_cls_token=True,
        retain_cls_token=False,
        ln_pre=True,
        ln_post=False,
        return_interm_layers=False,
        bias_patch_embed=False,
        # compile_mode=compile_mode,
        compile_mode=None,
    )


class SAM3UNetCNNDifussion(nn.Module):
    def __init__(self, checkpoint_path=None, img_size=336, num_classes=7,
                 latent_channels: int = 4, latent_embed_dim: int = 32,
                 latent_base_channels: int = 64, vae_checkpoint=None) -> None:
        super(SAM3UNetCNNDifussion, self).__init__()
        self.num_classes = num_classes
        self.latent_channels = latent_channels
        self.sam3_vit = _create_vit_backbone(img_size)
        self.cnn_branch = CNNFeatureBranch(in_chans=3, embed_dim=128)
        self.mask_embed = MaskEmbedding(num_classes, embed_dim=latent_embed_dim)
        self.image_encoder = LatentEncoder(in_channels=3, latent_channels=latent_channels,
                                           base_channels=latent_base_channels, num_down=2)
        self.mask_encoder = LatentEncoder(in_channels=latent_embed_dim, latent_channels=latent_channels,
                                          base_channels=latent_base_channels, num_down=2)
        self.mask_decoder = LatentDecoder(latent_channels=latent_channels, out_channels=num_classes,
                                          base_channels=latent_base_channels, num_up=2)
        self.diffusion_head = MaskDiffusionHead(latent_channels=latent_channels, hidden_dim=128)
        self.boundary_head = BoundaryHead(in_channels=128)


        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            new_ckpt = dict()
            for k, v in ckpt.items():
                if "detector.backbone.vision_backbone.trunk" in k and 'freqs_cis' not in k:
                    new_ckpt[k[len("detector.backbone.vision_backbone.trunk."):]] = v
            self.sam3_vit.load_state_dict(new_ckpt, strict=False)
        if vae_checkpoint:
            vae_state = torch.load(vae_checkpoint, map_location="cpu")
            if isinstance(vae_state, dict) and "state_dict" in vae_state:
                vae_state = vae_state["state_dict"]
            image_state = {k.replace("image_encoder.", ""): v for k, v in vae_state.items()
                           if k.startswith("image_encoder.")}
            mask_state = {k.replace("mask_encoder.", ""): v for k, v in vae_state.items()
                          if k.startswith("mask_encoder.")}
            decoder_state = {k.replace("mask_decoder.", ""): v for k, v in vae_state.items()
                             if k.startswith("mask_decoder.")}
            if image_state:
                self.image_encoder.load_state_dict(image_state, strict=False)
            if mask_state:
                self.mask_encoder.load_state_dict(mask_state, strict=False)
            if decoder_state:
                self.mask_decoder.load_state_dict(decoder_state, strict=False)
        for param in self.sam3_vit.parameters():
            param.requires_grad = False
        blocks = []
        for block in self.sam3_vit.blocks:
            blocks.append(
                Adapter(block)
            )  
        self.sam3_vit.blocks = nn.Sequential(
            *blocks
        )
        self.reduce1 = nn.Conv2d(1024, 128, 1)
        self.reduce2 = nn.Conv2d(1024, 128, 1)
        self.reduce3 = nn.Conv2d(1024, 128, 1)
        self.reduce4 = nn.Conv2d(1024, 128, 1)
        self.up1 = Up(256, 128)
        self.up2 = Up(256, 128)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 128)
        self.head = nn.Conv2d(128, num_classes, 1)
        
        self.wf3 = WF1(128, 128)
        self.wf2 = WF1(128, 128)
        self.wf1 = WF1(128, 128)

        self.gltb4 = GlobalLocalAttention(dim=128)
        self.gltb3 = GlobalLocalAttention(dim=128)
        self.gltb2 = GlobalLocalAttention(dim=128)
        self.gltb1 = GlobalLocalAttention(dim=128)

        self.eba_1 = EBAModule(channels=128)
        self.eba_2 = EBAModule(channels=128)
        self.eba_3 = EBAModule(channels=128)
        self.eba_4 = EBAModule(channels=128)
        self._default_requires_grad = {n: p.requires_grad for n, p in self.named_parameters()}
        self._train_stage = "stage1"

    def _set_module_requires_grad(self, module: nn.Module, requires_grad: bool):
        for p in module.parameters():
            p.requires_grad = requires_grad

    def _backbone_modules(self):
        return [
            self.sam3_vit,
            self.cnn_branch,
            self.reduce1, self.reduce2, self.reduce3, self.reduce4,
            self.up1, self.up2, self.up3, self.up4,
            self.head,
            self.wf1, self.wf2, self.wf3,
            self.gltb1, self.gltb2, self.gltb3, self.gltb4,
            self.eba_1, self.eba_2, self.eba_3, self.eba_4,
            self.boundary_head,
        ]

    def _diffusion_modules(self):
        return [self.mask_embed, self.image_encoder, self.mask_encoder, self.mask_decoder, self.diffusion_head]

    def configure_training_stage(self, stage: str = "stage1"):
        """
        stage1: train backbone (coarse segmentation) only.
        stage2: freeze backbone and train diffusion refinement only.
        """
        if stage not in {"stage1", "stage2"}:
            raise ValueError(f"Unsupported training stage: {stage}")
        self._train_stage = stage

        diffusion_prefixes = ("diffusion_head.", "mask_embed.", "image_encoder.",
                              "mask_encoder.", "mask_decoder.")
        if stage == "stage1":
            for n, p in self.named_parameters():
                p.requires_grad = self._default_requires_grad.get(n, True)
                if n.startswith(diffusion_prefixes):
                    p.requires_grad = False
        else:
            for n, p in self.named_parameters():
                p.requires_grad = n.startswith(diffusion_prefixes)
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep frozen submodules in eval mode to avoid BN/running-stat updates.
        if mode and self._train_stage == "stage1":
            for m in self._diffusion_modules():
                m.eval()
        if mode and self._train_stage == "stage2":
            for m in self._backbone_modules():
                m.eval()
        return self

    def encode_image_latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(x)

    def encode_mask_latent(self, mask: torch.Tensor, is_logits: bool = True) -> torch.Tensor:
        if is_logits:
            mask = F.softmax(mask, dim=1)
        emb = self.mask_embed(mask)
        latent = self.mask_encoder(emb)
        return torch.tanh(latent)

    def decode_mask_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return self.mask_decoder(latent)

    def _extract_features(self, x):
        """Extract visual features, coarse logits, and boundary logits at feature-map resolution."""
        vit_feat = self.sam3_vit(x)[-1]
        cnn_x1, cnn_x2, cnn_x3, cnn_x4 = self.cnn_branch(x)

        vit_x1 = F.interpolate(self.reduce1(vit_feat), size=cnn_x1.shape[2:], mode='bilinear', align_corners=False)
        vit_x2 = F.interpolate(self.reduce2(vit_feat), size=cnn_x2.shape[2:], mode='bilinear', align_corners=False)
        vit_x3 = F.interpolate(self.reduce3(vit_feat), size=cnn_x3.shape[2:], mode='bilinear', align_corners=False)
        vit_x4 = F.interpolate(self.reduce4(vit_feat), size=cnn_x4.shape[2:], mode='bilinear', align_corners=False)

        fused_x1 = self.eba_1(vit_x1, cnn_x1)
        fused_x2 = self.eba_2(vit_x2, cnn_x2)
        fused_x3 = self.eba_3(vit_x3, cnn_x3)
        fused_x4 = self.eba_4(vit_x4, cnn_x4)

        x_dec = self.gltb4(fused_x4)
        x_dec = self.wf1(x_dec, fused_x3)
        x_dec = self.gltb3(x_dec)
        x_dec = self.wf2(x_dec, fused_x2)
        x_dec = self.gltb2(x_dec)
        x_dec = self.wf3(x_dec, fused_x1)
        visual_features = self.gltb1(x_dec)

        coarse_feat = self.head(visual_features)        # [B, num_classes, fH, fW]
        boundary_feat = self.boundary_head(visual_features)  # [B, 1, fH, fW]
        return visual_features, coarse_feat, boundary_feat

    def forward(self, x, mask_t=None, t=None):
        B, C, H, W = x.shape

        visual_features, coarse_feat, boundary_feat = self._extract_features(x)

        # Upsample predictions to original image resolution
        out = F.interpolate(coarse_feat, size=(H, W), mode='bilinear', align_corners=False)
        boundary_logits = F.interpolate(boundary_feat, size=(H, W), mode='bilinear', align_corners=False)

        if mask_t is not None and t is not None:
            image_latent = self.encode_image_latent(x)
            coarse_latent = self.encode_mask_latent(out.detach(), is_logits=True)

            mask_t_input = mask_t
            if mask_t_input.shape[1] != self.latent_channels:
                mask_t_input = self.encode_mask_latent(mask_t_input, is_logits=True)
            if mask_t_input.shape[2:] != image_latent.shape[2:]:
                mask_t_input = F.interpolate(mask_t_input, size=image_latent.shape[2:], mode='bilinear', align_corners=False)

            if coarse_latent.shape[2:] != mask_t_input.shape[2:]:
                coarse_latent = F.interpolate(coarse_latent, size=mask_t_input.shape[2:], mode='bilinear', align_corners=False)
            if image_latent.shape[2:] != mask_t_input.shape[2:]:
                image_latent = F.interpolate(image_latent, size=mask_t_input.shape[2:], mode='bilinear', align_corners=False)
            boundary_cond = F.interpolate(boundary_logits, size=mask_t_input.shape[2:], mode='bilinear', align_corners=False)

            noise_pred = self.diffusion_head(mask_t_input, t, image_latent, coarse_latent, boundary_cond)
            return out, boundary_logits, noise_pred

        return out, boundary_logits

    @torch.no_grad()
    def ddpm_sample(self, x, num_steps=20, T=1000, betas=None, eta: float = 0.0):
        """
        DDIM reverse sampling for inference (eta=0 for deterministic).
        Uses latent diffusion conditioned on image latent, coarse mask latent, and boundary logits.

        Returns
        -------
        refined : torch.Tensor  [B, num_classes, H, W]
            Refined logits (un-normalised) at the original image resolution.
        """
        device = x.device
        B, _, H, W = x.shape

        if betas is None:
            betas = torch.linspace(1e-4, 2e-2, T, device=device)
        else:
            betas = betas.to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # ── Step 1: coarse prediction + latent conditions ────────────────
        visual_features, coarse_feat, boundary_feat = self._extract_features(x)
        coarse_logits = F.interpolate(coarse_feat, size=(H, W), mode='bilinear', align_corners=False)
        boundary_logits = F.interpolate(boundary_feat, size=(H, W), mode='bilinear', align_corners=False)

        image_latent = self.encode_image_latent(x)
        coarse_latent = self.encode_mask_latent(coarse_logits, is_logits=True)
        latent_h, latent_w = coarse_latent.shape[2:]
        if image_latent.shape[2:] != (latent_h, latent_w):
            image_latent = F.interpolate(image_latent, size=(latent_h, latent_w), mode='bilinear', align_corners=False)
        boundary_cond = F.interpolate(boundary_logits, size=(latent_h, latent_w), mode='bilinear', align_corners=False)

        # ── Step 2: initialise from pure noise in latent space ────────────
        xt = torch.randn_like(coarse_latent)

        # ── Step 3: uniformly spaced reverse timesteps ────────────────────
        step_indices = torch.linspace(T - 1, 0, num_steps, dtype=torch.long, device=device)
        step_indices = [int(s.item()) for s in step_indices]

        for idx, step in enumerate(step_indices):
            t_tensor = torch.full((B,), step, device=device, dtype=torch.float32)
            alpha_t = alphas_cumprod[step]
            prev_step = step_indices[idx + 1] if idx + 1 < len(step_indices) else -1
            alpha_prev = alphas_cumprod[prev_step] if prev_step >= 0 else torch.ones(1, device=device)

            noise_pred = self.diffusion_head(xt, t_tensor, image_latent, coarse_latent, boundary_cond)

            x0_pred = (xt - (1.0 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt().clamp(min=1e-8)
            x0_pred = x0_pred.clamp(-1.0, 1.0)

            sigma = eta * ((1 - alpha_prev) / (1 - alpha_t)).sqrt() * (1 - alpha_t / alpha_prev).sqrt()
            dir_coeff = (1 - alpha_prev - sigma ** 2).clamp(min=0).sqrt()
            dir_xt = dir_coeff * noise_pred
            xt = alpha_prev.sqrt() * x0_pred + dir_xt
            if prev_step >= 0 and eta > 0:
                xt = xt + sigma * torch.randn_like(xt)

        # ── Step 4: decode latent mask and upsample to original resolution ─
        refined_logits = self.decode_mask_latent(xt)
        return F.interpolate(refined_logits, size=(H, W), mode='bilinear', align_corners=False)

    

    
# if __name__ == "__main__":
#     model = SAM3UNet().cuda().eval()
#     with torch.no_grad():
#         x = torch.randn(1, 3, 336, 336).cuda()
#         out = model(x)
#         print(out.shape)
