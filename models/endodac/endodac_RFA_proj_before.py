import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.backbones as backbones
from models.backbones.mylora import Linear as LoraLinear
from models.backbones.mylora import DVLinear as DVLinear
from .layers import HeadDepth
from .layers import mark_only_part_as_trainable,_make_scratch, _make_fusion_block

# ===================== RFA / ConvMod block (can be deleted as a whole) =====================
class LayerNorm(nn.Module):
    r""" 支持两种数据格式的LayerNorm层：
    - channels_last：输入形状为(batch_size, height, width, channels)（通道在后）
    - channels_first：输入形状为(batch_size, channels, height, width)（通道在前，默认）
    功能：对特征图进行层归一化，加速训练收敛，抑制梯度爆炸
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        # 可学习的缩放参数（初始为1），形状与归一化维度一致
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        # 可学习的偏移参数（初始为0），形状与归一化维度一致
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        # 数值稳定性参数，避免分母为0
        self.eps = eps
        # 输入数据的通道格式
        self.data_format = data_format
        # 检查数据格式是否合法，仅支持channels_last和channels_first
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError("仅支持channels_last和channels_first格式")
        # 保存归一化的形状（通道数）
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        # 若为channels_last格式，直接调用PyTorch内置的layer_norm
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # 若为channels_first格式，手动实现LayerNorm（因PyTorch内置函数默认适配channels_last）
        elif self.data_format == "channels_first":
            # 1. 计算通道维度（dim=1）的均值u，keepdim=True保持维度不变（便于后续广播）
            u = x.mean(1, keepdim=True)
            # 2. 计算通道维度的方差s（均方误差）
            s = (x - u).pow(2).mean(1, keepdim=True)
            # 3. 归一化：(x - 均值) / sqrt(方差 + eps)
            x = (x - u) / torch.sqrt(s + self.eps)
            # 4. 应用可学习参数：缩放（weight）+ 偏移（bias），需扩展维度以匹配特征图形状
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


# 定义卷积模块ConvMod，继承自nn.Module（PyTorch中所有神经网络模块的基类）
class ConvMod(nn.Module):
    # 构造函数，接收参数dim（输入特征图的通道数）
    def __init__(self, dim):
        # 调用父类nn.Module的构造函数，初始化模块
        super().__init__()

        # -------------------------- 第一层感受野聚合（对应结构图H₁部分） --------------------------
        # 1. 归一化层：对输入进行LayerNorm，通道格式为channels_first（[batch, channels, H, W]）
        self.norm1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        # 2. 注意力/特征变换分支A₁：1×1卷积（降维+升维）→ GELU激活 → 7×7深度可分离卷积（扩大感受野）
        #    输入输出通道数均为dim//4（对应特征拆分后的单路通道数）
        self.a1 = nn.Sequential(
            nn.Conv2d(dim // 4, dim // 4, 1),  # 1×1卷积：调整通道维度，保持特征图尺寸
            nn.GELU(),  # 激活函数：引入非线性，增强模型表达能力
            nn.Conv2d(dim // 4, dim // 4, 7, padding=3, groups=dim // 4),  # 7×7深度可分离卷积（DW Conv）
        )
        # 3. 价值/特征映射分支V₁：1×1卷积，对拆分后的特征图进行通道变换
        self.v1 = nn.Conv2d(dim // 4, dim // 4, 1)
        # 4. 特征变换层V₁₁：1×1卷积，对A₁与V₁的乘积结果进一步变换
        self.v11 = nn.Conv2d(dim // 4, dim // 4, 1)
        # 5. 特征变换层V₁₂：1×1卷积，处理另一路拆分后的特征
        self.v12 = nn.Conv2d(dim // 4, dim // 4, 1)
        # 6. 3×3深度可分离卷积：对V₁₂处理后的特征进行局部特征提取
        self.conv3_1 = nn.Conv2d(dim // 4, dim // 4, 3, padding=1, groups=dim // 4)

        # -------------------------- 第二层感受野聚合（对应结构图H₂部分） --------------------------
        # 1. 归一化层：输入通道数为dim//2（第一层输出的两路特征拼接后通道数）
        self.norm2 = LayerNorm(dim // 2, eps=1e-6, data_format="channels_first")
        # 2. 注意力/特征变换分支A₂：1×1卷积 → GELU → 9×9深度可分离卷积（感受野扩大至9×9）
        self.a2 = nn.Sequential(
            nn.Conv2d(dim // 2, dim // 2, 1),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim // 2, 9, padding=4, groups=dim // 2),
        )
        # 3. 价值/特征映射分支V₂：1×1卷积，处理第二层输入特征
        self.v2 = nn.Conv2d(dim // 2, dim // 2, 1)
        # 4. 特征变换层V₂₁：1×1卷积，处理A₂与V₂的乘积结果
        self.v21 = nn.Conv2d(dim // 2, dim // 2, 1)
        # 5. 特征变换层V₂₂：1×1卷积，处理第三路拆分后的原始特征
        self.v22 = nn.Conv2d(dim // 4, dim // 4, 1)
        # 6. 通道投影层proj₂：1×1卷积，将A₂的输出（dim//2通道）降维至dim//4，用于与另一路特征融合
        self.proj2 = nn.Conv2d(dim // 2, dim // 4, 1)
        # 7. 3×3深度可分离卷积：对V₂₂处理后的特征进行局部特征提取
        self.conv3_2 = nn.Conv2d(dim // 4, dim // 4, 3, padding=1, groups=dim // 4)

        # -------------------------- 第三层感受野聚合（对应结构图H₃部分） --------------------------
        # 1. 归一化层：输入通道数为3dim//4（第二层输出的两路特征拼接后通道数）
        self.norm3 = LayerNorm(dim * 3 // 4, eps=1e-6, data_format="channels_first")
        # 2. 注意力/特征变换分支A₃：1×1卷积 → GELU → 11×11深度可分离卷积（感受野扩大至11×11）
        self.a3 = nn.Sequential(
            nn.Conv2d(dim * 3 // 4, dim * 3 // 4, 1),
            nn.GELU(),
            nn.Conv2d(dim * 3 // 4, dim * 3 // 4, 11, padding=5, groups=dim * 3 // 4),
        )
        # 3. 价值/特征映射分支V₃：1×1卷积，处理第三层输入特征
        self.v3 = nn.Conv2d(dim * 3 // 4, dim * 3 // 4, 1)
        # 4. 特征变换层V₃₁：1×1卷积，处理A₃与V₃的乘积结果
        self.v31 = nn.Conv2d(dim * 3 // 4, dim * 3 // 4, 1)
        # 5. 特征变换层V₃₂：1×1卷积，处理第四路拆分后的原始特征
        self.v32 = nn.Conv2d(dim // 4, dim // 4, 1)
        # 6. 通道投影层proj₃：1×1卷积，将A₃的输出（3dim//4通道）降维至dim//4，用于与另一路特征融合
        self.proj3 = nn.Conv2d(dim * 3 // 4, dim // 4, 1)
        # 7. 3×3深度可分离卷积：对V₃₂处理后的特征进行局部特征提取
        self.conv3_3 = nn.Conv2d(dim // 4, dim // 4, 3, padding=1, groups=dim // 4)

        # 保存输入通道数dim，用于后续特征拆分
        self.dim = dim

    # 前向传播函数，定义特征流动逻辑（核心！对应结构图的感受野流）
    def forward(self, x):
        if self.dim % 4 != 0:
            raise ValueError(f"ConvMod expects channels divisible by 4, got {self.dim}.")
        # -------------------------- 第一层特征处理（对应H₁） --------------------------
        # 1. 对输入x进行第一层归一化
        x = self.norm1(x)
        # 2. 按通道维度拆分x为4路，每路通道数为dim//4（例如dim=36时，每路9通道）
        x_split = torch.split(x, self.dim // 4, dim=1)  # x_split[0]~x_split[3]对应4路特征

        # 3. 处理A₁分支：输入为x_split[0]（第一路特征），通过A₁得到7×7感受野特征
        a = self.a1(x_split[0])
        # 4. 注意力机制：A₁输出与V₁处理后的x_split[0]逐元素相乘（模拟注意力权重）
        mul = a * self.v1(x_split[0])
        # 5. 对注意力结果进行通道变换
        mul = self.v11(mul)

        # 6. 处理另一路特征：x_split[1]（第二路）经V₁₂变换后，通过3×3深度可分离卷积提取局部特征
        x1 = self.conv3_1(self.v12(x_split[1]))
        # 7. 特征融合：将局部特征与A₁的输出a相加（残差连接思想，保留原始感受野特征）
        x1 = x1 + a
        # 8. 拼接特征：将融合后的局部特征x1与注意力结果mul拼接，通道数变为dim//2（两路各dim//4）
        x1 = torch.cat((x1, mul), dim=1)

        # -------------------------- 第二层特征处理（对应H₂） --------------------------
        # 1. 对第一层输出x1进行归一化
        x1 = self.norm2(x1)
        # 2. 处理A₂分支：输入为x1，通过A₂得到9×9感受野特征
        a = self.a2(x1)
        # 3. 注意力机制：A₂输出与V₂处理后的x1逐元素相乘
        mul = a * self.v2(x1)
        # 4. 对注意力结果进行通道变换
        mul = self.v21(mul)

        # 5. 处理第三路原始特征：x_split[2]（第三路）经V₂₂变换后，通过3×3深度可分离卷积提取局部特征
        x2 = self.conv3_2(self.v22(x_split[2]))
        # 6. 特征融合：局部特征与A₂输出的投影结果（proj₂降维至dim//4）相加
        x2 = x2 + self.proj2(a)
        # 7. 拼接特征：将融合后的x2与注意力结果mul拼接，通道数变为3dim//4（两路各dim//2）
        x2 = torch.cat((x2, mul), dim=1)

        # -------------------------- 第三层特征处理（对应H₃） --------------------------
        # 1. 对第二层输出x2进行归一化
        x2 = self.norm3(x2)
        # 2. 处理A₃分支：输入为x2，通过A₃得到11×11感受野特征
        a = self.a3(x2)
        # 3. 注意力机制：A₃输出与V₃处理后的x2逐元素相乘
        mul = a * self.v3(x2)
        # 4. 对注意力结果进行通道变换
        mul = self.v31(mul)

        # 5. 处理第四路原始特征：x_split[3]（第四路）经V₃₂变换后，通过3×3深度可分离卷积提取局部特征
        x3 = self.conv3_3(self.v32(x_split[3]))
        # 6. 特征融合：局部特征与A₃输出的投影结果（proj₃降维至dim//4）相加
        x3 = x3 + self.proj3(a)
        # 7. 最终拼接：将融合后的x3与注意力结果mul拼接，通道数恢复为dim（两路各dim//2）
        x = torch.cat((x3, mul), dim=1)

        # 返回最终输出特征图（尺寸与输入相同，通道数保持dim）
        return x


class _RFAProjects(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.rfa = ConvMod(in_channels)
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x):
        x = self.rfa(x)
        return self.proj(x)

# =================== end of RFA / ConvMod block (can be deleted as a whole) ===================

class DPTHead(nn.Module):
    def __init__(self, in_channels, features=128, use_bn=False, out_channels=[96, 192, 384, 768], use_clstoken=False):
        super(DPTHead, self).__init__()

        self.use_clstoken = use_clstoken
        
        self.projects = nn.ModuleList([
            _RFAProjects(in_channels=in_channels, out_channels=out_channel) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.conv_depth_1 = HeadDepth(features)
        self.conv_depth_2 = HeadDepth(features)
        self.conv_depth_3 = HeadDepth(features)
        self.conv_depth_4 = HeadDepth(features)
        
        self.sigmoid = nn.Sigmoid()
    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        outputs = {}
        outputs[("disp", 3)] = self.sigmoid(self.conv_depth_4(path_4))
        outputs[("disp", 2)] = self.sigmoid(self.conv_depth_3(path_3))
        outputs[("disp", 1)] = self.sigmoid(self.conv_depth_2(path_2))
        outputs[("disp", 0)] = self.sigmoid(self.conv_depth_1(path_1))

        return outputs
    
class endodac(nn.Module):
    """Applies low-rank adaptation to a ViT model's image encoder.

    Args:
        backbone_size: size of pretrained Dinov2 choice from: "small", "base", "large", "giant"
        r: rank of LoRA
        image_shape: input image shape, h,w need to be multiplier of 14, default:(224,280)
        lora_layer: which layer we apply LoRA.
    """

    def __init__(self, 
                 backbone_size = "base", 
                 r=4, 
                 image_shape=(224,280), 
                 lora_type="lora",
                 pretrained_path=None,
                 residual_block_indexes=[],
                 include_cls_token=True,
                 use_cls_token=False,
                 use_bn=False,
                 depth_backbone_ckpt=None):
        super(endodac, self).__init__()

        assert r > 0
        self.r = r
        self.backbone_size = backbone_size
        self.backbone = {
            "small": backbones.vits.vit_small(residual_block_indexes=residual_block_indexes,
                                              include_cls_token=include_cls_token),
            "base": backbones.vits.vit_base(residual_block_indexes=residual_block_indexes,
                                            include_cls_token=include_cls_token),
        }
        self.backbone_archs = {
            "small": "vits14",
            "base": "vitb14",
        }
        self.intermediate_layers = {
            "small": [2, 5, 8, 11],
            "base": [2, 5, 8, 11],
        }
        self.embedding_dims = {
            "small": 384,
            "base": 768,
        }
        self.depth_head_features = {
            "small": 64,
            "base": 128,
        }
        self.depth_head_out_channels = {
            "small": [48, 96, 192, 384],
            "base": [96, 192, 384, 768],
        }
        self.backbone_arch = self.backbone_archs[self.backbone_size]
        self.embedding_dim = self.embedding_dims[self.backbone_size]
        self.depth_head_feature = self.depth_head_features[self.backbone_size]
        self.depth_head_out_channel = self.depth_head_out_channels[self.backbone_size]
        encoder = self.backbone[self.backbone_size]

        self.image_shape = image_shape
        
        if lora_type != "none":
            for t_layer_i, blk in enumerate(encoder.blocks):
                mlp_in_features = blk.mlp.fc1.in_features
                mlp_hidden_features = blk.mlp.fc1.out_features
                mlp_out_features = blk.mlp.fc2.out_features
                if lora_type == "dvlora":
                    blk.mlp.fc1 = DVLinear(mlp_in_features, mlp_hidden_features, r=self.r, lora_alpha=self.r)
                    blk.mlp.fc2 = DVLinear(mlp_hidden_features, mlp_out_features, r=self.r, lora_alpha=self.r)
                elif lora_type == "lora":
                    blk.mlp.fc1 = LoraLinear(mlp_in_features, mlp_hidden_features, r=self.r)
                    blk.mlp.fc2 = LoraLinear(mlp_hidden_features, mlp_out_features, r=self.r)
            
        self.encoder = encoder
        self.depth_head = DPTHead(self.embedding_dim, self.depth_head_feature, use_bn, out_channels=self.depth_head_out_channel, use_clstoken=use_cls_token)
        
        if pretrained_path is not None:
            if depth_backbone_ckpt is not None:
                ckpt_name = depth_backbone_ckpt
            else:
                ckpt_name = "depth_anything_{}.pth".format(self.backbone_arch)
            pretrained_path = os.path.join(pretrained_path, ckpt_name)
            pretrained_dict = torch.load(pretrained_path)
            model_dict = self.state_dict()
            self.load_state_dict(pretrained_dict, strict=False)
            print("load pretrained weight from {}\n".format(pretrained_path))

        mark_only_part_as_trainable(self.encoder)
        mark_only_part_as_trainable(self.depth_head)
    def forward(self, pixel_values):
        pixel_values = torch.nn.functional.interpolate(pixel_values, size=self.image_shape, mode="bilinear", align_corners=True)
        h, w = pixel_values.shape[-2:]
        
        features = self.encoder.get_intermediate_layers(pixel_values, 4, return_class_token=True)
        patch_h, patch_w = h // 14, w // 14

        disp = self.depth_head(features, patch_h, patch_w)

        return disp

