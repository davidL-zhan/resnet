from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    ViT 的图片切块层。

    作用：
        1. 把图片按 patch_size 切成不重叠小块。
        2. 把每个 patch 投影成一个 token 向量。

    形状变化：
        输入图片: [B, C, H, W]
        卷积投影: [B, embed_dim, H / patch_h, W / patch_w]
        token 序列: [B, num_patches, embed_dim]

    例如 image_size=224, patch_size=16:
        grid_size = 14 x 14
        num_patches = 196
        输入 [B, 3, 224, 224] -> 输出 [B, 196, embed_dim]
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()

        # image_size 和 patch_size 都统一保存成 `(height, width)`。
        self.image_size = [image_size, image_size]
        self.patch_size = [patch_size, patch_size]

        # ViT 这里使用不重叠 patch，因此图片高宽必须能被 patch 高宽整除。
        if self.image_size[0] % self.patch_size[0] != 0:
            raise ValueError("image height must be divisible by patch height.")
        if self.image_size[1] % self.patch_size[1] != 0:
            raise ValueError("image width must be divisible by patch width.")

        # grid_size 是 patch 网格尺寸；224/16 时就是 14 x 14。
        self.grid_size = (
            self.image_size[0] // self.patch_size[0],
            self.image_size[1] // self.patch_size[1],
        )
        # num_patches 是一张图片被切出的 patch 数；224/16 时是 196。
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # Conv2d 的 kernel_size=stride=patch_size，等价于切 patch 后做线性投影。
        # 输入:  [B, in_channels, H, W]
        # 输出:  [B, embed_dim, H / patch_h, W / patch_w]
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]，例如 [2, 3, 224, 224]。
        batch_size, channels, height, width = x.shape
        expected_height, expected_width = self.image_size

        # 当前实现使用固定输入尺寸；如果训练/预测预处理尺寸不一致，提前报错。
        if height != expected_height or width != expected_width:
            raise ValueError(
                f"Expected input image size {(expected_height, expected_width)}, "
                f"but got {(height, width)}."
            )

        # [B, C, H, W] -> [B, embed_dim, grid_h, grid_w]
        # 224x224 且 patch=16 时: [B, 3, 224, 224] -> [B, embed_dim, 14, 14]
        x = self.proj(x)

        # flatten(2): [B, embed_dim, grid_h, grid_w] -> [B, embed_dim, num_patches]
        # transpose:  [B, embed_dim, num_patches] -> [B, num_patches, embed_dim]
        x = x.flatten(2).transpose(1, 2)
        return x


class MLP(nn.Module):
    """
    Transformer block 里的前馈网络 FFN/MLP。

    形状不变：
        输入: [B, N, embed_dim]
        中间: [B, N, hidden_features]
        输出: [B, N, embed_dim]

    其中 N 是 token 数，ViT-224/16 中 N=197，包含 196 个 patch token
    和 1 个 cls token。
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # Linear 会作用在最后一维，因此 batch 维和 token 维都会保留。
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, in_features),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, embed_dim] -> [B, N, embed_dim]
        return self.net(x)


class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力

    输入输出形状：
        输入 x: [B, N, embed_dim]
        输出 x: [B, N, embed_dim]

    其中：
        B = batch size
        N = token 数，ViT-224/16 中是 197
        embed_dim = token 特征维度
        num_heads = 注意力头数量
        head_dim = embed_dim / num_heads

    内部形状流程：
        qkv 线性映射: [B, N, embed_dim] -> [B, N, 3 * embed_dim]
        拆成多头:     [B, N, 3 * embed_dim] -> [3, B, num_heads, N, head_dim]
        attention:    [B, num_heads, N, N]
        加权求和:      [B, num_heads, N, head_dim]
        合并多头:      [B, N, embed_dim]
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        # 一次 Linear 同时生成 Q、K、V。
        # [B, N, embed_dim] -> [B, N, 3 * embed_dim]
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)

        # attention_dropout 作用在 softmax 后的注意力权重上。
        self.attn_drop = nn.Dropout(dropout)

        # 多头结果合并后再做一次输出投影。
        # [B, N, embed_dim] -> [B, N, embed_dim]
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, embed_dim]
        batch_size, num_tokens, embed_dim = x.shape

        # qkv: [B, N, 3 * embed_dim]
        qkv = self.qkv(x)

        # [B, N, 3 * embed_dim]
        # -> [B, N, 3, num_heads, head_dim]
        qkv = qkv.reshape(
            batch_size,
            num_tokens,
            3,
            self.num_heads,
            self.head_dim,
        )

        # [B, N, 3, num_heads, head_dim]
        # -> [3, B, num_heads, N, head_dim]
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # q, k, v: [B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 注意力分数:
        # q: [B, num_heads, N, head_dim]
        # k.transpose(-2, -1): [B, num_heads, head_dim, N]
        # attn: [B, num_heads, N, N]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # 对最后一维做 softmax，表示每个 query token 对所有 key token 的权重。
        # attn: [B, num_heads, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 用注意力权重对 value 加权求和:
        # attn: [B, num_heads, N, N]
        # v:    [B, num_heads, N, head_dim]
        # x:    [B, num_heads, N, head_dim]
        x = attn @ v

        # 合并多头:
        # [B, num_heads, N, head_dim]
        # -> [B, N, num_heads, head_dim]
        # -> [B, N, embed_dim]
        x = x.transpose(1, 2).reshape(batch_size, num_tokens, embed_dim)

        # 输出投影，形状保持 [B, N, embed_dim]。
        x = self.proj(x)
        return x


class TransformerEncoderBlock(nn.Module):
    """
    ViT 使用的 Pre-LN Transformer Encoder Block。

    数据流：
        x -> LayerNorm -> Multi-Head Self-Attention -> residual add
          -> LayerNorm -> MLP -> residual add

    形状不变：
        输入: [B, N, embed_dim]
        输出: [B, N, embed_dim]

    其中：
        B = batch size
        N = token 数，ViT-224/16 中是 197
        embed_dim = 每个 token 的特征维度
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # 多头注意力要求 embed_dim 能被 num_heads 整除。
        # 每个 head 的维度是 embed_dim / num_heads。
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        # MLP 的隐藏层维度通常是 embed_dim 的 4 倍。
        hidden_features = int(embed_dim * mlp_ratio)

        # Pre-LN: 注意力前先做 LayerNorm，输入/输出都是 [B, N, embed_dim]。
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
        )
        self.drop1 = nn.Dropout(dropout)

        # 第二个 LayerNorm 和 MLP 也保持 [B, N, embed_dim] 形状。
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=hidden_features,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, embed_dim]
        shortcut = x

        # LayerNorm 不改变形状: [B, N, embed_dim] -> [B, N, embed_dim]
        x_norm = self.norm1(x)

        # 自注意力的 query/key/value 都来自同一个 x_norm。
        # attn_out: [B, N, embed_dim]
        attn_out = self.attn(x_norm)

        # 第一次残差连接: [B, N, embed_dim] + [B, N, embed_dim]
        x = shortcut + self.drop1(attn_out)

        # 第二次残差连接，MLP 输出形状仍是 [B, N, embed_dim]。
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    用于图片分类的 Vision Transformer。

    完整形状流程，以 image_size=224, patch_size=16 为例：
        输入图片:      [B, 3, 224, 224]
        patch embedding: [B, 196, embed_dim]
        加 cls token:   [B, 197, embed_dim]
        加位置编码:      [B, 197, embed_dim]
        Transformer:    [B, 197, embed_dim]
        取 cls token:   [B, embed_dim]
        分类头:         [B, num_classes]

    默认参数对应 ViT-Base/16 的输入几何：
        image_size = 224
        patch_size = 16
        num_patches = 14 * 14 = 196
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # 图片 -> patch tokens。
        # 输出形状: [B, num_patches, embed_dim]。
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # cls_token 是额外的分类 token，会拼到 patch token 序列最前面。
        # 参数形状: [1, 1, embed_dim]
        # forward 中 expand 后: [B, 1, embed_dim]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 位置编码覆盖 cls token + 所有 patch token。
        # 224/16 时形状是 [1, 197, embed_dim]。
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # depth 个 Transformer block，每个 block 都保持 [B, N, embed_dim]。
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
                for _ in range(depth)
            ]
        )
        # 最后的 LayerNorm 仍保持 [B, N, embed_dim]。
        self.norm = nn.LayerNorm(embed_dim)

        # 分类头只作用于 cls token: [B, embed_dim] -> [B, num_classes]。
        self.classifier = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        """初始化 ViT 参数，使用和常见 ViT 实现接近的初始化方式。"""

        # cls token 和位置编码是可学习参数，初始值很小。
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def encoder(self, x: torch.Tensor) -> torch.Tensor:
        # 输入图片: [B, C, H, W]，例如 [2, 3, 224, 224]。
        # patch tokens: [B, num_patches, embed_dim]，例如 [2, 196, 192]。
        x = self.patch_embed(x)

        # 把 [1, 1, embed_dim] 的 cls token 复制到 batch 维。
        # cls_token: [B, 1, embed_dim]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        # 拼接后 token 数 +1。
        # [B, 1, embed_dim] + [B, num_patches, embed_dim]
        # -> [B, num_patches + 1, embed_dim]
        x = torch.cat((cls_token, x), dim=1)

        # 加位置编码，形状不变: [B, num_patches + 1, embed_dim]。
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # 每个 Transformer block 都保持形状不变。
        for block in self.blocks:
            x = block(x)

        # 最终 token 序列: [B, num_patches + 1, embed_dim]。
        x = self.norm(x)

        # 只取第 0 个 cls token 做分类特征: [B, embed_dim]。
        return x[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # features: [B, embed_dim]
        x = self.encoder(x)

        # logits: [B, num_classes]
        x = self.classifier(x)
        return x


def vit_tiny_patch16_224(
    num_classes: int = 1000, in_channels: int = 3
) -> VisionTransformer:
    """ViT-Tiny/16，输入 224x224，输出形状 [B, num_classes]。"""

    return VisionTransformer(
        image_size=224,
        patch_size=16,
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim=192,
        depth=12,
        num_heads=3,
    )


def vit_small_patch16_224(
    num_classes: int = 1000, in_channels: int = 3
) -> VisionTransformer:
    """ViT-Small/16，输入 224x224，输出形状 [B, num_classes]。"""

    return VisionTransformer(
        image_size=224,
        patch_size=16,
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim=384,
        depth=12,
        num_heads=6,
    )


def vit_base_patch16_224(
    num_classes: int = 1000, in_channels: int = 3
) -> VisionTransformer:
    """ViT-Base/16，输入 224x224，输出形状 [B, num_classes]。"""

    return VisionTransformer(
        image_size=224,
        patch_size=16,
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim=768,
        depth=12,
        num_heads=12,
    )


def vit_tiny(num_classes: int = 1000, in_channels: int = 3) -> VisionTransformer:
    return vit_tiny_patch16_224(num_classes=num_classes, in_channels=in_channels)


def vit_small(num_classes: int = 1000, in_channels: int = 3) -> VisionTransformer:
    return vit_small_patch16_224(num_classes=num_classes, in_channels=in_channels)


def vit_base(num_classes: int = 1000, in_channels: int = 3) -> VisionTransformer:
    return vit_base_patch16_224(num_classes=num_classes, in_channels=in_channels)




if __name__ == "__main__":
    from torchinfo import summary

    # 简单前向测试：
    # 输入 x: [B, C, H, W] = [2, 3, 224, 224]
    # 输出 y: [B, num_classes] = [2, 101]
    model = vit_base(num_classes=101)
    x = torch.randn(2, 3, 224, 224)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary(model, input_data=x, device=device)
    # with torch.no_grad():
    #     y = model(x)

    total_params = sum(param.numel() for param in model.parameters())
    print("input shape:", x.shape)
    print("output shape:", y.shape)
    print("total params:", f"{total_params:,}")
