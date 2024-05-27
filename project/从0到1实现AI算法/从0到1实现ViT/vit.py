import torch
from torch import nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768, img_size=224):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)  # (B, emb_size, H/P, W/P)
        x = x.flatten(2)  # (B, emb_size, N)
        x = x.transpose(1, 2)  # (B, N, emb_size)
        return x

class ViT(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768, img_size=224, num_classes=1000, depth=12, n_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positional_embedding = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2 + 1, emb_size))
        self.dropout = nn.Dropout(dropout)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=emb_size, nhead=n_heads, dim_feedforward=int(mlp_ratio * emb_size), dropout=dropout, batch_first=True),
            num_layers=depth
        )
        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, num_classes)
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.positional_embedding
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.to_cls_token(x[:, 0, :])   # 提取 class token，这是一个特殊的token，用于代表整个图像的特征。
        x = self.mlp_head(x)
        return x

# 示例使用
if __name__ == "__main__":
    # 假设我们有一个3通道的224x224大小的图像
    img = torch.randn(5, 1, 28, 28)

    # 初始化ViT模型
    model = ViT(in_channels=1, patch_size=4, emb_size=64, img_size=28, num_classes=10, depth=3, n_heads=2, mlp_ratio=4.0, dropout=0.1)
    print(model)

    # 前向传播
    logits = model(img)
    print(logits.shape)