import torch
from torch import nn

# 假想一张 1 通道、8x8 的小图
image = torch.randn(1, 1, 8, 8)  # (batch, channel, H, W)
print("输入图像形状:", image.shape)

# 一个卷积层：1个输入通道 → 3个输出通道，3x3 卷积核
conv = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1)
output = conv(image)

print("卷积后形状:", output.shape)
print()
print("解读：")
print("- 输入: 1 张图，1 个通道，8x8 像素")
print("- 3x3 卷积核在图上滑动，提取局部特征")
print("- 输出: 3 张特征图（3 个通道），尺寸仍约 8x8（因为 padding=1）")
print("- 后面还可接池化、更多卷积、全连接，做成分类网络")
