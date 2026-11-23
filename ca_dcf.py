import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ultralytics.nn.modules.conv import Conv  

class CoordinateAttention(nn.Module):
    def __init__(self, in_channels, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mid_channels = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.SiLU()
        self.conv_h = nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        identity = x
        b, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return identity * a_h * a_w

class AttentionGeneratorWithCA(nn.Module):
    def __init__(self, in_channels, num_bases):
        super(AttentionGeneratorWithCA, self).__init__()
        self.feature_enhancer = CoordinateAttention(in_channels)
        self.weight_generator = nn.Sequential(
            Conv(in_channels, in_channels // 4, 1),
            nn.Conv2d(in_channels // 4, num_bases, kernel_size=1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        enhanced_x = self.feature_enhancer(x)
        attention_weights = self.weight_generator(enhanced_x)
        return attention_weights

class CADCF(nn.Module):
    def __init__(self, c1, c2, k=3, num_bases=4, init_with_gabor=True, *args, **kwargs):
        super().__init__()
        self.c_in = c1
        self.c_out = c2
        self.k = k
        self.num_bases = num_bases
        self.padding = k // 2

        self.basis_filters = nn.Parameter(torch.randn(num_bases * c1, 1, k, k))
        nn.init.kaiming_normal_(self.basis_filters, mode='fan_out', nonlinearity='relu')
        if init_with_gabor and k % 2 == 1:
            self._initialize_with_gabor()

        self.attention_generator = AttentionGeneratorWithCA(c1, num_bases)
        self.output_conv = Conv(c1, c2, 1) if c1 != c2 else nn.Identity()

    def _initialize_with_gabor(self):
        for i in range(self.num_bases):
            theta = i * torch.pi / self.num_bases
            kernel = np.zeros((self.k, self.k), dtype=np.float32)
            for x_pos in range(self.k):
                for y_pos in range(self.k):
                    x = x_pos - self.padding
                    y = y_pos - self.padding
                    kernel[x_pos, y_pos] = np.exp(-0.5 * (x**2 + y**2) / (1.5**2)) * np.cos(
                        2 * np.pi / self.k * (x * np.cos(theta) + y * np.sin(theta))
                    )
            norm = np.sqrt(np.sum(kernel ** 2))
            kernel = kernel / norm if norm > 1e-6 else np.zeros_like(kernel)
            gabor_tensor = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0)
            self.basis_filters.data[i * self.c_in : (i + 1) * self.c_in] = gabor_tensor.repeat(self.c_in, 1, 1, 1)

    def forward(self, x):
        b, c, h, w = x.size()
        attention_weights = self.attention_generator(x)
        intermediate_outputs = F.conv2d(x, self.basis_filters, padding=self.padding, groups=self.c_in)
        intermediate_outputs = intermediate_outputs.view(b, self.num_bases, self.c_in, h, w)
        output = torch.sum(attention_weights.unsqueeze(2) * intermediate_outputs, dim=1)
        output = self.output_conv(output)
        return output
