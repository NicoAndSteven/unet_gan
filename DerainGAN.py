import functools

import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from einops import rearrange, reduce


# 新添加的
class Conv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', functools.partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


# 权重初始化-未改动
def weights_init_normal(m):
    # classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)


# HIN层-未改动
class Half_IN(nn.Module):
    def __init__(self, feature):
        super(Half_IN, self).__init__()
        self.norm = nn.InstanceNorm2d(feature // 2, affine=True)

    def forward(self, x):
        out_1, out_2 = torch.chunk(x, 2, dim=1)
        out = torch.cat([self.norm(out_1), out_2], dim=1)
        return out


##############################
#           RESNET
##############################
class DoubleConv(nn.Module):
    def __init__(self, in_features, out_channels=None, mid_channels=None, res=False, hin=False, bn=False):
        super(DoubleConv, self).__init__()
        self.res = res
        if not out_channels:
            out_channels = in_features
        if not mid_channels:
            mid_channels = out_channels
        if self.res:
            assert in_features == out_channels
        if bn:
            self.double_conv = nn.Sequential(
                Conv2d(in_features, mid_channels, kernel_size=3, padding=1, padding_mode='reflect'),
                # ReLu=>SiLu，同时去掉IN层
                nn.SiLU(),
                Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
                # ReLu=>SiLu，同时去掉IN层
                nn.SiLU()
            )
        elif hin:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_features, mid_channels, kernel_size=3, padding=1, padding_mode='reflect'),
                Half_IN(mid_channels),
                # ReLu => GELU
                # nn.ReLU(inplace=True),
                nn.GELU(),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
                Half_IN(mid_channels),
                # nn.ReLU(inplace=True)
                nn.GELU(),
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_features, mid_channels, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        if self.res:
            return self.double_conv(x) + x
        else:
            return self.double_conv(x)


# 上采样未改动
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# 下采样--改动SiLu()
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            DoubleConv(in_channels)
        )
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            # LeakyReLu=>GiLU
            nn.SiLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        x_down = self.down(x)
        return x_down


# 生成器--改动
class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_den_blocks=9):
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0]
        model_input = []
        model_blocks = []
        # Initial convolution block
        out_features = 64
        model_input += [
            #             nn.ReflectionPad2d(channels),
            #             nn.Conv2d(channels, out_features, 7),
            #             nn.InstanceNorm2d(out_features),
            #             nn.ReLU(inplace=True),
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 3),
            nn.Conv2d(out_features, out_features, 3),
            Conv2d(out_features, out_features, 3),
            # nn.SiLU(), => GELU()
            nn.GELU(),
        ]
        in_features = out_features
        # Downsampling
        out_features *= 2
        self.model_down1 = Down(in_features, out_features)
        in_features = out_features
        out_features *= 2
        self.model_down2 = Down(in_features, out_features)

        # Residual blocks
        for _ in range(num_den_blocks):
            model_blocks += [DoubleConv(out_features, res=True, hin=True)]

        # Upsampling
        in_features = out_features
        out_features //= 2
        self.model_up1 = Up(in_features, out_features)
        in_features = out_features
        out_features //= 2
        self.model_up2 = Up(in_features, out_features)
        # Output layer
        self.model_out = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_features, channels, 3))

        self.model_input = nn.Sequential(*model_input)
        self.model_blocks = nn.Sequential(*model_blocks)

    def forward(self, x):
        x1 = self.model_input(x)
        x2 = self.model_down1(x1)
        x3 = self.model_down2(x2)
        x4 = self.model_blocks(x3)
        x_up1 = self.model_up1(x4, x2)
        x_up2 = self.model_up2(x_up1, x1)
        x_out = self.model_out(x_up2)
        return x_out


class SpConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, *args, **kwargs):
        super(SpConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        n, c, h, w = x.size()
        # assert (c % 4) == 0
        x1 = x[:, :c // 4, :, :]
        x2 = x[:, c // 4:c // 2, :, :]
        x3 = x[:, c // 2:c // 4 * 3, :, :]
        x4 = x[:, c // 4 * 3:c, :, :]
        x1 = nn.functional.pad(x1, (1, 0, 1, 0), mode="constant", value=0)  # left top
        x2 = nn.functional.pad(x2, (0, 1, 1, 0), mode="constant", value=0)  # right top
        x3 = nn.functional.pad(x3, (1, 0, 0, 1), mode="constant", value=0)  # left bottom
        x4 = nn.functional.pad(x4, (0, 1, 0, 1), mode="constant", value=0)  # right bottom
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""

            if normalize:
                layers = [SpConv2d(in_filters, out_filters, 4, stride=2, padding=1)]
                layers.append(nn.InstanceNorm2d(out_filters))
            else:
                layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]

            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            #             nn.ZeroPad2d((1, 0, 1, 0)),
            #             nn.Conv2d(512, 1, 4, padding=1),
            #             nn.Sigmoid()
            SpConv2d(512, 1, 4, 1, padding=1),
            nn.Softmax(dim=1)
        )

    def forward(self, img):
        return self.model(img)
