import functools

import torch.nn as nn
import torch.nn.functional as F
import torch


def weights_init_normal(m):
    # classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)


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
                nn.Conv2d(in_features, mid_channels, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.InstanceNorm2d(mid_channels),
                nn.ReLU(),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU()
            )
        elif hin:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_features, mid_channels, kernel_size=3, padding=1, padding_mode='reflect'),
                Half_IN(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
                Half_IN(mid_channels),
                nn.ReLU(inplace=True)
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


##############################
#           Dense Block
##############################
class Den_block(nn.Module):
    def __init__(self, in_features, out_channels):
        super(Den_block, self).__init__()
        self.layer = nn.Sequential(
            # 把这块的IN改成HIN试试
            # nn.InstanceNorm2d(in_features),
            Half_IN(in_features),
            nn.LeakyReLU(),
            nn.Conv2d(in_features, out_channels, 3, 1, 1, bias=True, padding_mode='reflect')
        )

    def forward(self, x):
        return self.layer(x)


##############################
#           Dense Residual Block
##############################
class DenseResidualBlock(nn.Module):
    """
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, filters):
        super(DenseResidualBlock, self).__init__()
        # self.res_scale = res_scale

        self.b1 = Den_block(1 * filters, filters)
        self.b2 = Den_block(2 * filters, filters)
        self.b3 = Den_block(3 * filters, filters)
        self.b4 = Den_block(4 * filters, filters)
        # self.b4 = Den_block(5 * filters, filters)
        # self.blocks = [self.b1, self.b2, self.b3, self.b4]
        # self.b1 = DoubleConv(1 * filters, filters, hin=True, res=True)
        # self.b2 = DoubleConv(2 * filters, filters, hin=True, res=True)
        # self.b3 = DoubleConv(3 * filters, filters, hin=True, res=True)
        # self.b4 = DoubleConv(4 * filters, filters, hin=True, res=True)
        # self.b1 = DoubleConv(1 * filters, filters, hin=True)
        # self.b2 = DoubleConv(2 * filters, filters, hin=True)
        # self.b3 = DoubleConv(3 * filters, filters, hin=True)
        # self.b4 = DoubleConv(4 * filters, filters, hin=True)
        self.tran = nn.Sequential(
            nn.InstanceNorm2d(5 * filters),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=5 * filters, out_channels=filters, kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        inputs = x
        out = self.b1(inputs)
        inputs = torch.cat([inputs, out], 1)

        out = self.b2(inputs)
        inputs = torch.cat([inputs, out], 1)

        out = self.b3(inputs)
        inputs = torch.cat([inputs, out], 1)

        out = self.b4(inputs)
        inputs = torch.cat([inputs, out], 1)

        outputs = self.tran(inputs)
        return outputs + x

##############################
#           Upscaling the double conv
##############################
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


##############################
#           Down sampling
##############################
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            DoubleConv(in_channels, res=True, bn=True)
        )
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        x_down = self.down(x)
        return x_down


##############################
#           Generator ResNet 不同
##############################
class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_den_blocks=9):
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0]
        model_input = []
        model_blocks = []
        # Initial convolution block
        out_features = 64
        model_input += [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
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
            # model_blocks += [DoubleConv(out_features, hin=True, res=True)]
            model_blocks += [DenseResidualBlock(out_features)]

        # Upsampling
        in_features = out_features
        out_features //= 2
        self.model_up1 = Up(in_features, out_features)
        in_features = out_features
        out_features //= 2
        self.model_up2 = Up(in_features, out_features)
        # Output layer
        self.model_out = nn.Sequential(
            nn.ReflectionPad2d(1),  # 对输入图像以最外围像素为对称轴，做四周的轴对称镜像填充
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


##############################
#           Discriminator
##############################
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)

if __name__ == '__main__':
    input=torch.randn(50,3,224,224)
    resnet50=GeneratorResNet(1000)
    # resnet101=ResNet101(1000)
    # resnet152=ResNet152(1000)
    out=resnet50(input)
    print(out.shape)
