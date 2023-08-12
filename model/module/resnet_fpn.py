import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.common_utils import torch_nms


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                conv1x1(in_planes, planes, stride=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        y = x
        y = self.relu(self.bn1(self.conv1(y)))
        y = self.bn2(self.conv2(y))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = nn.Conv2d(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        # with warnings.catch_warnings():
        #     warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))
class PPM(nn.Module): # pspnet
    def __init__(self, dim0, down_dim):
        super(PPM, self).__init__()
        self.down_conv = nn.Sequential(nn.Conv2d(dim0, down_dim, 3,padding=1),nn.BatchNorm2d(down_dim),
             nn.PReLU())

        self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)),nn.Conv2d(down_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU())
        self.conv2 = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(2, 2)), nn.Conv2d(down_dim, down_dim, kernel_size=1),
            nn.BatchNorm2d(down_dim), nn.PReLU())
        self.conv3 = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(3, 3)),nn.Conv2d(down_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU())
        self.conv4 = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(6, 6)), nn.Conv2d(down_dim, down_dim, kernel_size=1),
            nn.BatchNorm2d(down_dim), nn.PReLU())

        self.fuse = nn.Sequential(nn.Conv2d(4 * down_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU())

    def forward(self, x):
        x = self.down_conv(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        conv1_up = F.upsample(conv1, size=x.size()[2:], mode='bilinear', align_corners=True)
        conv2_up = F.upsample(conv2, size=x.size()[2:], mode='bilinear', align_corners=True)
        conv3_up = F.upsample(conv3, size=x.size()[2:], mode='bilinear', align_corners=True)
        conv4_up = F.upsample(conv4, size=x.size()[2:], mode='bilinear', align_corners=True)

        return self.fuse(torch.cat((conv1_up, conv2_up, conv3_up, conv4_up), 1))


class ResNet(nn.Module):
    """
    ResNet+FPN, output resolution are 1/8 and 1.
    Each block has 2 layers.
    """

    def __init__(self, config):
        super().__init__()
        # Config
        block = BasicBlock
        initial_dim = config['initial_dim']
        block_dims = config['block_dims']
        # block_dims = [128, 256, 256]  # [a/4, a/2, a]
        # block_dimsD = [64, 128, 256]  # [a/4, a/2, a]

        # Class Variable
        self.in_planes = initial_dim

        # Networks
        self.conv1 = nn.Conv2d(1, initial_dim, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_dim)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, block_dims[0], stride=1)  # 1/2
        self.layer2 = self._make_layer(block, block_dims[1], stride=2)  # 1/4
        self.layer3 = self._make_layer(block, block_dims[2], stride=2)  # 1/8

        self.ppm1 = SPPF(initial_dim, initial_dim)
        self.ppm2 = SPPF(block_dims[0], block_dims[0])
        self.ppm3 = SPPF(block_dims[1], block_dims[1])
        # 3. FPN upsample
        # self.layer3_outconv_semi = conv1x1(block_dims[2], block_dims[2])

        self.layer3_outconv = conv1x1(block_dims[2], block_dims[2])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, dim, stride=1):
        layer1 = block(self.in_planes, dim, stride=stride)
        layer2 = block(dim, dim, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def run_block(self, x):
        x0 = self.relu(self.bn1(self.conv1(x)))
        x0 = self.ppm1(x0)
        x1 = self.layer1(x0)  # 1/2
        x1 = self.ppm2(x1)
        x2 = self.layer2(x1)
        x2 = self.ppm3(x2)
        x3 = self.layer3(x2)
        return x1, x2, x3

    def forward(self, x):
        # ResNet Backbone
        x1, x2, x3 = self.run_block(x)
        # x2 = self.layer2(x1)
        # x3 = self.layer3(x2)
        # descriptor_out_semi = self.layer3_outconv_semi(x3)
        # FPN
        x3_out = self.layer3_outconv(x3)

        return x3_out


class ResNetFPN(nn.Module):
    """
    ResNet+FPN, output resolution are 1/8 and 1.
    Each block has 2 layers.
    """

    def __init__(self, config):
        super().__init__()
        # Config
        block = BasicBlock
        initial_dim = config['initial_dim']
        block_dims = config['block_dims']
        # block_dims = [128, 256, 256]  # [a/4, a/2, a]
        # block_dimsD = [64, 128, 256]  # [a/4, a/2, a]

        # Class Variable
        self.in_planes = initial_dim

        # Networks
        self.conv1 = nn.Conv2d(1, initial_dim, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_dim)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, block_dims[0], stride=2)  # 1/2
        self.layer2 = self._make_layer(block, block_dims[1], stride=2)  # 1/4
        self.layer3 = self._make_layer(block, block_dims[2], stride=2)  # 1/8

        # self.ppm1 = SPPF(initial_dim, initial_dim)
        # self.ppm2 = SPPF(block_dims[0], block_dims[0])
        # self.ppm3 = SPPF(block_dims[1], block_dims[1])
        # 3. FPN upsample
        # self.layer3_outconv_semi = conv1x1(block_dims[2], block_dims[2])
        # self.layer3_outconv_down = nn.Sequential(
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     conv1x1(block_dims[2], block_dims[2])
        # )
        self.layer3_outconv = conv1x1(block_dims[2], block_dims[2])
        self.layer2_outconv = conv1x1(block_dims[1], block_dims[2])
        self.layer2_outconv2 = nn.Sequential(
            conv3x3(block_dims[2], block_dims[2]),
            nn.BatchNorm2d(block_dims[2]),
            nn.LeakyReLU(),
            conv3x3(block_dims[2], block_dims[1]),
        )
        self.layer1_outconv = conv1x1(block_dims[0], block_dims[1])
        self.layer1_outconv2 = nn.Sequential(
            conv3x3(block_dims[1], block_dims[1]),
            nn.BatchNorm2d(block_dims[1]),
            nn.LeakyReLU(),
            conv3x3(block_dims[1], block_dims[0]),
        )

        # self.layer0_outconv = conv1x1(block_dims[0], block_dims[1])
        # self.layer0_outconv2 = nn.Sequential(
        #     conv3x3(block_dims[1], block_dims[1]),
        #     nn.BatchNorm2d(block_dims[1]),
        #     nn.LeakyReLU(),
        #     conv3x3(block_dims[1], block_dims[0]),
        # )
        #
        # self.layer0_outconv = conv1x1(block_dims[0], block_dims[0])

        # 4. FPN upsample Det
        # self.dconv_up3 = double_conv(block_dims[2], block_dims[1])
        # self.dconv_up2 = double_conv(block_dims[1] + block_dims[1], block_dims[0])
        # self.dconv_up1 = double_conv(block_dims[0] + initial_dim, initial_dim)

        # self.layer3_outconvD = conv1x1(block_dims[2], block_dims[2])
        # self.layer2_outconvD = conv1x1(block_dims[1], block_dims[2])
        # self.layer2_outconvD2 = nn.Sequential(
        #     conv3x3(block_dims[2], block_dims[2]),
        #     nn.BatchNorm2d(block_dims[2]),
        #     nn.LeakyReLU(),
        #     conv3x3(block_dims[2], block_dims[1]),
        # )
        # self.layer1_outconvD = conv1x1(block_dims[0], block_dims[1])
        # self.layer1_outconvD2 = nn.Sequential(
        #     conv3x3(block_dims[1], block_dims[1]),
        #     nn.BatchNorm2d(block_dims[1]),
        #     nn.LeakyReLU(),
        #     conv3x3(block_dims[1], block_dims[0]),
        # )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, dim, stride=1):
        layer1 = block(self.in_planes, dim, stride=stride)
        layer2 = block(dim, dim, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def run_block(self, x):
        x0 = self.relu(self.bn1(self.conv1(x)))
        # x0 = self.ppm1(x0)
        x1 = self.layer1(x0)  # 1/2
        # x1 = self.ppm2(x1)
        x2 = self.layer2(x1)
        # x2 = self.ppm3(x2)
        x3 = self.layer3(x2)
        return x0, x1, x2, x3

    def forward(self, x):
        # ResNet Backbone
        x0, x1, x2, x3 = self.run_block(x)
        # x2 = self.layer2(x1)
        # x3 = self.layer3(x2)
        # descriptor_out_semi = self.layer3_outconv_semi(x3)
        # FPN
        x3_out = self.layer3_outconv(x3)

        x3_out_2x = F.interpolate(x3_out, scale_factor=2., mode='bilinear', align_corners=True)
        x2_out = self.layer2_outconv(x2)
        x2_out = self.layer2_outconv2(x2_out + x3_out_2x)

        x2_out_2x = F.interpolate(x2_out, scale_factor=2., mode='bilinear', align_corners=True)
        x1_out = self.layer1_outconv(x1)
        x1_out = self.layer1_outconv2(x1_out + x2_out_2x)

        # x1_out_2x = F.interpolate(x1_out, scale_factor=2., mode='bilinear', align_corners=True)
        # x0_out = self.layer0_outconv(x0)
        #
        # x0_out = self.layer0_outconv2(x0_out + x1_out_2x)

        # x1_out_2x = F.interpolate(x1_out, scale_factor=2., mode='bilinear', align_corners=True)

        return x3_out, x1_out


if __name__ == '__main__':
    config = {'initial_dim': 128, 'block_dims':[128, 256, 512]}
    R = ResNetFPN(config)
    R = R.cuda()
    a, b = R(torch.rand(3, 1, 512, 512).cuda())
    b = torch.ones([1, 1, 512, 512])
    b[0][0][10][:] = 2
    res = torch_nms(b, 10)
    t = 1