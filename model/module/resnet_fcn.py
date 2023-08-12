import torch
from torch import nn
from torch.nn import init
from torchvision.models.resnet import BasicBlock, ResNet
import torch.nn.functional as F


# Returns 2D convolutional layer with space-preserving padding
def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, bias=False, transposed=False):
    if transposed:
        layer1 = nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=True)
        layer2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        layer = (layer1, layer2)
        layer = nn.Sequential(*layer)
        # layer = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=1,
        #                            output_padding=1, dilation=dilation, bias=bias)
        # Bilinear interpolation init 用双线性插值法初始化反卷积核
        # w = torch.Tensor(kernel_size, kernel_size)
        # centre = kernel_size % 2 == 1 and stride - 1 or stride - 0.5
        # for y in range(kernel_size):
        #     for x in range(kernel_size):
        #         w[y, x] = (1 - abs((x - centre) / stride)) * (1 - abs((y - centre) / stride))
        # layer.weight.data.copy_(w.div(in_planes).repeat(in_planes, out_planes, 1, 1))
    else:
        padding = (kernel_size + 2 * (dilation - 1)) // 2
        layer = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                          dilation=dilation, bias=bias)
    if bias:
        init.constant(layer.bias, 0)
    return layer


# Returns 2D batch normalisation layer
def bn(planes):
    layer = nn.BatchNorm2d(planes)
    # Use mean 0, standard deviation 1 init
    init.constant(layer.weight, 1)
    init.constant(layer.bias, 0)
    return layer


class FeatureResNet(ResNet):
    def __init__(self):
        super().__init__(BasicBlock, [3, 4, 6, 3], 1000)  # 特征提取用resnet

    def forward(self, x):
        x1 = self.conv1(x)
        x = self.bn1(x1)
        x = self.relu(x)
        x2 = self.maxpool(x)
        x = self.layer1(x2)
        x3 = self.layer2(x)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        return x1, x2, x3, x4, x5


class MFCNet(nn.Module):
    def __init__(self, num_classes, pretrained_net):
        super().__init__()

        self.pretrained_net = pretrained_net
        # self.pretrained_net.reqire
        self.relu = nn.ReLU(inplace=True)
        self.convDa = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bnDa = bn(128)
        self.convDb = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)

        self.conv5 = conv(512, 256, stride=2, transposed=True)
        self.bn5 = bn(256)
        self.conv6 = conv(256+256, 256, stride=2, transposed=True)
        self.bn6 = bn(256)
        self.conv7 = conv(256+128, 256, stride=2, transposed=True)
        self.bn7 = bn(256)
        self.conv8 = conv(256+64, 256, stride=2, transposed=True)
        self.bn8 = bn(256)
        self.conv9 = conv(256+64, 256, stride=2, transposed=True)
        self.bn9 = bn(256)

        self.conv5_det = conv(512, 256, stride=2, transposed=True)
        self.bn5d = bn(256)
        self.conv6_det = conv(256, 128, stride=2, transposed=True)
        self.bn6d = bn(128)
        self.conv7_det = conv(128, 64, stride=2, transposed=True)
        self.bn7d = bn(64)
        self.conv8_det = conv(64, 64, stride=2, transposed=True)
        self.bn8d = bn(64)
        self.conv9_det = conv(64, 32, stride=2, transposed=True)
        self.bn9d = bn(32)
        self.pool = nn.AvgPool2d(8)

        self.convDet = conv(32, num_classes, kernel_size=7)
        # init.constant(self.convDet.weight, 0)  # Zero init
        self.convDsa = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bnDsa = bn(256)
        self.convDsb = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.pretrained_net(x)

        feature_det = self.relu(self.bn5d(self.conv5_det(x5)))
        feature_det = self.relu(self.bn6d(self.conv6_det(feature_det + x4)))

        feature_det = self.relu(self.bn7d(self.conv7_det(feature_det + x3)))
        feature_det = self.relu(self.bn8d(self.conv8_det(feature_det + x2)))
        feature_det = self.relu(self.bn9d(self.conv9_det(feature_det + x1)))

        dect_dense = self.convDet(feature_det)
        dect_dense = torch.sigmoid(dect_dense)

        desc_semi = self.relu(self.bnDa(self.convDa(x3)))
        desc_semi = self.convDb(desc_semi)

        feature = self.relu(self.bn5(self.conv5(x5)))
        feature = self.relu(self.bn6(self.conv6(torch.cat((feature, x4), 1))))
        feature = self.relu(self.bn7(self.conv7(torch.cat((feature, x3), 1))))
        feature = self.relu(self.bn8(self.conv8(torch.cat((feature, x2), 1))))
        feature = self.relu(self.bn9(self.conv9(torch.cat((feature, x1), 1))))

        desc_dense = self.relu(self.bnDsa(self.convDsa(feature)))
        desc_dense = self.convDsb(desc_dense)

        # desc_semi = self.pool(desc_dense)

        return dect_dense, desc_semi, desc_dense


