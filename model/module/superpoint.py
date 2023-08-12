from pathlib import Path
from typing import Dict, List

import torch
from torch import nn

from utils.common_utils import non_max_suppression, generate_conf
import torch.nn.functional as F


def max_pool(x, nms_radius: int):
    return torch.nn.functional.max_pool2d(
        x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)

def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert (nms_radius >= 0)

    size = nms_radius * 2 + 1
    # avg_size = 2
    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=size, stride=1, padding=nms_radius)

    # def avg_pool(x):  # using avg_pool to mask the repeated maximum values in the windows.
    #     return torch.nn.functional.avg_pool2d(
    #         x, kernel_size=avg_size * 2 + 1, stride=1, padding=avg_size)

    zeros = torch.zeros_like(scores)
    # max_map = max_pool(scores)
    # avg_map = avg_pool(scores)
    max_mask = scores == max_pool(scores)
    max_mask_ = torch.rand(max_mask.shape).to(max_mask.device) / 10
    max_mask_[~max_mask] = 0
    mask = (max_mask_ == max_pool(max_mask_)) & (max_mask_ > 0)

    return torch.where(mask, scores, zeros)
# def simple_nms(scores, nms_radius: int):
#     """ Fast Non-maximum suppression to remove nearby points """
#     assert (nms_radius >= 0)
#
#     zeros = torch.zeros_like(scores)
#     max_mask = scores == max_pool(scores, nms_radius)
#     # for _ in range(2):
#     #     supp_mask = max_pool(max_mask.float(), nms_radius) > 0
#     #     supp_scores = torch.where(supp_mask, zeros, scores)
#     #     new_max_mask = supp_scores == max_pool(supp_scores, nms_radius)
#     #     max_mask = max_mask | (new_max_mask & (~supp_mask))
#     return torch.where(max_mask, scores, zeros)


def remove_borders(keypoints, scores, border: int, height: int, width: int):
    """ Removes keypoints too close to the border """
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


def top_k_keypoints(keypoints, scores, k: int):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """ Interpolate descriptors at keypoint locations """
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor([(w * s - s / 2 - 0.5), (h * s - s / 2 - 0.5)],
                              ).to(keypoints)[None]
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    align_arg = True if int(torch.__version__[2]) > 2 else False
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', align_corners=align_arg)
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2., dim=1)
    return descriptors


class SuperPoint(nn.Module):
    """SuperPoint Convolutional Detector and Descriptor

    SuperPoint: Self-Supervised Interest Point Detection and
    Description. Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. In CVPRW, 2019. https://arxiv.org/abs/1712.07629

    """
    default_config = {
        'descriptor_dim': 128.,
        'nms_radius': 4.,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1.,
        'remove_borders': 4.,
    }

    def __init__(self, config=None):
        super().__init__()
        if config is None:
            self.config = self.default_config
        else:
            self.config = {**self.default_config, **config}

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 128

        self.nms_radius = 4
        self.keypoint_threshold = 0.05
        self.max_keypoints = 5000

        # self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        # self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        # self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        # self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        # self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        # self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        # self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        # self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.conv1a = single_conv(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = single_conv(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = single_conv(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = single_conv(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = single_conv(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = single_conv(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = single_conv(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = single_conv(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convDa = single_conv(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(
            c5, int(self.config['descriptor_dim']),
            kernel_size=1, stride=1, padding=0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dconv_up3 = double_conv(c3 + c4, c5)
        self.dconv_up2 = double_conv(c2 + c5, c5)
        self.dconv_up1 = double_conv(c1 + c5, c5)
        #
        self.conv_last = nn.Conv2d(c5, int(self.config['descriptor_dim']), kernel_size=1)

        # path = Path(__file__).parent / 'superpoint_v1.pth'
        # self.load_state_dict(torch.load(str(path)), strict=False)

        # mk = int(self.loftr_config['max_keypoints'])
        # if mk == 0 or mk < -1:
        #     raise ValueError('\"max_keypoints\" must be positive or \"-1\"')

        # print('Loaded SuperPoint pre_model')

    def add_extPoint(self, h, w, scale):
        xs = torch.linspace(0, w // scale - 1, steps=w // scale)
        ys = torch.linspace(0, h // scale - 1, steps=h // scale)

        xs, ys = torch.meshgrid(xs, ys)
        keypoint = torch.cat((xs.reshape(-1, 1), ys.reshape(-1, 1)), -1).long()
        keypoint = keypoint * scale
        return keypoint

    def getKeypoints(self, scores, train=True):
        b, _, h, w = scores.shape
        # scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        # scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)
        device = scores.device
        maps_np = scores.cpu().detach().numpy()
        nms_maps = [non_max_suppression(maps_np[i, 0, :, :], self.nms_radius, self.keypoint_threshold) for i in range(b)]
        nms_maps = [torch.from_numpy(mp).to(scores) for mp in nms_maps]
        keypoints = [torch.nonzero(mp) for mp in nms_maps]
        scores = [s[tuple(k.t())] for s, k in zip(scores[:, 0, :, :], keypoints)]

        # scores = simple_nms(scores, self.nms_radius)
        # scores_bin = 0

        # Extract keypoints
        # keypoints = [
        #     torch.nonzero(s > self.keypoint_threshold)
        #     for s in scores]
        # scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

        # Discard keypoints near the image borders
        keypoints, scores = list(zip(*[
            remove_borders(k, s, 4, h * 8, w * 8)
            for k, s in zip(keypoints, scores)]))

        # Keep the k keypoints with highest score
        if self.max_keypoints >= 0:
            keypoints, scores = list(zip(*[
                top_k_keypoints(k, s, self.max_keypoints)
                for k, s in zip(keypoints, scores)]))

        # Convert (h, w) to (x, y)
        keypoints = [torch.flip(k, [1]).long() for k in keypoints]
        if train:
            keypoints = [self.add_extPoint(h, w, 8).to(device) if len(k) < 400 else k
                         for k in keypoints]

        return keypoints

    def forward(self, image):
        """ Compute keypoints, scores, descriptors for image """
        # Shared Encoder
        x = self.conv1a(image)
        x = self.conv1b(x)
        conv1 = x
        x = self.pool(x)
        x = self.conv2a(x)
        x = self.conv2b(x)
        conv2 = x
        x = self.pool(x)
        x = self.conv3a(x)
        x = self.conv3b(x)
        conv3 = x
        x = self.pool(x)
        x = self.conv4a(x)
        x = self.conv4b(x)

        # cPa = self.relu(self.convPa(x))
        # scores = self.convPb(cPa)
        # scores = torch.nn.functional.softmax(scores, 1)

        # scores = scores[:, :-1]
        # b, _, h, w = scores.shape
        # scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        # dect = scores.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)

        cPa = self.upsample(x)
        cPa = torch.cat([cPa, conv3], dim=1)

        cPa = self.dconv_up3(cPa)
        cPa = self.upsample(cPa)
        cPa = torch.cat([cPa, conv2], dim=1)

        cPa = self.dconv_up2(cPa)
        cPa = self.upsample(cPa)
        cPa = torch.cat([cPa, conv1], dim=1)
        #
        cPa = self.dconv_up1(cPa)
        #
        cPa = self.conv_last(cPa)
        # dect = torch.sigmoid(dect)


        # Compute the dense descriptors
        cDa = self.convDa(x)
        descriptors = self.convDb(cDa)
        # descriptors = torch.nn.functional.normalize(descriptors, p=2., dim=1)

        # keypoints = self.getKeypoints(dect)
        return descriptors, cPa

def double_conv(in_channels, out_channels, kernel_size=3, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.InstanceNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.InstanceNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )
def single_conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, need_norm=True):
    if need_norm:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
    )

def MLP(channels: List[int], do_bn: bool = True) -> nn.Module:
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
        else:
            layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)

class DetectorMap(nn.Module):

    def __init__(self, desc_dim=128, bin=False):
        super().__init__()
        self.desc_dim = desc_dim
        self.mlp = MLP([desc_dim, 128, 64, 1])
        self.tempr = 1
        self.bin_score = nn.Parameter(
                torch.tensor(1., requires_grad=True))
        self.bin = bin

    def forward(self, query_desc, refer_desc):

        b = query_desc.size(0)
        # query_desc = query_desc.view(b, self.desc_dim, -1)
        # refer_desc = refer_desc.view(b, self.desc_dim, -1)
        m, n = query_desc.shape[-1], refer_desc.shape[-1]
        dect_query_map = self.mlp(query_desc.detach())
        dect_refer_map = self.mlp(refer_desc.detach())

        conf_matrix = generate_conf(self.bin_score, query_desc, refer_desc, self.desc_dim, tempr=self.tempr, bin=self.bin)

        # scores = torch.einsum('bdm,bdn->bmn', query_desc / self.desc_dim ** .5, refer_desc / self.desc_dim ** .5)
        # alpha = self.bin_score
        # bins0 = alpha.expand(b, m, 1)
        # bins1 = alpha.expand(b, 1, n)
        # alpha = alpha.expand(b, 1, 1)
        #
        # scores = torch.cat([torch.cat([scores, bins0], -1),
        #                        torch.cat([bins1, alpha], -1)], 1)
        # conf_matrix = F.softmax(scores, 1) * F.softmax(scores, 2)
        # conf_matrix = torch.clamp(conf_matrix, 1e-6, 1 - 1e-6)
        # conf_matrix = conf_matrix[:, :-1, :-1]
        # return dect_query_map, dect_refer_map
        return conf_matrix, dect_query_map, dect_refer_map
#