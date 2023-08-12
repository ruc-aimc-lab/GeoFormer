
import numpy as np
from skimage.feature import peak_local_max
import torch
import torch.nn.functional as F
import cv2

def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor,
                            log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)

def get_mask(img_ori):


    b, g, r = cv2.split(img_ori)

    # 对红色通道图进行CLAHE 处理

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    img_clahe_red = clahe.apply(r)

    # 获得 红色通道的掩膜图
    ret, img_threshold = cv2.threshold(img_clahe_red, 10, 255, cv2.THRESH_BINARY)

    # 创建 半径为 3 的圆形结构元素做kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # 对 ROI 区进行腐蚀运算
    img_erode = cv2.erode(img_threshold, kernel, iterations=1)
    return img_erode
def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor,
                          iters: int) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m * one).to(scores), (n * one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat(
        [torch.cat([scores, bins0], -1),
         torch.cat([bins1, alpha], -1)], 1)

    norm = -(ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def generate_window(keypoints_list, img_size, window_size=11, scale=2):
    h, w = img_size
    new_keypoints_list = []
    masks_list = []
    device = keypoints_list[0].device

    windows = torch.empty(window_size, window_size, 2).to(device)
    val = torch.arange(window_size).reshape(-1, 1).to(device)
    y_val = val.repeat([1, window_size])
    x_val = val.T.repeat([window_size, 1])
    windows[:, :, 0] = x_val
    windows[:, :, 1] = y_val
    windows = windows - windows[window_size // 2, window_size // 2][0]
    windows = windows.unsqueeze(0)*scale
    new_keypoints_list, masks_list = [], []
    for b, kps in enumerate(keypoints_list):
        kps = kps.unsqueeze(1).unsqueeze(1).repeat([1, window_size, window_size, 1])
        kps = kps + windows
        kps = kps.view(-1, window_size*window_size, 2)
        check_out = (kps[:, :, 0] < 0) | (kps[:, :, 1] < 0) | (kps[:, :, 0] >= w) | (kps[:, :, 1] >= h)
        mask = torch.ones_like(check_out).to(check_out.device)
        mask[check_out] = 0

        kps[check_out] = 0
        new_keypoints_list.append(kps.long())
        masks_list.append(mask)
    return new_keypoints_list, masks_list


def remove_borders(keypoints, border: int, height: int, width: int):
    """ Removes keypoints too close to the border """
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask]

def torch_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert (nms_radius >= 0)

    size = nms_radius * 2 + 1
    avg_size = 2
    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=size, stride=1, padding=nms_radius)

    def avg_pool(x):  # using avg_pool to mask the repeated maximum values in the windows.
        return torch.nn.functional.avg_pool2d(
            x, kernel_size=avg_size * 2 + 1, stride=1, padding=avg_size)

    zeros = torch.zeros_like(scores)
    # max_map = max_pool(scores)
    # avg_map = avg_pool(scores)
    max_mask = scores == max_pool(scores)
    max_mask_ = torch.rand(max_mask.shape).to(max_mask.device) / 10
    max_mask_[~max_mask] = 0
    mask = max_mask_ == max_pool(max_mask_)

    return torch.where(mask, scores, zeros)

def non_max_suppression(image, size_filter, proba):
    non_max = peak_local_max(image, min_distance=size_filter, threshold_abs=proba, exclude_border=True, indices=False)
    kp = np.where(non_max > 0)
    if len(kp[0]) != 0:
        for i in range(len(kp[0])):

            window = non_max[kp[0][i] - size_filter:kp[0][i] + (size_filter + 1), \
                     kp[1][i] - size_filter:kp[1][i] + (size_filter + 1)]
            if np.sum(window) > 1:
                window[:, :] = 0
    return non_max

def get_map_keypoints(h, w, scale=8):
    xs = torch.linspace(0, w // scale - 1, steps=w // scale)
    ys = torch.linspace(0, h // scale - 1, steps=h // scale)

    ys, xs = torch.meshgrid(ys, xs)
    keypoint = torch.cat((xs.reshape(-1, 1), ys.reshape(-1, 1)), -1).long()
    keypoint = keypoint * scale
    return keypoint.long()

def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert (nms_radius >= 0)

    size = nms_radius * 2 + 1
    avg_size = 2
    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=size, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    # max_map = max_pool(scores)

    max_mask = scores == max_pool(scores)
    max_mask_ = torch.rand(max_mask.shape).to(max_mask.device) / 10
    max_mask_[~max_mask] = 0
    mask = ((max_mask_ == max_pool(max_mask_)) & (max_mask_ > 0))

    return torch.where(mask, scores, zeros)

def sample_descriptors(keypoints_list, descriptor_maps, s: int = 8, norm=True) -> list:
    def sample_one(keypoints, descriptor):
        b, c, h, w = descriptor.shape
        if keypoints is None:
            return torch.empty([1, 0, c]).to(descriptor)
        keypoints = keypoints.clone().float()
        keypoints = keypoints // s

        if len(keypoints.shape) == 4:
            n, l, w, _ = keypoints.shape
            keypoints = keypoints.reshape(n, l*w, 2)

            descriptor = descriptor[0, :, keypoints[0, :, 1].long(),
                         keypoints[0, :, 0].long()]
            descriptor = descriptor.reshape(b, c, l, w)
            return descriptor.permute(0, 2, 3, 1)
        # # # if s == 1:
        descriptor = descriptor[0, :, keypoints[0, :, 1].long(),
                     keypoints[0, :, 0].long()]
        descriptor = descriptor.reshape(b, c, -1)
        if norm:
            descriptor = descriptor / c ** .5
        # descriptor = torch.nn.functional.normalize(descriptor.reshape(
        #     b, c, -1), p=2, dim=1)
        return descriptor.permute(0, 2, 1)

        # keypoints /= torch.tensor([(w * s - s / 2 - 0.5), (h * s - s / 2 - 0.5)],
        #                           ).to(keypoints)[None]
        # keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
        #
        # args = {'align_corners': True} if int(torch.__version__[2]) > 2 else {}
        # descriptor = torch.nn.functional.grid_sample(
        #     descriptor, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
        #
        # descriptor = descriptor.reshape(b, c, -1)
        # descriptor = torch.nn.functional.normalize(
        #     descriptor.reshape(b, c, -1), p=2, dim=1)
        # return descriptor.permute(0, 2, 1)

    descriptors = [sample_one(k[None], d[None])[0] if k is not None else None
                   for k, d in zip(keypoints_list, descriptor_maps)]

    return descriptors

def datasets_normalized(images):
    # images_normalized = np.empty(images.shape)
    images_std = np.std(images)
    images_mean = np.mean(images)
    images_normalized = (images - images_mean) / (images_std + 1e-6)
    minv = np.min(images_normalized)
    images_normalized = ((images_normalized - minv) /
                         (np.max(images_normalized) - minv)) * 255

    return images_normalized

def adjust_gamma(images, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    new_images = np.empty(images.shape)
    new_images[:, :] = cv2.LUT(np.array(images[:, :],
                                        dtype=np.uint8), table)

    return new_images

def clahe_equalized(images):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    images_equalized = np.empty(images.shape)
    images_equalized[:, :] = clahe.apply(np.array(images[:, :],
                                                  dtype=np.uint8))

    return images_equalized
def pre_processing(data):
    """ Enhance retinal images """
    train_imgs = datasets_normalized(data)
    train_imgs = clahe_equalized(train_imgs)
    train_imgs = adjust_gamma(train_imgs, 1.2)

    # train_imgs = train_imgs / 255.

    return train_imgs.astype(np.uint8)
