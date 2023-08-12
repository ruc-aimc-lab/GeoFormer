#this code is mainly copied from Superpoint[https://github.com/rpautrat/SuperPoint]
#-*-coding:utf8-*-
import cv2
from math import pi
from numpy.random import uniform
from scipy.stats import truncnorm
import kornia
# from utils.params import dict_update
# from utils.tensor_op import erosion2d
# from utils.keypoint_op import *
from imgaug import augmenters as iaa

import torch
import torch.nn.functional as f

#-*-coding:utf8-*-
import collections
import numpy as np
import torch

import collections
import random

def parse_primitives(names, all_primitives):
    p = all_primitives if (names == 'all') \
            else (names if isinstance(names, list) else [names])
    assert set(p) <= set(all_primitives)
    return p


def dict_update(d, u):
    """Improved update for nested dictionaries.
    Arguments:
        d: The dictionary to be updated.
        u: The update dictionary.
    Returns:
        The updated dictionary.
    """
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return


def filter_points(points, shape, device='cpu'):
    """
    :param points: (N,2), formated as (y, x)
    :param shape: (W, H)
    :return: filtered point without outliers
    """
    if len(points) != 0:
        if len(points.shape) == 3:
            n, l, _ = points.shape
            points = points.reshape(n*l, 2)
            mask = (points >= 0) & (points <=
                                    torch.tensor(shape, device=device) - 1)
            mask = torch.all(mask, dim=1)
            points[~mask] = 0
            points = points.reshape(n, l, 2)
            mask = mask.reshape(n, l)
            return points, mask
        mask = (points >= 0) & (points <=
                                torch.tensor(shape, device=device) - 1)
        mask = torch.all(mask, dim=1)
        return points[mask], mask
    else:
        return points, []


def compute_keypoint_map(points, shape, device='cpu'):
    """
    :param shape: (H, W)
    :param points: (N,2)
    :return:
    """

    coord = torch.minimum(
        torch.round(points).type(torch.int),
        torch.tensor(shape, device=device) - 1)
    kmap = torch.zeros((shape), dtype=torch.int, device=device)
    kmap[coord[:, 0], coord[:, 1]] = 1
    return kmap

def warp_points_batch(points, homographies):
    # x, y
    B, l = points.shape[:2]
    #points: B, l, 2
    points = torch.cat((points, torch.ones((B, l, 1), device=homographies.device)),
                       dim=-1)
    # warped_points = torch.bmm(homographies, points.permute(0, 2, 1))
    ##each row dot each column of points.transpose
    if len(homographies.shape) == 2:
        homographies = homographies.unsqueeze(0)
    if len(homographies.shape) == 3 and homographies.shape[0] != B:
        homographies = homographies.repeat(B, 1, 1)
    warped_points = torch.bmm(homographies, points.permute(0, 2, 1))  # batch dot
    #
    warped_points = warped_points.permute(0, 2, 1)
    sc = warped_points[:, :, 2:]
    sc[sc==0] = 1e-6
    warped_points = warped_points[:, :, :2] / sc
    # warped_points = torch.flip(warped_points, dims=(2,))
    return warped_points
    # return (warped_points+0.5).long()


def warp_points(points, homographies, device='cpu'):
    """
    :param points: (N,2), tensor
    :param homographies: [B, 3, 3], batch of homographies
    :return: warped points B,N,2
    """
    if len(points) == 0:
        return points

    #TODO: Part1, the following code maybe not appropriate for your code
    points = torch.fliplr(points)
    if len(homographies.shape) == 2:
        homographies = homographies.unsqueeze(0)
    B = homographies.shape[0]
    ##TODO: uncomment the following line to get same result as tf version
    # homographies = torch.linalg.inv(homographies)
    points = torch.cat((points, torch.ones(
        (points.shape[0], 1), device=device)),
                       dim=1)
    ##each row dot each column of points.transpose
    warped_points = torch.tensordot(homographies,
                                    points.transpose(1, 0),
                                    dims=([2], [0]))  #batch dot
    ##
    warped_points = warped_points.reshape([B, 3, -1])
    warped_points = warped_points.transpose(2, 1)
    warped_points = warped_points[:, :, :2] / warped_points[:, :, 2:]
    #TODO: Part2, the flip operation is combinated with Part1
    warped_points = torch.flip(warped_points, dims=(2, ))
    #TODO: Note: one point case
    warped_points = warped_points.squeeze(dim=0)
    return warped_points.long()

def erosion2d(image, strel, origin=(0, 0), border_value=1e6):
    """
    :param image:BCHW
    :param strel: BHW
    :param origin: default (0,0)
    :param border_value: default 1e6
    :return:
    """
    image_pad = f.pad(image, [origin[0], strel.shape[1]-origin[0]-1, origin[1], strel.shape[2]-origin[1]-1], mode='constant', value=border_value)
    image_unfold = f.unfold(image_pad, kernel_size=strel.shape[1])#[B,C*sH*sW,L],L is the number of patches
    strel_flatten = torch.flatten(strel,start_dim=1).unsqueeze(-1)
    diff = image_unfold - strel_flatten
    # Take maximum over the neighborhood
    result, _ = diff.min(dim=1)
    # Reshape the image to recover initial shape
    return torch.reshape(result, image.shape)


def pixel_shuffle(tensor, scale_factor):
    """
    Implementation of pixel shuffle using numpy
    Parameters:
    -----------
    tensor: input tensor, shape is [N, C, H, W]
    scale_factor: scale factor to up-sample tensor
    Returns:
    --------
    tensor: tensor after pixel shuffle, shape is [N, C/(r*r), r*H, r*W],
        where r refers to scale factor
    """
    num, ch, height, width = tensor.shape
    assert ch % (scale_factor * scale_factor) == 0

    new_ch = ch // (scale_factor * scale_factor)
    new_height = height * scale_factor
    new_width = width * scale_factor

    tensor = tensor.reshape(
        [num, new_ch, scale_factor, scale_factor, height, width])
    # new axis: [num, new_ch, height, scale_factor, width, scale_factor]
    tensor = tensor.permute(0, 1, 4, 2, 5, 3)
    tensor = tensor.reshape(num, new_ch, new_height, new_width)
    return tensor


def pixel_shuffle_inv(tensor, scale_factor):
    """
    Implementation of inverted pixel shuffle using numpy
    Parameters:
    -----------
    tensor: input tensor, shape is [N, C, H, W]
    scale_factor: scale factor to down-sample tensor
    Returns:
    --------
    tensor: tensor after pixel shuffle, shape is [N, (r*r)*C, H/r, W/r],
        where r refers to scale factor
    """
    num, ch, height, width = tensor.shape
    assert height % scale_factor == 0
    assert width % scale_factor == 0

    new_ch = ch * (scale_factor * scale_factor)
    new_height = height // scale_factor
    new_width = width // scale_factor

    tensor = tensor.reshape(
        [num, ch, new_height, scale_factor, new_width, scale_factor])
    # new axis: [num, ch, scale_factor, scale_factor, new_height, new_width]
    tensor = tensor.permute(0, 1, 3, 5, 2, 4)
    tensor = tensor.reshape(num, new_ch, new_height, new_width)
    return tensor

def homographic_aug_pipline(img, pts=None, homography=None, device='cpu'):
    """
    :param img: [1,1,H,W]
    :param pts:[N,2]
    :param loftr_config:parameters
    :param device: cpu or cuda
    :return:
    """

    if len(img.shape) == 2:
        img = img.unsqueeze(dim=0).unsqueeze(dim=0)
    image_shape = img.shape[2:]  #HW
    if homography is None:
        homography = sample_homography(image_shape,
                                       device=device)
    ##
    #warped_image = cv2.warpPerspective(img, homography, tuple(image_shape[::-1]))
    warped_image = kornia.geometry.transform.warp_perspective(
        img, homography, image_shape, align_corners=True)

    warped_valid_mask = compute_valid_mask(image_shape,
                                           homography,
                                           device=device)
    if pts is None:
        return warped_image, warped_valid_mask, homography
    warped_points = warp_points(pts, homography, device=device)
    warped_points, points_mask = filter_points(warped_points, image_shape, device=device)
    # warped_points_map = compute_keypoint_map(warped_points,
    #                                          img.shape[2:],
    #                                          device=device)
    return warped_image, warped_points, points_mask, warped_valid_mask, homography
    # return {
    #     'warp': {
    #         'img': warped_image.squeeze(),
    #         'kpts': warped_points,
    #         'kpts_map':
    #         warped_points_map.squeeze(),  #some point maybe filtered
    #         'mask': warped_valid_mask.squeeze()
    #     },
    #     'homography': homography.squeeze(),
    # }
    #return warpped_image, warped_points, valid_mask, homography


def compute_valid_mask(image_shape,
                       homographies,
                       erosion_radius=0,
                       device='cpu'):
    """
    Compute a boolean mask of the valid pixels resulting from an homography applied to
    an image of a given shape. Pixels that are False correspond to bordering artifacts.
    A margin can be discarded using erosion.
    Arguments:
        input_shape: `[H, W]`, tuple, list or ndarray
        homography: B*3*3 homography
        erosion_radius: radius of the margin to be discarded.
    Returns: mask with values 0 or 1
    """
    if len(homographies.shape) == 2:
        homographies = homographies.unsqueeze(0)
    # TODO:uncomment this line if your want to get same result as tf version
    # homographies = torch.linalg.inv(homographies)
    B = homographies.shape[0]
    img_one = torch.ones(tuple([B, 1, *image_shape]),
                         device=device,
                         dtype=torch.float32)  #B,C,H,W
    mask = kornia.geometry.transform.warp_perspective(img_one,
                                                      homographies,
                                                      tuple(image_shape),
                                                      align_corners=True)
    mask = mask.round()  #B1HW
    #mask = cv2.warpPerspective(np.ones(image_shape), homography, dsize=tuple(image_shape[::-1]))#dsize=tuple([w,h])
    if erosion_radius > 0:
        # TODO: validation & debug
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (erosion_radius * 2, ) * 2)
        kernel = torch.as_tensor(kernel[np.newaxis, :, :], device=device)
        _, kH, kW = kernel.shape
        origin = ((kH - 1) // 2, (kW - 1) // 2)
        mask = erosion2d(mask, torch.flip(kernel, dims=[
            1, 2
        ]), origin=origin) + 1.  # flip kernel so perform as tf.nn.erosion2d

    return mask.squeeze(dim=1)  #BHW

from torchvision import transforms as T
import math
# def sample_homography(shape, device, seed=None):
#     h, w = shape
#     affine_params = T.RandomAffine(20).get_params(degrees=[-30, 30], translate=[0.3, 0.3],
#                                                   scale_ranges=[0.5, 0.6],
#                                                   shears=[0.1, 0.1], img_size=[h, w])
#     angle = -affine_params[0] * math.pi / 180
#     theta_ = torch.tensor([
#         [1 / affine_params[2] * math.cos(angle), math.sin(-angle), -affine_params[1][0] / w],
#         [math.sin(angle), 1 / affine_params[2] * math.cos(angle), -affine_params[1][1] / h],
#         [0, 0, 1]
#     ], dtype=torch.float)
#
#     theta_flips = [
#         torch.tensor([
#             [1, 0, 0],
#             [0, -1, h-1],
#             [0, 0, 1]], dtype=torch.float),
#         torch.tensor([
#             [-1, 0, w-1],
#             [0, -1, h-1],
#             [0, 0, 1]], dtype=torch.float),
#         torch.tensor([
#             [-1, 0, w-1],
#             [0, 1, 0],
#             [0, 0, 1]], dtype=torch.float),
#     ]
#     if random.random() > 0.9:
#         theta_flip = theta_flips[np.random.randint(3)]
#         theta_ = theta_.mm(theta_flip)
#     return theta_.unsqueeze(0).to(device)

def affine_image(h, w):
    """
    Perform affine transformation on images
    :param images: (B, C, H, W)
    :param keypoint_labels: corresponding labels
    :param value_map: value maps, used to record history learned geo_points
    :return: results of affine images, affine labels, affine value maps, affine transformed grid_inverse, inverse transformed grid_inverse
    """

    affine_params = T.RandomAffine(20).get_params(degrees=[-15, 15], translate=[0.2, 0.2],
                                                  scale_ranges=[0.9, 1.35],
                                                  shears=None, img_size=[h, w])

    angle = -affine_params[0] * math.pi / 180
    theta_ = torch.tensor([
        [1 / affine_params[2] * math.cos(angle), math.sin(-angle), -affine_params[1][0] / h],
        [math.sin(angle), 1 / affine_params[2] * math.cos(angle), -affine_params[1][1] / w],
        [0, 0, 1]
    ], dtype=torch.float).numpy()
    return theta_
def sample_homography(shape, device):
    h, w = shape
    corners = np.array([[0, 0], [0, h], [w, 0], [w, h]], dtype=np.float32)

    rg = max(h, w)
    warp = np.random.randint(-rg//3, rg//3, size=(4, 2)).astype(np.float32)
    if random.random() < 0.2:
        warp = np.random.randint(-5, 5, size=(4, 2)).astype(np.float32)
    # get the corresponding warped image
    M = cv2.getPerspectiveTransform(corners, corners + warp)
    # warped = cv2.warpPerspective(src=image, M=M, dsize=(image.shape[1], image.shape[0]))  # return an image type
    homography = torch.from_numpy(M).to(device).float()
    theta_flips = [
        torch.tensor([
            [-1, 0, w],
            [0, 1, 0],
            [0, 0, 1]], dtype=torch.float),
        torch.tensor([
            [1, 0, 0],
            [0, -1, h],
            [0, 0, 1]], dtype=torch.float),
        # torch.tensor([
        #     [-1, 0, w - 1],
        #     [0, -1, h - 1],
        #     [0, 0, 1]], dtype=torch.float),
    ]
    if random.random() < 0.2:
        theta_flip = theta_flips[random.randint(0, 1)]
        if random.random() < 0.6:
            homography = theta_flip
        else:
            homography = homography.mm(theta_flip)

    homography = homography.unsqueeze(0)
    return homography

# def sample_homography(shape, device, seed=None, mode=0):
#     """Sample a random valid homography, as a composition of translation, rotation,
#     scaling, shearing and perspective transforms.
#     Arguments:
#         shape: A rank-2 `Tensor` specifying the height and width of the original image.
#         parameters: dictionnary containing all infor on the transformations to apply.
#         ex:
#         parameters={}
#         scaling={'use_scaling':True, 'min_scaling_x':0.7, 'max_scaling_x':2.0, \
#                  'min_scaling_y':0.7, 'max_scaling_y':2.0}
#         perspective={'use_perspective':False, 'min_perspective_x':0.000001, 'max_perspective_x':0.0009, \
#                   'min_perspective_y':0.000001, 'max_perspective_y':0.0009}
#         translation={'use_translation':True, 'max_horizontal_dis':100, 'max_vertical_dis':100}
#         shearing={'use_shearing':True, 'min_shearing_x':-0.3, 'max_shearing_x':0.3, \
#                   'min_shearing_y':-0.3, 'max_shearing_y':0.3}
#         rotation={'use_rotation':True, 'max_angle':90}
#         parameters['scaling']=scaling
#         parameters['perspective']=perspective
#         parameters['translation']=translation
#         parameters['shearing']=shearing
#         parameters['rotation']=rotation
#     Returns:
#         A 3x3 matrix corresponding to the homography transform.
#     """
#     parameters = {}
#     modes = ['easy', 'hard']
#     if random.random() < 0.5:
#         mode = modes[0]
#     else:
#         mode = modes[-1]
#     if mode == 'hard':
#         scaling = {'use_scaling': 1, 'min_scaling_x': 0.4, 'max_scaling_x': 2, \
#                    'min_scaling_y': 0.4, 'max_scaling_y': 2}
#         perspective = {'use_perspective': 1, 'min_perspective_x': 0.00001, 'max_perspective_x': 0.0009, \
#                        'min_perspective_y': 0.00001, 'max_perspective_y': 0.0009}
#         translation = {'use_translation': 1, 'max_horizontal_dis': 100, 'max_vertical_dis': 100}
#         shearing = {'use_shearing': 0.3, 'min_shearing_x': -0.1, 'max_shearing_x': 0.1, \
#                     'min_shearing_y': -0.1, 'max_shearing_y': 0.1}
#         rotation = {'use_rotation': 0.8, 'max_angle': 30}
#         flip = 0.1
#     elif mode == 'mid':
#         scaling = {'use_scaling': 1, 'min_scaling_x': 0.4, 'max_scaling_x': 2, \
#                    'min_scaling_y': 0.4, 'max_scaling_y': 2}
#         perspective = {'use_perspective': 0, 'min_perspective_x': 0.000001, 'max_perspective_x': 0.0009,
#                        'min_perspective_y': 0.000001, 'max_perspective_y': 0.0009}
#         translation = {'use_translation': 0.3, 'max_horizontal_dis': 100, 'max_vertical_dis': 100}
#         shearing = {'use_shearing': 0.1, 'min_shearing_x': -0.1, 'max_shearing_x': 0.1, \
#                     'min_shearing_y': -0.1, 'max_shearing_y': 0.1}
#         rotation = {'use_rotation': 0.5, 'max_angle': 30}
#         flip = 0.1
#     else:
#         scaling = {'use_scaling': 1, 'min_scaling_x': 0.7, 'max_scaling_x': 1.5, \
#                    'min_scaling_y': 0.7, 'max_scaling_y': 1.5}
#         perspective = {'use_perspective': 0, 'min_perspective_x': 0.000001, 'max_perspective_x': 0.0009, \
#                        'min_perspective_y': 0.000001, 'max_perspective_y': 0.0009}
#         translation = {'use_translation': 0.3, 'max_horizontal_dis': 50, 'max_vertical_dis': 50}
#         shearing = {'use_shearing': 0., 'min_shearing_x': -0.1, 'max_shearing_x': 0.1, \
#                     'min_shearing_y': -0.1, 'max_shearing_y': 0.1}
#         rotation = {'use_rotation': 1., 'max_angle': 25}
#         flip = 0.
#     parameters['scaling'] = scaling
#     parameters['perspective'] = perspective
#     parameters['translation'] = translation
#     parameters['shearing'] = shearing
#     parameters['rotation'] = rotation
#     (h, w) = shape
#     if seed is not None:
#         random.seed(seed)
#     if random.random() < parameters['rotation']['use_rotation']:
#         (h, w) = shape
#         center = (w // 2, h // 2)
#         y = random.randint(-parameters['rotation']['max_angle'], \
#                            parameters['rotation']['max_angle'])
#         # perform the rotation
#         M = cv2.getRotationMatrix2D(center, y, 1.0)
#         homography_rotation = np.concatenate([M, np.array([[0, 0, 1]])], axis=0)
#     else:
#         homography_rotation = np.eye(3)
#
#     if random.random() < parameters['translation']['use_translation']:
#         tx = random.randint(-parameters['translation']['max_horizontal_dis'], \
#                             parameters['translation']['max_horizontal_dis'])
#         ty = random.randint(-parameters['translation']['max_vertical_dis'], \
#                             parameters['translation']['max_vertical_dis'])
#         homography_translation = np.eye(3)
#         homography_translation[0, 2] = tx
#         homography_translation[1, 2] = ty
#     else:
#         homography_translation = np.eye(3)
#
#     if random.random() < parameters['scaling']['use_scaling']:
#         scaling_x = random.choice(np.arange(parameters['scaling']['min_scaling_x'], \
#                                             parameters['scaling']['max_scaling_x'], 0.1))
#         scaling_y = random.choice(np.arange(parameters['scaling']['min_scaling_y'], \
#                                             parameters['scaling']['max_scaling_y'], 0.1))
#         if random.random() < 0.7:
#             scaling_x = scaling_y
#         homography_scaling = np.eye(3)
#         homography_scaling[0, 0] = scaling_x
#         homography_scaling[1, 1] = scaling_y
#     else:
#         homography_scaling = np.eye(3)
#
#     if random.random() < parameters['shearing']['use_shearing']:
#         shearing_x = random.choice(np.arange(parameters['shearing']['min_shearing_x'], \
#                                              parameters['shearing']['max_shearing_x'], 0.0001))
#         shearing_y = random.choice(np.arange(parameters['shearing']['min_shearing_y'], \
#                                              parameters['shearing']['max_shearing_y'], 0.0001))
#         homography_shearing = np.eye(3)
#         homography_shearing[0, 1] = shearing_y
#         homography_shearing[1, 0] = shearing_x
#     else:
#         homography_shearing = np.eye(3)
#
#     if random.random() < parameters['perspective']['use_perspective']:
#         perspective_x = random.choice(np.arange(parameters['perspective']['min_perspective_x'],
#                                                 parameters['perspective']['max_perspective_x'], 0.00001))
#         perspective_y = random.choice(np.arange(parameters['perspective']['min_perspective_y'], \
#                                                 parameters['perspective']['max_perspective_y'], 0.00001))
#         homography_perspective = np.eye(3)
#         homography_perspective[2, 0] = perspective_x
#         homography_perspective[2, 1] = perspective_y
#     else:
#         homography_perspective = np.eye(3)
#
#     homography = np.matmul(np.matmul(np.matmul(np.matmul(homography_rotation, homography_translation), \
#                                                homography_shearing), homography_scaling), \
#                            homography_perspective)
#     homography = torch.from_numpy(homography).to(device).float()
#     theta_flips = [
#         torch.tensor([
#             [-1, 0, w],
#             [0, 1, 0],
#             [0, 0, 1]], dtype=torch.float),
#         torch.tensor([
#             [1, 0, 0],
#             [0, -1, h],
#             [0, 0, 1]], dtype=torch.float),
#         torch.tensor([
#             [-1, 0, w],
#             [0, -1, h],
#             [0, 0, 1]], dtype=torch.float),
#
#     ]
#     if random.random() < 0.2:
#         theta_flip = theta_flips[random.randint(0, 2)]
#         if random.random() < 0.5:
#             homography = theta_flip
#         else:
#             homography = homography.mm(theta_flip)
#
#     homography = homography.unsqueeze(0)
#     return homography

# def sample_homography(shape, loftr_config=None, device='cpu'):
#
#     default_config = {
#         'perspective': False,
#         'scaling': True,
#         'rotation': True,
#         'translation': True,
#         'n_scales': 5,
#         'n_angles': 25,
#         'scaling_amplitude': 0.2,
#         'perspective_amplitude_x': 0.1,
#         'perspective_amplitude_y': 0.1,
#         'patch_ratio': 0.5,
#         'max_angle': pi / 18,
#         'allow_artifacts': False,
#         'translation_overflow': 0.
#     }
#
#     #TODO: not tested
#     if loftr_config is not None:
#         loftr_config = dict_update(default_config, loftr_config)
#     else:
#         loftr_config = default_config
#
#     std_trunc = 2
#
#     # Corners of the input patch
#     margin = (1 - loftr_config['patch_ratio']) / 2
#     pts1 = margin + np.array([[0, 0], [0, loftr_config['patch_ratio']],
#                               [loftr_config['patch_ratio'], loftr_config['patch_ratio']],
#                               [loftr_config['patch_ratio'], 0]])
#     pts2 = pts1.copy()
#
#     # Random perspective and affine perturbations
#     if loftr_config['perspective']:
#         if not loftr_config['allow_artifacts']:
#             perspective_amplitude_x = min(loftr_config['perspective_amplitude_x'],
#                                           margin)
#             perspective_amplitude_y = min(loftr_config['perspective_amplitude_y'],
#                                           margin)
#         else:
#             perspective_amplitude_x = loftr_config['perspective_amplitude_x']
#             perspective_amplitude_y = loftr_config['perspective_amplitude_y']
#         perspective_displacement = truncnorm(-std_trunc,
#                                              std_trunc,
#                                              loc=0.,
#                                              scale=perspective_amplitude_y /
#                                              2).rvs(1)
#         h_displacement_left = truncnorm(-std_trunc,
#                                         std_trunc,
#                                         loc=0.,
#                                         scale=perspective_amplitude_x /
#                                         2).rvs(1)
#         h_displacement_right = truncnorm(-std_trunc,
#                                          std_trunc,
#                                          loc=0.,
#                                          scale=perspective_amplitude_x /
#                                          2).rvs(1)
#         pts2 += np.array([[h_displacement_left, perspective_displacement],
#                           [h_displacement_left, -perspective_displacement],
#                           [h_displacement_right, perspective_displacement],
#                           [h_displacement_right,
#                            -perspective_displacement]]).squeeze()
#
#     # Random scaling
#     # sample several scales, check collision with borders, randomly pick a valid one
#     if loftr_config['scaling']:
#         scales = truncnorm(-std_trunc,
#                            std_trunc,
#                            loc=1,
#                            scale=loftr_config['scaling_amplitude'] / 2).rvs(
#                                loftr_config['n_scales'])
#         #scales = np.random.uniform(0.8, 2, loftr_config['n_scales'])
#         scales = np.concatenate((np.array([1]), scales), axis=0)
#
#         center = np.mean(pts2, axis=0, keepdims=True)
#         scaled = (pts2 - center)[
#             np.newaxis, :, :] * scales[:, np.newaxis, np.newaxis] + center
#         if loftr_config['allow_artifacts']:
#             valid = np.arange(
#                 loftr_config['n_scales'])  # all scales are valid except scale=1
#         else:
#             valid = (scaled >= 0.) * (scaled < 1.)
#             valid = valid.prod(axis=1).prod(axis=1)
#             valid = np.where(valid)[0]
#         idx = valid[np.random.randint(valid.shape[0],
#                                       size=1)].squeeze().astype(int)
#         pts2 = scaled[idx, :, :]
#
#     # Random translation
#     if loftr_config['translation']:
#         t_min, t_max = np.min(pts2, axis=0), np.min(1 - pts2, axis=0)
#         if loftr_config['allow_artifacts']:
#             t_min += loftr_config['translation_overflow']
#             t_max += loftr_config['translation_overflow']
#         pts2 += np.array(
#             [uniform(-t_min[0], t_max[0], 1),
#              uniform(-t_min[1], t_max[1], 1)]).T
#
#     # Random rotation
#     # sample several rotations, check collision with borders, randomly pick a valid one
#     if loftr_config['rotation']:
#         angles = np.linspace(-loftr_config['max_angle'],
#                              loftr_config['max_angle'],
#                              num=loftr_config['n_angles'])
#         angles = np.concatenate((np.array([0.]), angles),
#                                 axis=0)  # in case no rotation is valid
#         center = np.mean(pts2, axis=0, keepdims=True)
#         rot_mat = np.reshape(
#             np.stack([
#                 np.cos(angles), -np.sin(angles),
#                 np.sin(angles),
#                 np.cos(angles)
#             ],
#                      axis=1), [-1, 2, 2])
#         rotated = np.matmul(
#             (pts2 - center)[np.newaxis, :, :], rot_mat) + center
#
#         if loftr_config['allow_artifacts']:
#             valid = np.arange(
#                 loftr_config['n_angles'])  # all scales are valid except scale=1
#         else:
#             valid = (rotated >= 0.) * (rotated < 1.)
#             valid = valid.prod(axis=1).prod(axis=1)
#             valid = np.where(valid)[0]
#         idx = valid[np.random.randint(valid.shape[0],
#                                       size=1)].squeeze().astype(int)
#         pts2 = rotated[idx, :, :]
#
#     # Rescale to actual size
#     shape = np.array(shape[::-1])  # different convention [y, x]
#     pts1 *= shape[np.newaxis, :]
#     pts2 *= shape[np.newaxis, :]
#
#     # this homography is the same with tf version and this line
#     homography = cv2.getPerspectiveTransform(np.float32(pts1),
#                                              np.float32(pts2))
#     homography = torch.tensor(homography, device=device,
#                               dtype=torch.float32).unsqueeze(dim=0)
#     ## equals to the following 3 lines
#     # pts1 = torch.tensor(pts1[np.newaxis,:], device=device, dtype=torch.float32)
#     # pts2 = torch.tensor(pts2[np.newaxis,:], device=device, dtype=torch.float32)
#     # homography0 = kornia.get_perspective_transform(pts1, pts2)
#
#     #TODO: comment the follwing line if you want same result as tf version
#     # since if we use homography directly ofr opencv function, for example warpPerspective
#     # the result we get is different from tf version. In order to get a same result, we have to
#     # apply inverse operation,like this
#     #homography = np.linalg.inv(homography)
#     homography = torch.inverse(
#         homography)  #inverse here to be consistent with tf version
#     #debug
#     #homography = torch.eye(3,device=device).unsqueeze(dim=0)
#     return homography  #[1,3,3]


def ratio_preserving_resize(img, target_size):
    '''
    :param img: raw img
    :param target_size: (h,w)
    :return:
    '''
    scales = np.array((target_size[0] / img.shape[0],
                       target_size[1] / img.shape[1]))  ##h_s,w_s

    new_size = np.round(np.array(img.shape) * np.max(scales)).astype(int)  #
    temp_img = cv2.resize(img, tuple(new_size[::-1]))
    curr_h, curr_w = temp_img.shape
    target_h, target_w = target_size
    ##
    hp = (target_h - curr_h) // 2
    wp = (target_w - curr_w) // 2
    aug = iaa.Sequential([
        iaa.CropAndPad(px=(hp, wp, target_h - curr_h - hp,
                           target_w - curr_w - wp),
                       keep_size=False),
    ])
    new_img = aug(images=temp_img)
    return new_img


if __name__ == '__main__':
    pass
