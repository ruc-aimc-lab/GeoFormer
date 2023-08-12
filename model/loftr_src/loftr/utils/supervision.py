from math import log
from loguru import logger

import torch
from einops import repeat
from kornia.utils import create_meshgrid

from utils.homography import warp_points_batch
from .geometry import warp_kpts

##############  ↓  Coarse-Level supervision  ↓  ##############


@torch.no_grad()
def mask_pts_at_padded_regions(grid_pt, mask):
    """For megadepth dataset, zero-padding exists in images"""
    mask = repeat(mask, 'n h w -> n (h w) c', c=2)
    grid_pt[~mask.bool()] = 0
    return grid_pt


@torch.no_grad()
def spvs_coarse(data, config):
    """
    Update:
        data (dict): {
            "conf_matrix_gt": [N, hw0, hw1],
            'spv_b_ids': [M]
            'spv_i_ids': [M]
            'spv_j_ids': [M]
            'spv_w_pt0_i': [N, hw0, 2], in original image resolution
            'spv_pt1_i': [N, hw1, 2], in original image resolution
        }
        
    NOTE:
        - for scannet dataset, there're 3 kinds of resolution {i, c, f}
        - for megadepth dataset, there're 4 kinds of resolution {i, i_resize, c, f}
    """
    # 1. misc
    device = data['image0'].device
    N, _, H0, W0 = data['image0'].shape
    _, _, H1, W1 = data['image1'].shape
    scale = config['LOFTR']['RESOLUTION'][0]
    scale0 = scale * data['scale0'][:, None] if 'scale0' in data else scale
    scale1 = scale * data['scale1'][:, None] if 'scale0' in data else scale
    h0, w0, h1, w1 = map(lambda x: x // scale, [H0, W0, H1, W1])

    # 2. warp grids
    # create kpts in meshgrid and resize them to image resolution
    grid_pt0_c = create_meshgrid(h0, w0, False, device).reshape(1, h0*w0, 2).repeat(N, 1, 1)    # [N, hw, 2]
    grid_pt0_i = scale0 * grid_pt0_c
    grid_pt1_c = create_meshgrid(h1, w1, False, device).reshape(1, h1*w1, 2).repeat(N, 1, 1)
    grid_pt1_i = scale1 * grid_pt1_c

    # mask padded region to (0, 0), so no need to manually mask conf_matrix_gt
    if 'mask0' in data:
        grid_pt0_i = mask_pts_at_padded_regions(grid_pt0_i, data['mask0'])
        grid_pt1_i = mask_pts_at_padded_regions(grid_pt1_i, data['mask1'])

    # warp kpts bi-directionally and resize them to coarse-level resolution
    # (no depth consistency check, since it leads to worse results experimentally)
    # (unhandled edge case: points with 0-depth will be warped to the left-up corner)
    if 'depth0' in data:
        _, w_pt0_i = warp_kpts(grid_pt0_i, data['depth0'], data['depth1'], data['T_0to1'], data['K0'], data['K1'])
        _, w_pt1_i = warp_kpts(grid_pt1_i, data['depth1'], data['depth0'], data['T_1to0'], data['K1'], data['K0'])
    else:
        w_pt0_i = warp_points_batch(grid_pt0_i, homographies=data['H_0to1'])
        w_pt1_i = warp_points_batch(grid_pt1_i, homographies=data['H_1to0'])

    w_pt0_c = w_pt0_i / scale1
    w_pt1_c = w_pt1_i / scale0

    # 3. check if mutual nearest neighbor
    w_pt0_c_round = w_pt0_c[:, :, :].round().long()
    nearest_index1 = w_pt0_c_round[..., 0] + w_pt0_c_round[..., 1] * w1
    w_pt1_c_round = w_pt1_c[:, :, :].round().long()
    nearest_index0 = w_pt1_c_round[..., 0] + w_pt1_c_round[..., 1] * w0

    # corner case: out of boundary
    def out_bound_mask(pt, w, h):
        return (pt[..., 0] < 0) + (pt[..., 0] >= w) + (pt[..., 1] < 0) + (pt[..., 1] >= h)
    nearest_index1[out_bound_mask(w_pt0_c_round, w1, h1)] = 0
    nearest_index0[out_bound_mask(w_pt1_c_round, w0, h0)] = 0

    loop_back = torch.stack([nearest_index0[_b][_i] for _b, _i in enumerate(nearest_index1)], dim=0)
    correct_0to1 = loop_back == torch.arange(h0*w0, device=device)[None].repeat(N, 1)
    correct_0to1[:, 0] = False  # ignore the top-left corner

    # 4. construct a gt conf_matrix
    conf_matrix_gt = torch.zeros(N, h0*w0, h1*w1, device=device)
    b_ids, i_ids = torch.where(correct_0to1 != 0)
    j_ids = nearest_index1[b_ids, i_ids]

    conf_matrix_gt[b_ids, i_ids, j_ids] = 1
    data.update({'conf_matrix_gt': conf_matrix_gt})

    # 5. save coarse matches(gt) for training fine level
    if len(b_ids) == 0:
        logger.warning(f"No groundtruth coarse match found for: {data['pair_names']}")
        # this won't affect fine-level loss calculation
        b_ids = torch.tensor([0], device=device)
        i_ids = torch.tensor([0], device=device)
        j_ids = torch.tensor([0], device=device)

    data.update({
        'spv_b_ids': b_ids,
        'spv_i_ids': i_ids,
        'spv_j_ids': j_ids
    })

    # 6. save intermediate results (for fast fine-level computation)
    data.update({
        'spv_w_pt0_i': w_pt0_i,
        'spv_pt1_i': grid_pt1_i
    })


def compute_supervision_coarse(data, config):
    assert len(set(data['dataset_name'])) == 1, "Do not support mixed datasets training!"
    data_source = data['dataset_name'][0]
    if data_source.lower() in ['scannet', 'megadepth', 'oxford']:
        spvs_coarse(data, config)
    else:
        raise ValueError(f'Unknown data source: {data_source}')


##############  ↓  Fine-Level supervision  ↓  ##############

@torch.no_grad()
def spvs_fine(data, config):
    """
    Update:
        data (dict):{
            "expec_f_gt": [M, 2]}
    """
    # 1. misc
    # w_pt0_i, pt1_i = data.pop('spv_w_pt0_i'), data.pop('spv_pt1_i')
    w_pt0_i, pt1_i = data['spv_w_pt0_i'], data['spv_pt1_i']
    scale = config['LOFTR']['RESOLUTION'][1]
    radius = config['LOFTR']['FINE_WINDOW_SIZE'] // 2

    # 2. get coarse prediction
    b_ids, i_ids, j_ids = data['b_ids'], data['i_ids'], data['j_ids']

    # 3. compute gt
    scale = scale * data['scale1'][b_ids] if 'scale0' in data else scale
    # `expec_f_gt` might exceed the window, i.e. abs(*) > 1, which would be filtered later
    expec_f_gt = (w_pt0_i[b_ids, i_ids] - pt1_i[b_ids, j_ids]) / scale / radius  # [M, 2]
    data.update({"expec_f_gt": expec_f_gt})

@torch.no_grad()
def spvs_fine2_bak(data, config):
    """
    Update:
        data (dict): {
            "conf_matrix_gt": [N, hw0, hw1],
            'spv_b_ids': [M]
            'spv_i_ids': [M]
            'spv_j_ids': [M]
            'spv_w_pt0_i': [N, hw0, 2], in original image resolution
            'spv_pt1_i': [N, hw1, 2], in original image resolution
        }

    NOTE:
        - for scannet dataset, there're 3 kinds of resolution {i, c, f}
        - for megadepth dataset, there're 4 kinds of resolution {i, i_resize, c, f}
    """
    # 1. misc
    device = data['image0'].device
    N, _, H0, W0 = data['image0'].shape
    _, _, H1, W1 = data['image1'].shape

    scale = config['LOFTR']['RESOLUTION'][1]
    h0, w0, h1, w1 = map(lambda x: x // scale, [H0, W0, H1, W1])
    W = data['W']
    # 2. warp grids
    # create kpts in meshgrid and resize them to image resolution
    center_kp0 = data['mkpts0_c']
    center_kp1 = data['mkpts1_c']

    grid_w = create_meshgrid(W, W, False, device).reshape(1, W * W, 2).repeat(center_kp1.size(0), 1, 1)  # [N, hw, 2]
    grid_w -= W // 2
    coarse_scale = data['hw0_i'][0] / data['hw0_c'][0]
    coarse_scale0 = coarse_scale * data['scale0'][data['b_ids']] if 'scale0' in data else coarse_scale
    coarse_scale1 = coarse_scale * data['scale1'][data['b_ids']] if 'scale1' in data else coarse_scale
    scale_c_f = data['hw0_f'][0] / data['hw0_c'][0]
    center_kp0 = center_kp0 / coarse_scale0 * scale_c_f
    center_kp1 = center_kp1 / coarse_scale1 * scale_c_f
    center_kp0 = center_kp0.unsqueeze(1).repeat([1, W*W, 1])
    center_kp1 = center_kp1.unsqueeze(1).repeat([1, W * W, 1])
    kpts0 = center_kp0 + grid_w
    kpts1 = center_kp1 + grid_w

    fine_scale = data['hw0_i'][0] / data['hw0_f'][0]
    fine_scale0 = (fine_scale * data['scale0'][data['b_ids']]).unsqueeze(1).repeat([1, W*W, 1]) if 'scale0' in data else fine_scale
    fine_scale1 = (fine_scale * data['scale1'][data['b_ids']]).unsqueeze(1).repeat([1, W*W, 1]) if 'scale1' in data else fine_scale
    kpts0_raw = kpts0 * fine_scale0
    kpts1_raw = kpts1 * fine_scale1

    # warp kpts bi-directionally and resize them to coarse-level resolution
    # (no depth consistency check, since it leads to worse results experimentally)
    # (unhandled edge case: points with 0-depth will be warped to the left-up corner)
    if 'depth0' in data:
        m, l, _ = kpts0_raw.shape
        kpts0_raw_tmp = kpts0_raw.reshape(1, m * l, 2)
        kpts1_raw_tmp = kpts1_raw.reshape(1, m * l, 2)
        mk0, w_pt0_i = warp_kpts(kpts0_raw_tmp, data['depth0'], data['depth1'], data['T_0to1'], data['K0'], data['K1'])
        mk1, w_pt1_i = warp_kpts(kpts1_raw_tmp, data['depth1'], data['depth0'], data['T_1to0'], data['K1'], data['K0'])
        w_pt0_i[~mk0] = -100000
        w_pt1_i[~mk1] = -100000
        w_pt0_i = w_pt0_i.view(m, l, 2)
        w_pt1_i = w_pt1_i.view(m, l, 2)
    else:
        w_pt0_i = warp_points_batch(kpts0_raw, homographies=data['H_0to1'])
        w_pt1_i = warp_points_batch(kpts1_raw, homographies=data['H_1to0'])

    comp_wk0 = w_pt0_i / fine_scale0
    comp_wk1 = w_pt1_i / fine_scale1
    comp_rk0 = kpts0_raw / fine_scale0
    comp_rk1 = kpts1_raw / fine_scale1
    # kp_dis_map1 = torch.cat(
    #     [torch.sqrt((((comp_wk0[i][:, None] - comp_rk1[i]) ** 2).sum(-1))).unsqueeze(0) for i in
    #      range(len(kpts1_raw))])
    kp_dis_map0 = torch.cat(
        [torch.sqrt((((comp_wk1[i][:, None] - comp_rk0[i]) ** 2).sum(-1))).unsqueeze(0) for i in
         range(len(kpts0_raw))])


    require_one_ind0 = kp_dis_map0.view(-1, W * W * W * W).argmin(1)
    non_mask = torch.zeros([kp_dis_map0.shape[0], W * W * W * W]).to(kp_dis_map0)
    non_mask[range(kp_dis_map0.shape[0]), require_one_ind0] = 1
    kp_dis_map0 = kp_dis_map0 * non_mask.view(-1, W * W, W * W)

    # require_one_ind1 = kp_dis_map1.view(-1, W * W * W * W).argmin(1)
    # non_mask = torch.zeros([kp_dis_map1.shape[0], W * W * W * W]).to(kp_dis_map1)
    # non_mask[range(kp_dis_map1.shape[0]), require_one_ind1] = 1
    # kp_dis_map1 = kp_dis_map1 * non_mask.view(-1, W * W, W * W)
    # label = (kp_dis_map0 <= 1) * (kp_dis_map1 <= 1) * (kp_dis_map0 > 0) * (kp_dis_map1 > 0)
    label = (kp_dis_map0 <= 0.001) * (kp_dis_map0 > 0)
    # label = label \
    #        * (kp_dis_map0 == kp_dis_map0.min(dim=2, keepdim=True)[0]) \
    #        * (kp_dis_map0 == kp_dis_map0.min(dim=1, keepdim=True)[0])
    #
    # label = label \
    #         * (kp_dis_map1 == kp_dis_map1.min(dim=2, keepdim=True)[0]) \
    #         * (kp_dis_map1 == kp_dis_map1.min(dim=1, keepdim=True)[0])
    # import matplotlib.pyplot as plt
    # bid, mid, nid = torch.where(label)
    # plt.subplot(121)
    # plt.imshow(data['image0'].cpu()[0][0])
    # plt.scatter(comp_rk0[bid, mid, 0].cpu() * scale, comp_rk0[bid, mid, 1].cpu() * scale, c='r', s=1)
    # plt.subplot(122)
    # plt.imshow(data['image1'].cpu()[0][0])
    # plt.scatter(comp_rk1[bid, nid, 0].cpu() * scale, comp_rk1[bid, nid, 1].cpu() * scale, c='r', s=1)
    # plt.show()
    # import matplotlib.pyplot as plt
    # bid, mid, nid = torch.where(label)
    # plt.subplot(121)
    # plt.imshow(data['image0'].cpu()[0][0])
    # plt.scatter((data['mkpts0_f'].cpu() / fine_scale0[:, 0].cpu())[:, 0] * scale,
    #             (data['mkpts0_f'].cpu() / fine_scale0[:, 0].cpu())[:, 1] * scale, c='r', s=1)
    # plt.subplot(122)
    # plt.imshow(data['image1'].cpu()[0][0])
    # plt.scatter((data['mkpts1_f'].cpu() / fine_scale1[:, 0].cpu())[:, 0] * scale,
    #             (data['mkpts1_f'].cpu() / fine_scale1[:, 0].cpu())[:, 1] * scale, c='r', s=1)
    # plt.show()
    data.update({'conf_matrix_fine_gt': label})

@torch.no_grad()
def spvs_fine2(data, config):
    """
    Update:
        data (dict): {
            "conf_matrix_gt": [N, hw0, hw1],
            'spv_b_ids': [M]
            'spv_i_ids': [M]
            'spv_j_ids': [M]
            'spv_w_pt0_i': [N, hw0, 2], in original image resolution
            'spv_pt1_i': [N, hw1, 2], in original image resolution
        }

    NOTE:
        - for scannet dataset, there're 3 kinds of resolution {i, c, f}
        - for megadepth dataset, there're 4 kinds of resolution {i, i_resize, c, f}
    """
    # 1. misc
    device = data['image0'].device
    N, _, H0, W0 = data['image0'].shape
    _, _, H1, W1 = data['image1'].shape

    scale = config['LOFTR']['RESOLUTION'][1]
    h0, w0, h1, w1 = map(lambda x: x // scale, [H0, W0, H1, W1])
    W = data['W']
    # 2. warp grids
    # create kpts in meshgrid and resize them to image resolution
    center_kp0 = data['mkpts0_c']
    center_kp1 = data['mkpts1_c']

    grid_w = create_meshgrid(W, W, False, device).reshape(1, W * W, 2).repeat(center_kp1.size(0), 1, 1)  # [N, hw, 2]
    grid_w -= W // 2
    # coarse_scale = 1
    coarse_scale = data['hw0_i'][0] // data['hw0_c'][0]
    coarse_scale0 = coarse_scale * data['scale0'][data['b_ids']] if 'scale0' in data else coarse_scale
    coarse_scale1 = coarse_scale * data['scale1'][data['b_ids']] if 'scale1' in data else coarse_scale
    scale_c_f = data['hw0_f'][0] // data['hw0_c'][0]
    center_kp0 = center_kp0 / coarse_scale0 * scale_c_f
    center_kp1 = center_kp1 / coarse_scale1 * scale_c_f
    center_kp0 = center_kp0.unsqueeze(1).repeat([1, W*W, 1])
    center_kp1 = center_kp1.unsqueeze(1).repeat([1, W * W, 1])
    kpts0 = center_kp0 + grid_w
    kpts1 = center_kp1 + grid_w

    # fine_scale = 1
    fine_scale = data['hw0_i'][0] // data['hw0_f'][0]
    fine_scale0 = (fine_scale * data['scale0'][data['b_ids']]).unsqueeze(1).repeat([1, W*W, 1]) if 'scale0' in data else fine_scale
    fine_scale1 = (fine_scale * data['scale1'][data['b_ids']]).unsqueeze(1).repeat([1, W*W, 1]) if 'scale1' in data else fine_scale
    kpts0_raw = kpts0 * fine_scale0
    kpts1_raw = kpts1 * fine_scale1

    # warp kpts bi-directionally and resize them to coarse-level resolution
    # (no depth consistency check, since it leads to worse results experimentally)
    # (unhandled edge case: points with 0-depth will be warped to the left-up corner)
    if 'depth0' in data:
        m, l, _ = kpts0_raw.shape
        kpts0_raw_tmp = kpts0_raw.reshape(1, m * l, 2)
        kpts1_raw_tmp = kpts1_raw.reshape(1, m * l, 2)
        mk0, w_pt0_i = warp_kpts(kpts0_raw_tmp, data['depth0'], data['depth1'], data['T_0to1'], data['K0'], data['K1'])
        mk1, w_pt1_i = warp_kpts(kpts1_raw_tmp, data['depth1'], data['depth0'], data['T_1to0'], data['K1'], data['K0'])
        w_pt0_i[~mk0] = -100000
        w_pt1_i[~mk1] = -100000
        w_pt0_i = w_pt0_i.view(m, l, 2)
        w_pt1_i = w_pt1_i.view(m, l, 2)
    else:
        w_pt0_i = warp_points_batch(kpts0_raw, homographies=data['H_0to1'])
        w_pt1_i = warp_points_batch(kpts1_raw, homographies=data['H_1to0'])

    comp_wk0 = w_pt0_i / fine_scale0
    comp_wk1 = w_pt1_i / fine_scale1
    comp_rk0 = kpts0_raw / fine_scale0
    comp_rk1 = kpts1_raw / fine_scale1
    # kp_dis_map1 = torch.cat(
    #     [torch.sqrt((((comp_wk0[i][:, None] - comp_rk1[i]) ** 2).sum(-1))).unsqueeze(0) for i in
    #      range(len(kpts1_raw))])
    kp_dis_map0 = torch.cat(
        [torch.sqrt((((w_pt0_i[i][:, None] - kpts1_raw[i]) ** 2).sum(-1))).unsqueeze(0) for i in
         range(len(kpts0_raw))])


    require_one_ind0 = kp_dis_map0.view(-1, W * W * W * W).argmin(1)
    non_mask = torch.zeros([kp_dis_map0.shape[0], W * W * W * W]).to(kp_dis_map0)
    non_mask[range(kp_dis_map0.shape[0]), require_one_ind0] = 1
    kp_dis_map0 = kp_dis_map0 * non_mask.view(-1, W * W, W * W)

    # require_one_ind1 = kp_dis_map1.view(-1, W * W * W * W).argmin(1)
    # non_mask = torch.zeros([kp_dis_map1.shape[0], W * W * W * W]).to(kp_dis_map1)
    # non_mask[range(kp_dis_map1.shape[0]), require_one_ind1] = 1
    # kp_dis_map1 = kp_dis_map1 * non_mask.view(-1, W * W, W * W)
    # label = (kp_dis_map0 <= 1) * (kp_dis_map1 <= 1) * (kp_dis_map0 > 0) * (kp_dis_map1 > 0)
    label = (kp_dis_map0 <= 3) * (kp_dis_map0 > 0)
    # label = label \
    #        * (kp_dis_map0 == kp_dis_map0.min(dim=2, keepdim=True)[0]) \
    #        * (kp_dis_map0 == kp_dis_map0.min(dim=1, keepdim=True)[0])
    #
    # label = label \
    #         * (kp_dis_map1 == kp_dis_map1.min(dim=2, keepdim=True)[0]) \
    #         * (kp_dis_map1 == kp_dis_map1.min(dim=1, keepdim=True)[0])
    # import matplotlib.pyplot as plt
    # bid, mid, nid = torch.where(label)
    # plt.subplot(121)
    # plt.imshow(data['image0'].cpu()[0][0])
    # plt.scatter((comp_rk0 * fine_scale)[bid, mid, 0].cpu(), (comp_rk0 * fine_scale)[bid, mid, 1].cpu(), c='r', s=1)
    # plt.subplot(122)
    # plt.imshow(data['image1'].cpu()[0][0])
    # plt.scatter((comp_rk1 * fine_scale)[bid, nid, 0].cpu(), (comp_rk1 * fine_scale)[bid, nid, 1].cpu(), c='r', s=1)
    # plt.show()
    # import matplotlib.pyplot as plt
    # bid, mid, nid = torch.where(label)
    # plt.subplot(121)
    # plt.imshow(data['image0'].cpu()[0][0])
    # plt.scatter((data['mkpts0_f'].cpu() / fine_scale0[:, 0].cpu())[:, 0],
    #             (data['mkpts0_f'].cpu() / fine_scale0[:, 0].cpu())[:, 1], c='r', s=1)
    # plt.subplot(122)
    # plt.imshow(data['image1'].cpu()[0][0])
    # plt.scatter((data['mkpts1_f'].cpu() / fine_scale1[:, 0].cpu())[:, 0],
    #             (data['mkpts1_f'].cpu() / fine_scale1[:, 0].cpu())[:, 1], c='r', s=1)
    # plt.show()
    data.update({'conf_matrix_fine_gt': label})

def compute_supervision_fine(data, config):
    data_source = data['dataset_name'][0]
    if data_source.lower() in ['scannet', 'megadepth', 'oxford']:
        spvs_fine2(data, config)
    else:
        raise NotImplementedError
