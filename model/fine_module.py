import torch
from torch import nn

import torch.nn.functional as F

from model.transformer.transformer import LocalFeatureTransformer
from utils.common_utils import generate_window, sample_descriptors, generate_conf

import torch
import torch.nn as nn
from kornia.geometry.subpix import dsnt
from kornia.utils.grid import create_meshgrid


# class FinePreprocess(nn.Module):
#     def __init__(self, loftr_config):
#         super().__init__()
#
#         self.loftr_config = loftr_config
#         self.cat_c_feat = loftr_config['fine_concat_coarse_feat']
#         self.W = self.loftr_config['window_size']
#
#         self.d_model = self.loftr_config['d_model']
#         self.scale_c = self.loftr_config['scale_c']
#         self.scale_f = self.loftr_config['scale_f']
#         if self.cat_c_feat:
#             self.down_proj = nn.Linear(self.d_model, self.d_model, bias=True)
#             self.merge_feat = nn.Linear(2*self.d_model, self.d_model, bias=True)
#
#         self._reset_parameters()
#
#     def _reset_parameters(self):
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")
#
#     def forward(self, feat_f0, feat_f1, feat_c0, feat_c1, data):
#         W = self.W
#         stride = self.scale_c // self.scale_f
#
#         data.update({'W': W})
#         if data['b_ids'].shape[0] == 0:
#             feat0 = torch.empty(0, self.W**2, self.d_model, device=feat_f0.device)
#             feat1 = torch.empty(0, self.W**2, self.d_model, device=feat_f0.device)
#             return feat0, feat1
#
#         # 1. unfold(crop) all local windows
#         feat_f0_unfold = F.unfold(feat_f0, kernel_size=(W, W), stride=stride, padding=W//2)
#         feat_f0_unfold = rearrange(feat_f0_unfold, 'n (c ww) l -> n l ww c', ww=W**2)
#         feat_f1_unfold = F.unfold(feat_f1, kernel_size=(W, W), stride=stride, padding=W//2)
#         feat_f1_unfold = rearrange(feat_f1_unfold, 'n (c ww) l -> n l ww c', ww=W**2)
#
#         # 2. select only the predicted matches
#         feat_f0_unfold = feat_f0_unfold[data['b_ids'], data['i_ids']]  # [n, ww, cf]
#         feat_f1_unfold = feat_f1_unfold[data['b_ids'], data['j_ids']]
#
#         # option: use coarse-level loftr feature as context: concat and linear
#         if self.cat_c_feat:
#             feat_c_win = self.down_proj(torch.cat([feat_c0[data['b_ids'], data['i_ids']],
#                                                    feat_c1[data['b_ids'], data['j_ids']]], 0))  # [2n, c]
#             feat_cf_win = self.merge_feat(torch.cat([
#                 torch.cat([feat_f0_unfold, feat_f1_unfold], 0),  # [2n, ww, cf]
#                 repeat(feat_c_win, 'n c -> n ww c', ww=W**2),  # [2n, ww, cf]
#             ], -1))
#             feat_f0_unfold, feat_f1_unfold = torch.chunk(feat_cf_win, 2, dim=0)
#
#         return feat_f0_unfold, feat_f1_unfold

class Fine_Net(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.window_size = config['window_size']
        self.des_transformer = LocalFeatureTransformer(config, config['fine_layer_names'], config['fine_d_model'], linear=True)
        self.bin_score = nn.Parameter(
            torch.tensor(1., requires_grad=True))
        self.d_model = config['d_model']
        self.fine_d_model = config['fine_d_model']
        self.scale_c = None
        self.scale_d = None
        self.tempr = 1
        self.merge_desc = nn.Linear(self.fine_d_model + self.d_model, self.fine_d_model, bias=True)
        # self.fp = FinePreprocess(loftr_config)
    # def sample_descripors_windows(self, kp_windows_list, desc_map_up, scale):
    #     kps_dense_list = [kps.reshape(-1, 2) for kps in kp_windows_list]
    #
    #     desc_windows_list = sample_descriptors(kps_dense_list, desc_map_up, s=scale)
    #     c = desc_map_up.shape[1]
    #     desc_windows_list = [desc_dense.reshape(kps.shape[0], kps.shape[1], c) for desc_dense, kps in zip(desc_windows_list, kp_windows_list)]
    #     return desc_windows_list
    def sample_descripors_windows(self, kp_windows_list, desc_map_up, desc_map, norm=False):
        kps_dense_list = [kps.reshape(-1, 2) for kps in kp_windows_list]
        c = desc_map_up.shape[1]
        desc_windows_list = sample_descriptors(kps_dense_list, desc_map_up, s=self.scale_d, norm=norm)
        desc_windows_list = [desc_dense.reshape(kps.shape[0], kps.shape[1], c) for desc_dense, kps in
                             zip(desc_windows_list, kp_windows_list)]
        return desc_windows_list
        # desc_windows_list_c = sample_descriptors(kps_dense_list, desc_map, s=self.scale_c, norm=norm)
        # desc_windows_list = [self.merge_desc(torch.cat((desc, desc_c), -1)) for desc, desc_c in
        #                      zip(desc_windows_list, desc_windows_list_c)]
        #
        #
        # desc_windows_list = [desc_dense.reshape(kps.shape[0], kps.shape[1], c) for desc_dense, kps in zip(desc_windows_list, kp_windows_list)]
        # return desc_windows_list
    def forward(self, desc_map_up0, desc_map_up1, desc_map0, desc_map1, keypoint_list0, keypoint_list1, data):
        H0, W0 = data['image0'].shape[2:]
        H1, W1 = data['image1'].shape[2:]
        self.scale_c = data['desc_scale']
        self.scale_d = data['desc_up_scale']
        kp_windows_list0, masks_list0 = generate_window(keypoint_list0, (H0, W0), window_size=self.window_size, scale=self.scale_d)
        kp_windows_list1, masks_list1 = generate_window(keypoint_list1, (H1, W1), window_size=self.window_size, scale=self.scale_d)

        desc_windows_list0 = self.sample_descripors_windows(kp_windows_list0, desc_map_up0, desc_map0)
        desc_windows_list1 = self.sample_descripors_windows(kp_windows_list1, desc_map_up1, desc_map1)
        matrix_list = []
        for i in range(len(keypoint_list0)):
            desc_windows0, desc_windows1 = desc_windows_list0[i], desc_windows_list1[i]
            mask0, mask1 = masks_list0[i], masks_list1[i]
            # desc_windows0_norm = F.normalize(desc_windows0, p=2, dim=-1)
            # desc_windows1_norm = F.normalize(desc_windows1, p=2, dim=-1)
            # cos_matrix = torch.einsum("nlc,nsc->nls", desc_windows0,
            #                           desc_windows1)  # cos distance, see sample_descriptors
            if len(desc_windows0) >= 1:
                desc_windows0, desc_windows1 = self.des_transformer(desc_windows0, desc_windows1, mask0, mask1)
            conf_matrix = generate_conf(self.bin_score, desc_windows0.permute(0, 2, 1), desc_windows1.permute(0, 2, 1), self.d_model, self.tempr, bin=True)
            mask = mask0[:, :, None] * mask1[:, None, :]
            temp = conf_matrix[:, :-1, :-1]
            temp[mask == 0] = 0*temp[mask == 0]
            conf_matrix[:, :-1, :-1] = temp
            # conf_matrix[mask == 0] = 0
            conf_matrix = torch.clamp(conf_matrix, 1e-6, 1-1e-6)
            matrix_list.append(conf_matrix)

        return matrix_list, kp_windows_list0, kp_windows_list1

    # def forward(self, desc_map_up0, desc_map_up1, desc_map0, desc_map1, keypoint_list0, keypoint_list1, data):
    #     H0, W0 = data['image0'].shape[2:]
    #     H1, W1 = data['image1'].shape[2:]
    #     self.scale_c = data['desc_scale']
    #     self.scale_d = data['desc_up_scale']
    #     kp_windows_list0, masks_list0 = generate_window(keypoint_list0, (H0, W0), window_size=self.window_size,
    #                                                     scale=self.scale_d)
    #     kp_windows_list1, masks_list1 = generate_window(keypoint_list1, (H1, W1), window_size=self.window_size,
    #                                                     scale=self.scale_d)
    #
    #     desc_windows_list0 = self.sample_descripors_windows(kp_windows_list0, desc_map_up0, desc_map0, False)
    #     desc_windows_list1 = self.sample_descripors_windows(kp_windows_list1, desc_map_up1, desc_map1, False)
    #     expec_list = []
    #     for i in range(len(keypoint_list0)):
    #         desc_windows0, desc_windows1 = desc_windows_list0[i], desc_windows_list1[i]
    #         mask0, mask1 = masks_list0[i], masks_list1[i]
    #         if len(desc_windows0) >= 1:
    #             desc_windows0, desc_windows1 = self.des_transformer(desc_windows0, desc_windows1, mask0, mask1)
    #         else:
    #             expec = torch.empty(0, 3, device=desc_windows0.device)
    #             expec_list.append(expec)
    #             continue
    #
    #         desc0_picked = desc_windows0[:, self.window_size ** 2 // 2, :]
    #         sim_matrix = torch.einsum('mc,mrc->mr', desc0_picked, desc_windows1)
    #         sim_matrix[mask1==0] = -1e9
    #         softmax_temp = 1. / self.d_model ** .5
    #         W = self.window_size
    #         heatmap = torch.softmax(sim_matrix * softmax_temp, dim=1).view(-1, W, W)
    #         coords_normalized = dsnt.spatial_expectation2d(heatmap[None], True)[0]  # [M, 2]
    #         grid_normalized = create_meshgrid(W, W, True, heatmap.device).reshape(1, -1, 2)  # [1, WW, 2]
    #         # compute std over <x, y>
    #         var = torch.sum(grid_normalized ** 2 * heatmap.view(-1, W*W, 1), dim=1) - coords_normalized ** 2  # [M, 2]
    #         std = torch.sum(torch.sqrt(torch.clamp(var, min=1e-10)), -1)  # [M]  clamp needed for numerical stability
    #         expec = torch.cat([coords_normalized, std.unsqueeze(1)], -1)
    #         expec_list.append(expec)
    #
    #         # conf_matrix = generate_conf(self.bin_score, desc_windows0.permute(0, 2, 1), desc_windows1.permute(0, 2, 1), self.d_model, 0.01)
    #         # mask = mask0[:, :, None] * mask1[:, None, :]
    #         # temp = conf_matrix[:, :-1, :-1]
    #         # temp[mask == 0] = 0
    #         # conf_matrix[:, :-1, :-1] = temp
    #         # conf_matrix = torch.clamp(conf_matrix, 1e-6, 1-1e-6)
    #         # expec_list.append(conf_matrix)
    #
    #     return expec_list, kp_windows_list0, kp_windows_list1