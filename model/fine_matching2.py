from typing import Dict

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.geometry.subpix import dsnt
from kornia.utils.grid import create_meshgrid

from utils.common_utils import arange_like


class FineMatching2(nn.Module):
    """FineMatching with s2d paradigm"""

    def __init__(self, temperature=0.1, thr=0.1):
        super().__init__()
        self.temperature = temperature
        self.thr = thr

    def forward(self, feat_f0, feat_f1, data: Dict[str, torch.Tensor]):
        """
        Args:
            feat0 (torch.Tensor): [M, WW, C]
            feat1 (torch.Tensor): [M, WW, C]
            data (dict)
        Update:
            data (dict):{
                'expec_f' (torch.Tensor): [M, 3],
                'mkpts0_f' (torch.Tensor): [M, 2],
                'mkpts1_f' (torch.Tensor): [M, 2]}
        """
        M, WW, C = feat_f0.shape
        if M == 0:
            # assert self.training == False, "M is always >0, when training, see coarse_matching.py"
            # logger.warning('No matches found in coarse-level.')
            data.update({
                'fine_matrix': torch.empty(0, WW, WW, device=feat_f0.device),
                'mkpts0_f': data['mkpts0_c'],
                'mkpts1_f': data['mkpts1_c'],
            })
            return

        W = int(math.sqrt(WW))
        scale = int(data['hw0_i'][0] // data['hw0_f'][0])
        b_ids = data['b_ids']
        bs = data['image0'].size(0)
        # feat_f0 = F.upsample(feat_f0.view(-1, W, W, C).permute(0, 3, 1, 2), scale_factor=scale, align_corners=True,
        #                      mode='bilinear').permute(0, 2, 3, 1).contiguous().view(M, -1, C)
        # feat_f1 = F.upsample(feat_f1.view(-1, W, W, C).permute(0, 3, 1, 2), scale_factor=scale, align_corners=True,
        #                      mode='bilinear').permute(0, 2, 3, 1).contiguous().view(M, -1, C)
        feat_f0, feat_f1 = feat_f0 / feat_f0.shape[-1] ** .5, feat_f1 / feat_f1.shape[-1] ** .5
        # feat_c0, feat_c1 = map(lambda feat: feat / feat.shape[-1]**.5,
        #                        [feat_c0, feat_c1])
        INF = 1e9
        # if self.match_type == 'dual_softmax':
        sim_matrix = torch.einsum("nlc,nsc->nls", feat_f0,
                                  feat_f1) / self.temperature

        conf_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)

        data['fine_matrix'] = conf_matrix
        self.get_fine_match(conf_matrix, data, W)

    @torch.no_grad()
    def get_fine_match(self, conf_matrix, data, W):
        conf_matrix = conf_matrix

        center_kp0 = data['mkpts0_c']
        center_kp1 = data['mkpts1_c']
        _device = conf_matrix.device
        # 1. confidence thresholding
        mask = conf_matrix > self.thr
        b = mask.shape[0]

        # 2. mutual nearest
        mask = mask \
               * (conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0]) \
               * (conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0])
        require_one_ind = conf_matrix.view(-1, W * W * W * W).argmax(1)
        non_mask = torch.zeros([mask.shape[0], W*W*W*W]).to(mask)
        non_mask[range(mask.shape[0]), require_one_ind] = 1
        mask = mask * non_mask.view(-1, W*W, W*W)
        fine_b_ids = data['b_ids'].unsqueeze(-1).unsqueeze(-1).repeat([1, W*W, W*W])
        fine_b_ids = fine_b_ids[mask]
        # 3. find all valid coarse matches
        # this only works when at most one `True` in each row
        mask_v, all_j_ids = mask.max(dim=2)
        b_ids, i_ids = torch.where(mask_v)
        j_ids = all_j_ids[b_ids, i_ids]
        mconf = conf_matrix[b_ids, i_ids, j_ids]
        fine_matches = {'b_ids': b_ids, 'i_ids': i_ids, 'j_ids': j_ids}
        # 4. Update with matches in original image resolution
        coarse_scale = data['hw0_i'][0] / data['hw0_c'][0]
        # coarse_scale = 1
        coarse_scale0 = coarse_scale * data['scale0'][data['b_ids']] if 'scale0' in data else coarse_scale
        coarse_scale1 = coarse_scale * data['scale1'][data['b_ids']] if 'scale1' in data else coarse_scale
        scale_c_f = data['hw0_f'][0] / data['hw0_c'][0]
        # scale_c_f = 1
        center_kp0 = center_kp0 / coarse_scale0 * scale_c_f
        center_kp1 = center_kp1 / coarse_scale1 * scale_c_f
        # print(b_ids, center_kp0.shape)
        mkpts0_f = torch.stack(
            [i_ids % W - W // 2, i_ids // W - W // 2],
            dim=1) + center_kp0[b_ids]
        mkpts1_f = torch.stack(
            [j_ids % W - W // 2, j_ids // W - W // 2],
            dim=1) + center_kp1[b_ids]

        fine_scale = data['hw0_i'][0] / data['hw0_f'][0]
        # fine_scale = 1
        fine_scale0 = fine_scale * data['scale0'][fine_b_ids] if 'scale0' in data else fine_scale
        fine_scale1 = fine_scale * data['scale1'][fine_b_ids] if 'scale1' in data else fine_scale
        mkpts1_f = mkpts1_f * fine_scale1
        mkpts0_f = mkpts0_f * fine_scale0


        # These matches is the current prediction (for visualization)
        data.update({
            'm_bids': fine_b_ids,  # mconf == 0 => gt matches
            'mkpts0_f': mkpts0_f,
            'mkpts1_f': mkpts1_f,
            'mconf': mconf
        })

        return fine_matches


    def fine_matching(self, conf_matrix_list, kp_windows_list0, kp_windows_list1, bin=True):
        # kp.shape : n, w, 2 (w: window_size)
        B = len(conf_matrix_list)
        fine_kps_list0 = []
        fine_kps_list1 = []
        scores_list = []
        for i in range(B):
            conf_matrix = conf_matrix_list[i]
            if bin:
                conf_matrix = conf_matrix[:, :-1, :-1]
            fine_kps0 = torch.empty([0, 2]).to(conf_matrix.device)
            fine_kps1 = torch.empty([0, 2]).to(conf_matrix.device)
            scores = torch.empty([0]).to(conf_matrix.device)
            kp_windows0, kp_windows1 = kp_windows_list0[i], kp_windows_list1[i]
            result = self.get_matching_result(conf_matrix[:, :, :], self.fine_thresh, topK=self.topK)
            matches0 = result['matches0']
            if matches0 is not None:
                valid0 = matches0 > -1
                ind = torch.where(matches0 > -1)
                fine_kps0 = []
                fine_kps1 = []
                if len(ind) > 0:
                    match_scores = result['matching_scores0']
                    line = ind[0]
                    column = matches0[valid0]
                    fine_kps0 = kp_windows0[valid0]
                    fine_kps1 = kp_windows1[line, column, :]
                    scores = match_scores[valid0]
            fine_kps_list0.append(fine_kps0.data)
            fine_kps_list1.append(fine_kps1.data)
            scores_list.append(scores.data)

        return fine_kps_list0, fine_kps_list1, scores_list

    def get_matching_result(scores, matching_thresh, knn=False, topK=None):
        # print(scores.shape)
        if len(scores.shape) != 3 or scores.shape[0] == 0:
            # device = scores.device
            return {
                'matches0': None,  # use -1 for invalid match
                'matches1': None,  # use -1 for invalid match
                'matching_scores0': None,
                'matching_scores1': None,
                'scores': scores,
            }
        non = scores.new_tensor(-1)
        if knn:
            try:
                val, ind = torch.topk(scores, 2, dim=2, largest=True)
                val = val.squeeze()
                ind = ind.squeeze()
                valid0 = (val[:, 0] * matching_thresh > val[:, 1])
                mscores0 = torch.where(valid0, val[:, 0], non)
                indices0 = torch.where(valid0, ind[:, 0].float(), non)
                indices1 = -torch.ones(scores.shape[-1]).to(indices0)
                indices1[indices0[indices0 != -1].long()] = torch.where(indices0 != -1)[0].float()
                indices1 = indices1.long()
                indices0 = indices0.long()
                mscores1 = mscores0

            except Exception:
                knn = False
        if not knn:
            max0 = scores.max(2)
            max1 = scores.max(1)
            indices0, indices1 = max0.indices, max1.indices
            mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
            mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
            mscores0 = torch.where(mutual0, max0.values, non)
            mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), non)
            valid0 = mutual0 & (mscores0 > matching_thresh)
            valid1 = mutual1 & valid0.gather(1, indices1)
            non = torch.tensor(-1).to(scores.device)
            indices0 = torch.where(valid0, indices0, non)
            indices1 = torch.where(valid1, indices1, non)

        # good_ratio =

        # good = valid0.sum()
        # good_ratio = valid0.sum() / max(1, len(valid0))
        if topK is not None:
            val, ind = torch.topk(mscores0, topK, dim=1, largest=True)

            def select(inp_tensor, ind, val=None):
                l, k = inp_tensor.shape
                arange = torch.arange(l).to(ind)
                arange = arange[:, None].repeat([1, ind.shape[-1]])
                inp_tensor = inp_tensor.clone()
                ind = arange * k + ind
                inp_tensor = inp_tensor.view(-1)
                ind = ind.view(-1)
                if val is None:
                    return inp_tensor[ind]
                inp_tensor[ind] = val
                inp_tensor = inp_tensor.view(l, k)
                return inp_tensor

            select_indices1_val = select(indices1, ind)
            select_indices0_val = select(indices0, ind)
            select_mscores1_val = select(mscores1, ind)

            select_mscores0 = mscores0.clone()
            select_mscores1 = mscores1.clone()
            select_indices0 = indices0.clone()
            select_indices1 = indices1.clone()
            select_mscores0[:] = -1
            select_mscores1[:] = -1
            select_indices0[:] = -1
            select_indices1[:] = -1

            mscores0 = select(select_mscores0, ind, val.view(-1))
            mscores1 = select(select_mscores1, ind, select_mscores1_val)
            indices0 = select(select_indices0, ind, select_indices0_val)
            indices1 = select(select_indices1, ind, select_indices1_val)

        result = {
            'matches0': indices0,  # use -1 for invalid match
            'matches1': indices1,  # use -1 for invalid match
            'matching_scores0': mscores0,
            'matching_scores1': mscores1,
            'scores': scores,
        }
        return result
