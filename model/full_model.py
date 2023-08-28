from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.fine_matching2 import FineMatching2
# from einops.einops import rearrange

from model.loftr_src.loftr.backbone import build_backbone
from model.loftr_src.loftr.utils.position_encoding import PositionEncodingSine
from model.loftr_src.loftr.loftr_module import LocalFeatureTransformer, FinePreprocess
from model.loftr_src.loftr.utils.coarse_matching import CoarseMatching

from model.geo_module import GeoModule
from .geo_config import default_cfg


class GeoFormer(nn.Module):
    def __init__(self, loftr_config, geoformer_cfg=default_cfg):
        super().__init__()
        # Misc
        self.config = loftr_config

        # Modules
        # with torch.no_grad():
        self.backbone = build_backbone(loftr_config)
        self.loftr_coarse = LocalFeatureTransformer(loftr_config['coarse'])
        self.pos_encoding = PositionEncodingSine(
            loftr_config['coarse']['d_model'],
            temp_bug_fix=loftr_config['coarse']['temp_bug_fix'])
        loftr_config['match_coarse']['thr'] = geoformer_cfg['coarse_thr']
        self.coarse_matching = CoarseMatching(loftr_config['match_coarse'])
        self.fine_preprocess = FinePreprocess(loftr_config)
        self.loftr_fine = LocalFeatureTransformer(loftr_config["fine"])
        self.fine_matching = FineMatching2(geoformer_cfg['fine_temperature'], geoformer_cfg['fine_thr'])
        # loftr_config = geoformer_cfg
        self.geo_module = GeoModule(geoformer_cfg, loftr_config['coarse']['d_model'])

    def forward(self, data: Dict[str, torch.Tensor]):
        """
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        # 1. Local Feature CNN
        data.update({
            'bs': torch.tensor(data['image0'].size(0)),
            'hw0_i': torch.tensor(data['image0'].shape[2:]), 'hw1_i': torch.tensor(data['image1'].shape[2:])
        })

        if data['hw0_i'][0] == data['hw1_i'][0] and data['hw0_i'][1] == data['hw1_i'][1]:  # faster & better BN convergence
            feats_c, feats_f = self.backbone(torch.cat([data['image0'], data['image1']], dim=0))
            (feat_c0, feat_c1), (feat_f0, feat_f1) = feats_c.split(data['bs']), feats_f.split(data['bs'])
        else:  # handle different input shapes
            (feat_c0, feat_f0), (feat_c1, feat_f1) = self.backbone(data['image0']), self.backbone(data['image1'])
        cnn_feat0 = feat_c0
        cnn_feat1 = feat_c1
        data.update({
            'hw0_c': torch.tensor(feat_c0.shape[2:]), 'hw1_c': torch.tensor(feat_c1.shape[2:]),
            'hw0_f': torch.tensor(feat_f0.shape[2:]), 'hw1_f': torch.tensor(feat_f1.shape[2:])
        })

        # 2. coarse-level loftr module
        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
        feat_c0 = self.pos_encoding(feat_c0).permute(0, 2, 3, 1)
        # feat_c0 = feat_c0.permute(0, 2, 3, 1)
        n, h, w, c = feat_c0.shape
        feat_c0 = feat_c0.reshape(n, -1, c)

        feat_c1 = self.pos_encoding(feat_c1).permute(0, 2, 3, 1)
        # feat_c1 = feat_c1.permute(0, 2, 3, 1)
        n1, h1, w1, c1 = feat_c1.shape
        feat_c1 = feat_c1.reshape(n1, -1, c1)
        # feat_c1 = rearrange(self.pos_encoding(feat_c1), 'n c h w -> n (h w) c')

        mask_c0: Optional[torch.Tensor] = None
        mask_c1: Optional[torch.Tensor] = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)
        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

        # 3. match coarse-level
        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)
        data['dect_conf_matrix'] = data['conf_matrix']
        feat_c0, feat_c1 = self.geo_module(cnn_feat0, cnn_feat1, data)
        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)

        # feat_c0, feat_c1 = self.geo_module(feat_c0, feat_c1, feat_c0, feat_c1, data)
        # feat_f0 = F.upsample(feat_f0, scale_factor=2, mode='bilinear', align_corners=True)
        # feat_f1 = F.upsample(feat_f1, scale_factor=2, mode='bilinear', align_corners=True)
        # 4. fine-level refinement
        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat_f0, feat_f1, feat_c0, feat_c1, data)
        if feat_f0_unfold.size(0) != 0:  # at least one coarse level predicted
            feat_f0_unfold, feat_f1_unfold = self.loftr_fine(feat_f0_unfold, feat_f1_unfold)

        # 5. match fine-level
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)

        # kpts1 = data['mkpts0_f'].cpu().numpy()
        # kpts2 = data['mkpts1_f'].cpu().numpy()
        # gray1, gray2 = data['image0'], data['image1']
        # import matplotlib.pyplot as plt
        # import cv2
        # import numpy as np
        # plt.figure(dpi=200)
        # kp0 = kpts1
        # kp1 = kpts2
        # # if len(kp0) > 0:
        # kp0 = [cv2.KeyPoint(int(k[0]), int(k[1]), 30) for k in kp0]
        # kp1 = [cv2.KeyPoint(int(k[0]), int(k[1]), 30) for k in kp1]
        # matches = [cv2.DMatch(_trainIdx=i, _queryIdx=i, _distance=1, _imgIdx=-1) for i in
        #            range(len(kp0))]
        #
        # show = cv2.drawMatches((gray1.cpu()[0][0].numpy() * 255).astype(np.uint8), kp0,
        #                        (gray2.cpu()[0][0].numpy() * 255).astype(np.uint8), kp1, matches,
        #                        None)
        # plt.imshow(show)
        # plt.show()
        return data

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)
