from argparse import Namespace
import torch
import numpy as np
import cv2

from model.loftr_src.loftr.utils.cvpr_ds_config import default_cfg
from model.full_model import GeoFormer as GeoFormer_
from .base import Matching
from eval_tool.immatch.utils.data_io import load_gray_scale_tensor_cv
from model.geo_config import default_cfg as geoformer_cfg

class GeoFormer(Matching):
    def __init__(self, args, gpuid=0):
        super().__init__(gpuid)
        if type(args) == dict:
            args = Namespace(**args)

        self.imsize = args.imsize
        self.match_threshold = args.match_threshold
        self.no_match_upscale = args.no_match_upscale

        # Load model
        conf = dict(default_cfg)
        conf['match_coarse']['thr'] = self.match_threshold
        geoformer_cfg['coarse_thr'] = self.match_threshold
        self.model = GeoFormer_(conf)
        ckpt_dict = torch.load(args.ckpt, map_location=torch.device('cpu'))
        if 'state_dict' in ckpt_dict:
            ckpt_dict = ckpt_dict['state_dict']
        self.model.load_state_dict(ckpt_dict, strict=False)
        self.model = self.model.eval().to(self.device)

        # Name the method
        self.ckpt_name = args.ckpt.split('/')[-1].split('.')[0]
        self.name = f'GeoFormer_{self.ckpt_name}'
        if self.no_match_upscale:
            self.name += '_noms'
        print(f'Initialize {self.name}')

    def change_deivce(self, device):
        self.device = device
        self.model.to(device)
    def load_im(self, im_path, enhanced=False):
        return load_gray_scale_tensor_cv(
            im_path, self.device, imsize=self.imsize, dfactor=8, enhanced=enhanced, value_to_scale=min
        )

    def match_inputs_(self, gray1, gray2):

        batch = {'image0': gray1, 'image1': gray2}
        with torch.no_grad():
            batch = self.model(batch)
        kpts1 = batch['mkpts0_f'].cpu().numpy()
        kpts2 = batch['mkpts1_f'].cpu().numpy()

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
        scores = batch['mconf'].cpu().numpy()
        matches = np.concatenate([kpts1, kpts2], axis=1)
        return matches, kpts1, kpts2, scores

    def match_pairs(self, im1_path, im2_path, cpu=False):
        torch.cuda.empty_cache()
        tmp_device = self.device
        if cpu:
            self.change_deivce('cpu')
        gray1, sc1 = self.load_im(im1_path)
        gray2, sc2 = self.load_im(im2_path)

        upscale = np.array([sc1 + sc2])
        matches, kpts1, kpts2, scores = self.match_inputs_(gray1, gray2)

        if self.no_match_upscale:
            return matches, kpts1, kpts2, scores, upscale.squeeze(0)

        # Upscale matches &  kpts
        matches = upscale * matches
        kpts1 = sc1 * kpts1
        kpts2 = sc2 * kpts2

        if cpu:
            self.change_deivce(tmp_device)

        return matches, kpts1, kpts2, scores


    # def match_pairs(self, im1_path, im2_path, cpu=False):
    #     tmp_device = self.device
    #     if cpu:
    #         self.change_deivce('cpu')
    #     gray1, sc1 = self.load_im(im1_path)
    #     gray2, sc2 = self.load_im(im2_path)
    #
    #
    #     torch.cuda.empty_cache()
    #     upscale = np.array([sc1 + sc2])
    #     data = {'image0': gray1, 'image1': gray2}
    #
    #     with torch.no_grad():
    #         self.model(data)
    #
    #         fine_kps_list1, fine_kps_list2, fine_scores_list = data['fine_kps_list0'], data['fine_kps_list1'], data['fine_scores_list']
    #
    #     kpts1, kpts2, scores = fine_kps_list1[0].cpu().numpy(), fine_kps_list2[0].cpu().numpy(), fine_scores_list[0].cpu().numpy()
    #     matches = np.concatenate([kpts1, kpts2], axis=1)
    #
    #     if self.no_match_upscale:
    #         if 'first_match_num' in data:
    #             return matches, kpts1, kpts2, scores, upscale.squeeze(0), data['first_match_num'], data['first_ransac_num']
    #         return matches, kpts1, kpts2, scores, upscale.squeeze(0)
    #
    #     # Upscale matches &  kpts
    #     matches = upscale * matches
    #     kpts1 = sc1 * kpts1
    #     kpts2 = sc2 * kpts2
    #     if cpu:
    #         self.change_deivce(tmp_device)
    #     return matches, kpts1, kpts2, scores

