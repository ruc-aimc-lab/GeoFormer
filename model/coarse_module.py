import kornia
import torch
import torch.nn.functional as F
from einops.einops import rearrange
from torch import nn
import cv2
# from model.module.resnet_fpn import ResNetFPN
from model.module.superpoint import DetectorMap
# from model.transformer.new_transformer import normalize_keypoints
from model.transformer.posencoder import PositionEncodingSine
from model.transformer.transformer import LocalFeatureTransformer, LocalFeatureTransformer_local, \
    LocalFeatureTransformer_my
# from model.transformer.new_transformer import KeypointEncoder
from utils.common_utils import generate_conf, get_map_keypoints, generate_window, sample_descriptors, simple_nms, \
    get_mask
from utils.homography import warp_points_batch, filter_points


class Coarse_Net(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.d_model = config['d_model']
        self.map_transformer = LocalFeatureTransformer(config, config['map_layer_names'], config['dect_dim'],
                                                       linear=True)

        self.des_transformer = LocalFeatureTransformer_local(config, config['des_layer_names'], config['d_model'],
                                                             linear=False)
        self.pos_encoding = PositionEncodingSine(
            config['d_model'])
        self.pos_encoding_dect = PositionEncodingSine(
            config['dect_dim'])
        self.mode = config['mode']
        self.bin = config['bin']
        self.tempr = 1
        self.bin_dect_score = nn.Parameter(
            torch.tensor(1., requires_grad=True))
        # self.DMap = DetectorMap(desc_dim=loftr_config['dect_dim'], bin=self.bin)
        self.bin_score = nn.Parameter(
            torch.tensor(1., requires_grad=True))
        self.thresh = 0.5
        self.window_size = config['window_size']
        self.matcher = None
        self.merge_desc = nn.Linear(2 * self.d_model, self.d_model, bias=True)

        self.query, self.refer = None, None
        # self.kernel =

    def sample_descripors_windows(self, kp_windows_list, desc_map_up, scale, norm=True):
        kps_dense_list = [kps.reshape(-1, 2) for kps in kp_windows_list]
        c = desc_map_up.shape[1]
        desc_windows_list = sample_descriptors(kps_dense_list, desc_map_up, s=scale, norm=norm)
        desc_windows_list = [desc_dense.reshape(kps.shape[0], kps.shape[1], c) for desc_dense, kps in
                             zip(desc_windows_list, kp_windows_list)]
        return desc_windows_list

    def get_kp_mk(self, keypoints1):
        bid_raw = (torch.arange(len(keypoints1)) * len(keypoints1)).unsqueeze(-1).unsqueeze(-1).repeat(
            [1, keypoints1.shape[1], 2]).to(keypoints1)
        idx1 = keypoints1 + bid_raw
        idx1 = idx1.view(-1, 2)

        idx1, rd, ct = torch.unique(idx1, dim=0, sorted=False, return_inverse=True, return_counts=True)
        bid = (idx1 // len(keypoints1))[:, 0]

        new_kp1 = torch.zeros_like(keypoints1)
        for i in range(len(new_kp1)):
            new_kp1[i, :(bid == i).sum()] = idx1[bid == i]
        new_kp1 = new_kp1.reshape(len(new_kp1), -1, 2) - bid_raw
        new_kp1[new_kp1 < 0] = 0
        new_mask = new_kp1.sum(-1) != 0
        keypoints1, mask1 = new_kp1, new_mask
        return keypoints1, mask1

    def cross_draw(self, data, kps0, kps1, kp0, kp1, kp_cross_list0, kp_cross_list1, wc0, wc1, l0, l1, M, mask, s=0):
        import matplotlib.pyplot as plt
        a = data['image0'][0][0].cpu()
        raw_img0 = cv2.imread(data['im0_path'])
        raw_img0 = cv2.cvtColor(raw_img0, cv2.COLOR_BGR2RGB)
        a0 = cv2.resize(raw_img0, (a.shape[1], a.shape[0]))
        a = data['image1'][0][0].cpu()
        raw_img1 = cv2.imread(data['im1_path'])
        raw_img1 = cv2.cvtColor(raw_img1, cv2.COLOR_BGR2RGB)
        a1 = cv2.resize(raw_img1, (a.shape[1], a.shape[0]))
        # M, mask = cv2.findHomography(kp0.cpu().numpy(), kp1.cpu().numpy(), cv2.USAC_MAGSAC, 3.5)
        kps0[s] = kp0[mask[:, 0] == 1].long()
        import numpy as np

        def draw(a0, a1, l0, kp_cross_list1, wc0, name, swap=False):
            raw_a0 = a0.copy()
            if swap:
                show0 = np.hstack([a1, a0])
                plt.figure(dpi=300)

                for id in range(0, l0, 800):
                    plt.axis('off')
                    i = (id % wc0 * data['desc_scale'] + a0.shape[1], id // wc0 * data['desc_scale'])
                    if i[0] == 0 or i[1] == 0 or i[0] == wc0 * data['desc_scale']-1:
                        continue
                    plt.imshow(show0)

                    kps = kp_cross_list1[s][id, :].cpu()
                    plt.scatter(kps[:, 0], kps[:, 1], c='lime', s=1)
                    flag = 0
                    for pt in kps:
                        if pt[0] == 0 and pt[1] == 0:
                            continue
                        flag = 1
                        # pt[0] = pt[0] + a0.shape[1]
                        plt.plot([i[0], pt[0]], [i[1], pt[1]], color='lime', linewidth=1, alpha=0.1)
                    if flag:
                        plt.scatter(i[0], i[1], c='cyan', s=12)
            else:
                show0 = np.hstack([a0, a1])
                plt.figure(dpi=300)

                for id in range(0, l0, 800):
                    plt.axis('off')
                    i = (id % wc0 * data['desc_scale'], id // wc0 * data['desc_scale'])
                    if i[0] == 0 or i[1] == 0 or i[0] == wc0 * data['desc_scale']:
                        continue


                    kps = kp_cross_list1[s][id, :].cpu()
                    if kps.sum() == 0:
                        continue
                    plt.imshow(show0)

                    plt.scatter(kps[:, 0] + a0.shape[1], kps[:, 1], c='lime', s=1)

                    flag = 0
                    for pt in kps:
                        if pt[0] == 0 and pt[1] == 0:
                            continue
                        flag = 1
                        pt[0] = pt[0] + a0.shape[1]
                        plt.plot([i[0], pt[0]], [i[1], pt[1]], color='lime', linewidth=1, alpha=0.1)
                    if flag:
                        plt.scatter(i[0], i[1], c='cyan', s=12)
            plt.savefig(f'figure/sp_cross_at{name}.png', bbox_inches='tight', dpi=300, pad_inches=0.0)
            plt.show()

        draw(a0, a1, l0, kp_cross_list1, wc0, 0)
        draw(a1, a0, l1, kp_cross_list0, wc1, 1, swap=True)

    def self_draw_my(self, data, kps0, kps1, kp0, kp1, M, mask, s=0,):
        import matplotlib.pyplot as plt
        a = data['image0'][0][0].cpu()
        raw_img0 = cv2.imread(data['im0_path'])
        raw_img0 = cv2.cvtColor(raw_img0, cv2.COLOR_BGR2RGB)
        a0 = cv2.resize(raw_img0, (a.shape[1], a.shape[0]))
        a = data['image1'][0][0].cpu()
        raw_img1 = cv2.imread(data['im1_path'])
        raw_img1 = cv2.cvtColor(raw_img1, cv2.COLOR_BGR2RGB)
        a1 = cv2.resize(raw_img1, (a.shape[1], a.shape[0]))
        # M, mask = cv2.findHomography(kp0.cpu().numpy(), kp1.cpu().numpy(), cv2.USAC_MAGSAC, 3.5)
        kps0[s] = kp0[mask[:, 0] == 1].long()

        def run_one(a0, kps0, name='0', swap=False):
            mask0 = torch.zeros([1, 1, a0.shape[0], a0.shape[1]])
            mask0[0, 0, kps0[s][:, 1], kps0[s][:, 0]] = 1
            sz = 9
            weight = torch.ones([1, 1, sz, sz])
            mask0 = F.conv2d(mask0, weight, stride=1, padding=sz // 2)
            mask0[mask0 > 0] = 1
            raw_a0 = a0.copy()
            v = a0[mask0.squeeze().numpy() == 0].mean(-1) * 0.4
            tmp = a0[mask0.squeeze().numpy() == 0]
            tmp[:, 0] = v
            tmp[:, 1] = v
            tmp[:, 2] = v
            a0[mask0.squeeze().numpy() == 0] = tmp
            import numpy as np
            # show0 = np.ones([raw_a0.shape[0], raw_a0.shape[1]*2, 4]) * 255
            # show0[:, :raw_a0.shape[1], :3] = raw_a0
            # show0[:, raw_a0.shape[1]:, :3] = a0
            plt.figure(dpi=300)
            plt.axis('off')
            show0 = a0
            i = (a0.shape[1] // 2 + a0.shape[1], a0.shape[0] // 2)
            # for k in kps0[s]:
            #     pt = (int(k[0]), int(k[1]))
            #     plt.plot([i[0], pt[0]], [i[1], pt[1]], color='lime', linewidth=1, alpha=0.1)
            plt.imshow(show0)
            # plt.scatter(kps0[s][:, 0].cpu(), kps0[s][:, 1].cpu(), c='lime', s=1)

            if swap:
                show0 = np.hstack([a0, raw_a0])
                # show0 = a0
                i = (a0.shape[1] // 2 + a0.shape[1], a0.shape[0] // 2)
                for k in kps0[s]:
                    pt = (int(k[0]), int(k[1]))
                    plt.plot([i[0], pt[0]], [i[1], pt[1]], color='lime', linewidth=1, alpha=0.1)
                plt.imshow(show0)
                plt.scatter(kps0[s][:, 0].cpu(), kps0[s][:, 1].cpu(), c='lime', s=1)
                plt.scatter(i[0], i[1], c='cyan', s=12)
            else:
                show0 = np.hstack([raw_a0, a0])
                i = (a0.shape[1] // 2, a0.shape[0] // 2)

                for k in kps0[s]:
                    pt = (int(k[0]) + a0.shape[1], int(k[1]))
                    plt.plot([i[0], pt[0]], [i[1], pt[1]], color='lime', linewidth=1, alpha=0.1)
                    # cv2.line(show0, i, pt, (0, 255, 0, 25), 1)

                plt.imshow(show0)
                plt.scatter(kps0[s][:, 0].cpu() + a0.shape[1], kps0[s][:, 1].cpu(), c='lime', s=1)
                plt.scatter(i[0], i[1], c='cyan', s=12)
            plt.savefig(f'figure/sp_self_at{name}.png', bbox_inches='tight', dpi=300, pad_inches=0.0)
            plt.show()
            # plt.imshow(a1)
            # plt.scatter(kps1[s][:, 0].cpu(), kps1[s][:, 1].cpu(), c='lime', s=1)
            # plt.show()

        run_one(a0, kps0, 0)
        run_one(a1, kps1, 1, swap=True)
        a = 1

    def forward(self, desc_map_down0, desc_map_down1, desc_map0, desc_map1, data):

        nc0, cc, hc0, wc0 = desc_map_down0.shape
        nc1, cc, hc1, wc1 = desc_map_down1.shape

        dect_query_map = rearrange(self.pos_encoding_dect(desc_map_down0), 'n c h w -> n (h w) c')
        dect_refer_map = rearrange(self.pos_encoding_dect(desc_map_down1), 'n c h w -> n (h w) c')
        # dect_query_map = rearrange(desc_map_down0, 'n c h w -> n (h w) c')
        # dect_refer_map = rearrange(desc_map_down1, 'n c h w -> n (h w) c')

        dect_query_map, dect_refer_map = self.map_transformer(dect_query_map, dect_refer_map)
        dect_desc_map0, dect_desc_map1 = dect_query_map, dect_refer_map
        dect_query_map = dect_query_map.permute(0, 2, 1)
        dect_refer_map = dect_refer_map.permute(0, 2, 1)
        # desc_map_down0 = desc_map_down0.view(nc0, cc, hc0, wc0)
        # desc_map_down1 = desc_map_down1.view(nc1, cc, hc1, wc1)

        # dect_conf_matrix, dect_query_map, dect_refer_map = self.DMap(dect_query_map, dect_refer_map)
        dect_conf_matrix = generate_conf(self.bin_dect_score, dect_query_map, dect_refer_map, self.d_model, self.tempr,
                                         self.bin)

        # dect_conf_matrix = torch.einsum('bdn,bdm->bnm', desc_map_down0.view(nc0, cc, -1), desc_map_down1.view(nc0, cc, -1))
        # dect_conf_matrix = desc_map_down0.mm(desc_map_down1.T)

        # dect_conf_matrix = torch.clamp(dect_conf_matrix, 1e-6, 1 - 1e-6)
        H0, W0 = data['image0'].shape[2:]
        H1, W1 = data['image1'].shape[2:]

        kps0, kps1, scores_list = self.matcher.coarse_matching(dect_conf_matrix, H0, W0,
                                                               H1, W1,
                                                               scale=data['desc_scale'],
                                                               coarse_thresh=0.1,
                                                               bin=self.bin)

        # kp_cross_list1, kp_cross_list0, kps0, kps1, = self.matcher.coarse_matching_idx(dect_conf_matrix, H0, W0,
        #                                                                                H1, W1,
        #                                                                                data['desc_scale'],
        #                                                                                coarse_thresh=0.1,
        #                                                                                bin=self.bin)
        # kp_cross_list0, mask_cross_list0 = generate_window(kp_cross_list0, (H0, W0),
        #                                                    window_size=self.window_size, scale=data['desc_scale'])
        # kp_cross_list1, mask_cross_list1 = generate_window(kp_cross_list1, (H1, W1),
        #                                                    window_size=self.window_size, scale=data['desc_scale'])
        # kps0, kps1 = [], []
        # for s, (kp0, kp1) in enumerate(zip(raw_kps0, raw_kps1)):
        #     M = None
        #     if len(kp0) > 8:
        #         if self.mode == 'H':
        #             M, mask = cv2.findHomography(kp0.cpu().numpy(), kp1.cpu().numpy(), cv2.RANSAC, 5.0)
        #         else:
        #             M, mask = cv2.findFundamentalMat(kp0.cpu().numpy(), kp1.cpu().numpy(), method=cv2.FM_RANSAC,
        #                                              ransacReprojThreshold=3)
        #             # cv2.perspectiveTransform()
        #
        #     if M is not None:
        #         # _, inmask = kornia.geometry.ransac.RANSAC(inl_th=10)(kp0.float(), kp1.float())
        #         kps0.append(kp0[mask[:, 0] == 1].long())
        #         kps1.append(kp1[mask[:, 0] == 1].long())
        #     else:
        #         kps0.append(kp0.long())
        #         kps1.append(kp1.long())
        _, cc, hm0, wm0 = desc_map0.shape
        _, cc, hm1, wm1 = desc_map1.shape

        kp_cross_list1 = []
        kp_cross_list0 = []
        mask_cross_list1 = []
        mask_cross_list0 = []
        for s, (kp0, kp1) in enumerate(zip(kps0, kps1)):
            M = None
            if len(kp0) > 8:
                if self.mode == 'H':
                    M, mask = cv2.findHomography(kp0.cpu().numpy(), kp1.cpu().numpy(), cv2.RANSAC, 8.0)
                else:
                    F_hat, mask = cv2.findFundamentalMat(kp0.cpu().numpy(), kp1.cpu().numpy(), method=cv2.FM_RANSAC,
                                                         ransacReprojThreshold=5)
                    # cv2.perspectiveTransform()
                # if mask.sum() / len(kp0) < 0.3:
                #     M = None
        #     # draw common area--------------------------------
        #     # import matplotlib.pyplot as plt
        #     # import numpy as np
        #     #
        #     # a = data['image0'][0][0].cpu()
        #     # raw_img0 = cv2.imread(data['im0_path'])
        #     # raw_img0 = cv2.cvtColor(raw_img0, cv2.COLOR_BGR2RGB)
        #     # a0 = cv2.resize(raw_img0, (a.shape[1], a.shape[0]))
        #     # a = data['image1'][0][0].cpu()
        #     # raw_img1 = cv2.imread(data['im1_path'])
        #     # raw_img1 = cv2.cvtColor(raw_img1, cv2.COLOR_BGR2RGB)
        #     # a1 = cv2.resize(raw_img1, (a.shape[1], a.shape[0]))
        #     # M, mask = cv2.findHomography(kp0.cpu().numpy(), kp1.cpu().numpy(), cv2.USAC_MAGSAC, 3.5)
        #     # kps0[s] = kp0[mask[:, 0] == 1].long()
        #     # m0 = get_mask(a0)
        #     # m1 = get_mask(a1)
        #     # # m1 = np.ones((a1.shape[0], a1.shape[1], 3))
        #     # m0 = cv2.cvtColor(m0, cv2.COLOR_GRAY2BGR) / 255
        #     # m1 = cv2.cvtColor(m1, cv2.COLOR_GRAY2BGR) / 255
        #     # # m0 = np.ones((a0.shape[0], a0.shape[1], 3))
        #     # m0_warp = cv2.warpPerspective(m0, M, dsize=(a0.shape[1], a0.shape[0]))
        #     # raw_m0_warp = m0_warp.copy()
        #     # m0_warp[m0_warp==0] = 0.3
        #     # a1 = a1 * m0_warp
        #     # # m1 = np.ones((a1.shape[0], a1.shape[1], 3))
        #     # m1_warp = cv2.warpPerspective(raw_m0_warp * m1, np.linalg.inv(M), dsize=(a1.shape[1], a1.shape[0]))
        #     # # m1_warp = m1_warp
        #     # m1_warp[m1_warp == 0] = 0.3
        #     # a0 = a0 * m1_warp
        #     # plt.figure(dpi=300)
        #     # plt.axis('off')
        #     # plt.imshow(a0/255)
        #     # plt.savefig(f'figure/common0.png', bbox_inches='tight', dpi=300, pad_inches=0.0)
        #     # plt.show()
        #     # plt.figure(dpi=300)
        #     # plt.axis('off')
        #     # plt.imshow(a1/255)
        #     # plt.savefig(f'figure/common1.png', bbox_inches='tight', dpi=300, pad_inches=0.0)
        #     # plt.show()
        #     # --------------------------------
        #     data['first_match_num'] = len(kp0)
        #     data['first_ransac_num'] = 0
            if M is not None:
                kps0[s] = kp0[mask[:, 0] == 1].long()
                kps1[s] = kp1[mask[:, 0] == 1].long()
                a = 1
                data['first_ransac_num'] = (mask[:, 0] == 1).sum()
                # -----------------
                # -----------------

                # keypoints = get_map_keypoints(H0, W0, data['desc_scale']).to(desc_map0.device)
                # kp_windows_list1, masks_list1 = generate_window([keypoints], (H0, W0),
                #                                                 window_size=self.window_size, scale=data['desc_scale'])
                # keypoints1, mask1 = kp_windows_list1[0], masks_list1[0]
                # keypoints1 = \
                #     warp_points_batch(keypoints1,
                #                       homographies=torch.from_numpy(M).unsqueeze(0).to(desc_map0))
                #
                # keypoints1, _mask1 = filter_points(keypoints1, [W1, H1], keypoints1.device)
                # mask1 = _mask1 * mask1
                #
                # keypoints1, mask1 = self.get_kp_mk(keypoints1)
                # kp_windows_list1, masks_list1 = [keypoints1], [mask1]
                # # desc_windows1 = sample_descriptors(kp_windows_list1, desc_map1, data['desc_scale'])
                #
                # keypoints = get_map_keypoints(H1, W1, data['desc_scale']).to(desc_map0.device)
                # # kp_windows_list0, masks_list0 = generate_window([keypoints0], (hm0 * dect_scale, wm0 * dect_scale),
                # #                                                 window_size=self.window_size)
                # kp_windows_list0, masks_list0 = generate_window([keypoints], (H1, W1),
                #                                                 window_size=self.window_size, scale=data['desc_scale'])
                # keypoints0, mask0 = kp_windows_list0[0], masks_list0[0]
                # keypoints0 = \
                #     warp_points_batch(keypoints0,
                #                       homographies=torch.inverse(torch.from_numpy(M)).unsqueeze(0).to(desc_map0))
                # keypoints0, _mask0 = filter_points(keypoints0, [W0, H0], keypoints0.device)
                # mask0 = _mask0 * mask0
                # keypoints0, mask0 = self.get_kp_mk(keypoints0)
                #
                #
                # kp_windows_list0, masks_list0 = [keypoints0], [mask0]
                # desc_windows0 = sample_descriptors(kp_windows_list0, desc_map0, data['desc_scale'])

                keypoints = get_map_keypoints(H0, W0, data['desc_scale']).to(desc_map0.device)
                # kp_windows_list0, masks_list0 = generate_window([keypoints0], (hm0 * dect_scale, wm0 * dect_scale),
                #                                                 window_size=self.window_size)
                keypoints1 = \
                    warp_points_batch(keypoints.unsqueeze(0),
                                      homographies=torch.from_numpy(M).unsqueeze(0).to(desc_map0))[0]
                kp_windows_list1, masks_list1 = generate_window([keypoints1], (H1, W1),
                                                                window_size=self.window_size, scale=data['desc_scale'])
                # desc_windows1 = sample_descriptors(kp_windows_list1, desc_map1, desc_scale)
                # desc_cross_list1.extend(desc_windows1)

                keypoints = get_map_keypoints(H1, W1, data['desc_scale']).to(desc_map0.device)

                keypoints0 = \
                    warp_points_batch(keypoints.unsqueeze(0),
                                      homographies=torch.inverse(torch.from_numpy(M).unsqueeze(0)).to(desc_map0))[0]

                kp_windows_list0, masks_list0 = generate_window([keypoints0], (H0, W0),
                                                                window_size=self.window_size, scale=data['desc_scale'])
                # desc_windows0 = sample_descriptors(kp_windows_list0, desc_map0, desc_scale)

                kp_cross_list0.extend(kp_windows_list0)
                kp_cross_list1.extend(kp_windows_list1)
                mask_cross_list0.extend(masks_list0)
                mask_cross_list1.extend(masks_list1)
            else:
                kp_cross_list1.append(None)
                kp_cross_list0.append(None)
                mask_cross_list1.append(None)
                mask_cross_list0.append(None)

        # import matplotlib.pyplot as plt
        # import numpy as np
        # im1 = (data['image0'][0][0].cpu().numpy()*255).astype(np.uint8)
        # im2 = (data['image1'][0][0].cpu().numpy()*255).astype(np.uint8)
        # self.self_draw(data, kps0, kps1, kp0, kp1, M, mask)
        # self.cross_draw(data, kps0, kps1, kp0, kp1, kp_cross_list0, kp_cross_list1, wc0, wc1, hm0 * wm0, hm1 * wm1,
        #                 M, mask, s=0, )
        a = 1

        # w = im1.shape[1]
        # plt.figure(dpi=200)
        # kp0_kp = [cv2.KeyPoint(int(k[0]), int(k[1]), 30) for k in kps0[0]]
        # kp1_kp = [cv2.KeyPoint(int(k[0]), int(k[1]), 30) for k in kps1[0]]
        # matches = [cv2.DMatch(_trainIdx=i, _queryIdx=i, _distance=1, _imgIdx=-1) for i in range(len(kps0[0]))]
        # show = cv2.drawMatches(im1, kp0_kp,
        #                        im2, kp1_kp, matches, None)
        # plt.imshow(show)
        # plt.show()
        sc = data['detect_scale'] // data['desc_scale']
        map0 = torch.zeros([nc0, 1, hc0 * sc, wc0 * sc]).to(
            desc_map0.device)
        map1 = torch.zeros([nc1, 1, hc1 * sc, wc1 * sc]).to(
            desc_map0.device)
        for i in range(nc0):
            # q_v = dect_query_map_raw[i, :, kps0[i][:, 1] // data['detect_scale'], kps0[i][:, 0] // data['detect_scale']]
            # r_v = dect_refer_map_raw[i, :, kps1[i][:, 1] // data['detect_scale'], kps1[i][:, 0] // data['detect_scale']]
            map0[i, :, kps0[i][:, 1] // data['desc_scale'], kps0[i][:, 0] // data['desc_scale']] = 1
            map1[i, :, kps1[i][:, 1] // data['desc_scale'], kps1[i][:, 0] // data['desc_scale']] = 1

        map0 = map0 > self.thresh
        map1 = map1 > self.thresh
        # dect_query_map_raw = dect_query_map.view(nc0, 1, hc0, wc0)
        # dect_refer_map_raw = dect_refer_map.view(nc1, 1, hc1, wc1)
        map0_show = map0
        map1_show = map1
        map0 = map0.view(nc0, -1)
        map1 = map1.view(nc1, -1)

        nc0, cc, hc0, wc0 = desc_map0.shape
        nc1, cc, hc1, wc1 = desc_map1.shape
        # desc_map0 = rearrange(desc_map0, 'n c h w -> n (h w) c') * dect_query_map
        # desc_map1 = rearrange(desc_map1, 'n c h w -> n (h w) c') * dect_refer_map
        # desc_map0 = rearrange(self.pos_encoding(desc_map0*map0), 'n c h w -> n (h w) c')
        # desc_map1 = rearrange(self.pos_encoding(desc_map1*map1), 'n c h w -> n (h w) c')
        desc_map0 = rearrange(self.pos_encoding(desc_map0), 'n c h w -> n (h w) c')
        desc_map1 = rearrange(self.pos_encoding(desc_map1), 'n c h w -> n (h w) c')
        # desc_map0 = dect_query_map.permute(0, 2, 1)
        # desc_map1 = dect_refer_map.permute(0, 2, 1)
        # desc_map0, desc_map1 = self.des_transformer(desc_map0, desc_map1)
        # desc_map0, desc_map1 = self.des_transformer(desc_map0, desc_map1, mask_self0=map0, mask_self1=map1)
        # desc_map0, desc_map1 = self.des_transformer(dect_desc_map0.clone(), dect_desc_map1.clone(), kp_cross_list0, kp_cross_list1, hc0, wc0, hc1,
        #                                             wc1, data['desc_scale'],
        #                                             mask_self0=map0, mask_self1=map1, mask_cross0=mask_cross_list0,
        #                                             mask_cross1=mask_cross_list1)
        desc_map0_o, desc_map1_o = self.des_transformer(desc_map0, desc_map1, kp_cross_list0,
                                                    kp_cross_list1, hc0, wc0, hc1,
                                                    wc1, data['desc_scale'],
                                                    mask_self0=map0, mask_self1=map1, mask_cross0=mask_cross_list0,
                                                    mask_cross1=mask_cross_list1)
        # desc_map0 = torch.cat((desc_map0, desc_map0_o), -1)
        # desc_map1 = torch.cat((desc_map1, desc_map1_o), -1)
        # desc_map0 = self.merge_desc(desc_map0)
        # desc_map1 = self.merge_desc(desc_map1)
        desc_map0, desc_map1 = desc_map0_o, desc_map1_o
        # import matplotlib.pyplot as plt
        # plt.figure(dpi=300)
        # id = 5000
        # id2 = 4000
        # a = data['image0'][0][0].cpu()
        # raw_img0 = cv2.imread(data['im0_path'])
        # a = cv2.resize(raw_img0, (a.shape[1], a.shape[0]))
        # plt.axis('off')
        # plt.imshow(a, 'gray')
        # plt.scatter(id % wc0 * data['desc_scale'], id // wc0 * data['desc_scale'], c='r', s=8)
        # plt.scatter(id2 % wc0 * data['desc_scale'], id2 // wc0 * data['desc_scale'], c='r', s=8)
        # plt.savefig('figure/cross_my0_0.png', bbox_inches='tight', dpi=300, pad_inches=0.0)
        # plt.close()
        # plt.figure(dpi=300)
        # plt.axis('off')
        # a = data['image1'][0][0].cpu()
        # raw_img1 = cv2.imread(data['im1_path'])
        # a = cv2.resize(raw_img1, (a.shape[1], a.shape[0]))
        # plt.imshow(a, 'gray')
        # plt.scatter(kp_cross_list1[0][id, :, 0].cpu(), kp_cross_list1[0][id, :, 1].cpu(), c='r', s=1, alpha=1)
        # plt.scatter(kp_cross_list1[0][id2, :, 0].cpu(), kp_cross_list1[0][id2, :, 1].cpu(), c='r', s=1, alpha=1)
        # plt.savefig('figure/cross_my0_1.png', bbox_inches='tight', dpi=300, pad_inches=0.0)
        # plt.close()

        # desc_map0 = desc_map0 + dect_query_map
        # desc_map1 = desc_map1 + dect_refer_map
        # desc_map0, desc_map1 = self.merge_desc(torch.cat((desc_map0,dect_desc_map0), -1)), self.merge_desc(torch.cat((desc_map1,dect_desc_map1), -1))

        # kp_windows_list0, masks_list0 = generate_window(raw_kps0, (H0, W0), window_size=self.window_size,
        #                                                 scale=data['desc_scale'])
        # kp_windows_list1, masks_list1 = generate_window(raw_kps1, (H1, W1), window_size=self.window_size,
        #                                                 scale=data['desc_scale'])
        # import matplotlib.pyplot as plt
        #
        #
        # plt.figure(dpi=300)
        #
        # id = 2888
        # id2 = 1118
        # plt.axis('off')
        # a = data['image0'][0][0].cpu()
        # a[0][0] = 1.1
        # plt.imshow(a, 'gray')
        # plt.scatter(id % wc0 * data['desc_scale'], id // wc0 * data['desc_scale'], c='r', s=8)
        # plt.scatter(id2 % wc0 * data['desc_scale'], id2 // wc0 * data['desc_scale'], c='r', s=8)
        # plt.savefig('sp_cross_at0.png', bbox_inches='tight', dpi=300, pad_inches=0.0)
        # plt.close()
        # plt.figure(dpi=300)
        # plt.axis('off')
        # a = data['image1'][0][0].cpu()
        # a[0][0] = 1.1
        # plt.imshow(a, 'gray')
        # plt.scatter(kp_cross_list1[0][id, :, 0].cpu(), kp_cross_list1[0][id, :, 1].cpu(), c='r', s=6)
        # plt.scatter(kp_cross_list1[0][id2, :, 0].cpu(), kp_cross_list1[0][id2, :, 1].cpu(), c='r', s=6)
        # plt.savefig('sp_cross_at1.png', bbox_inches='tight', dpi=300, pad_inches=0.0)
        #
        # plt.figure(dpi=300)
        # plt.axis('off')
        # show0 = F.interpolate(data['image0'], scale_factor=1 / data['detect_scale'])
        # show = torch.zeros([show0.shape[2], show0.shape[3], 3])
        # show[:, :, 2] = show0[0][0].cpu()
        # show[:, :, 1] = show0[0][0].cpu()
        # show[:, :, 0] = map0_show[0][0].cpu() + show0[0][0].cpu()
        # plt.imshow(show, 'gray')
        # plt.savefig('sp_self_at0.png', bbox_inches='tight', dpi=300, pad_inches=0.0)
        # plt.close()
        # plt.figure(dpi=300)
        # plt.axis('off')
        # show1 = F.interpolate(data['image1'], scale_factor=1 / data['detect_scale'])
        # show = torch.zeros([show1.shape[2], show1.shape[3], 3])
        # show[:, :, 2] = show1[0][0].cpu()
        # show[:, :, 1] = show1[0][0].cpu()
        # show[:, :, 0] = map1_show[0][0].cpu() + show1[0][0].cpu()
        # plt.imshow(show, 'gray')
        # plt.savefig('sp_self_at1.png', bbox_inches='tight', dpi=300, pad_inches=0.0)

        desc_map0 = desc_map0.permute(0, 2, 1)
        desc_map1 = desc_map1.permute(0, 2, 1)
        desc_conf_global = generate_conf(self.bin_score, desc_map0, desc_map1, self.d_model, self.tempr, self.bin)
        desc_map0, desc_map1 = desc_map0.view(nc0, cc, hc0, wc0), desc_map1.view(nc1, cc, hc1, wc1)
        # kp_windows_list0 = [kp.unsqueeze(1) for kp in raw_kps0]
        # masks_list0 = [torch.ones(len(kp), 1).to(kp.device) for kp in raw_kps0]
        # desc_windows_list0 = self.sample_descripors_windows(kp_windows_list0, desc_map0, data['desc_scale'], norm=False)
        # desc_windows_list1 = self.sample_descripors_windows(kp_windows_list1, desc_map1, data['desc_scale'], norm=False)
        # matrix_list = []
        # for i in range(len(kps0)):
        #     desc_windows0, desc_windows1 = desc_windows_list0[i], desc_windows_list1[i]
        #     mask0, mask1 = masks_list0[i], masks_list1[i]
        #
        #     conf_matrix = generate_conf(self.bin_score, desc_windows0.permute(0, 2, 1), desc_windows1.permute(0, 2, 1),
        #                                 self.d_model, self.tempr, self.bin)
        #     mask = mask0[:, :, None] * mask1[:, None, :]
        #     temp = conf_matrix[:, :-1, :-1]
        #     temp[mask == 0] = 0 * temp[mask == 0]
        #     conf_matrix[:, :-1, :-1] = temp
        #     # conf_matrix[mask == 0] = 0
        #     conf_matrix = torch.clamp(conf_matrix, 1e-6, 1 - 1e-6)
        #     matrix_list.append(conf_matrix)
        #
        # data['coarse_windows_kp0'] = kp_windows_list0
        # data['coarse_windows_kp1'] = kp_windows_list1
        # data['desc_conf_global'] = desc_conf_global
        # return desc_conf, desc_map0, desc_map1, dect_conf_matrix
        # desc_conf, desc_map0, desc_map = dect_conf_matrix, dect_query_map, dect_refer_map
        return desc_conf_global, desc_map0, desc_map1, dect_conf_matrix, kps0, kps1
