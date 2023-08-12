import os
import numpy as np
import glob
import time
import pydegensac
import cv2
import torch
from tqdm import tqdm


def compute_auc(s_error, p_error, a_error):
    assert (len(s_error) == 71)  # Easy pairs
    assert (len(p_error) == 48)  # Hard pairs. Note file control_points_P37_1_2.txt is ignored
    assert (len(a_error) == 14)  # Moderate pairs

    s_error = np.array(s_error)
    p_error = np.array(p_error)
    a_error = np.array(a_error)

    limit = 25
    gs_error = np.zeros(limit + 1)
    gp_error = np.zeros(limit + 1)
    ga_error = np.zeros(limit + 1)

    accum_s = 0
    accum_p = 0
    accum_a = 0

    for i in range(1, limit + 1):
        gs_error[i] = np.sum(s_error < i) * 100 / len(s_error)
        gp_error[i] = np.sum(p_error < i) * 100 / len(p_error)
        ga_error[i] = np.sum(a_error < i) * 100 / len(a_error)

        accum_s = accum_s + gs_error[i]
        accum_p = accum_p + gp_error[i]
        accum_a = accum_a + ga_error[i]

    auc_s = accum_s / (limit * 100)
    auc_p = accum_p / (limit * 100)
    auc_a = accum_a / (limit * 100)
    mAUC = (auc_s + auc_p + auc_a) / 3.0
    return {'s': auc_s, 'p': auc_p, 'a': auc_a, 'mAUC': mAUC}


def cal_reproj_dists(p1s, p2s, homography):
    '''Compute the reprojection errors using the GT homography'''

    p1s_h = np.concatenate([p1s, np.ones([p1s.shape[0], 1])], axis=1)  # Homogenous
    p2s_proj_h = np.transpose(np.dot(homography, np.transpose(p1s_h)))
    p2s_proj = p2s_proj_h[:, :2] / p2s_proj_h[:, 2:]
    dist = np.sqrt(np.sum((p2s - p2s_proj) ** 2, axis=1))
    return dist


def eval_summary_homography(dists_ss, dists_sp, dists_sa):
    dists_ss, dists_sp, dists_sa = map(lambda dist: np.array(dist), [dists_ss, dists_sp, dists_sa])
    # Compute aucs
    auc = compute_auc(dists_ss, dists_sp, dists_sa)

    # Generate summary
    summary = f'Hest AUC: m={auc["mAUC"]}\ns={auc["s"]}\np={auc["p"]}\na={auc["a"]}\n'
    print(summary)
    return auc["mAUC"]

def scale_homography(sw, sh):
    return np.array([[sw,  0, 0],
                     [ 0, sh, 0],
                     [ 0,  0, 1]])

def eval_fire(
        matcher,
        match_pairs,
        im_dir,
        gt_dir,
        method='',
        task='homography',
        scale_H=False,
        h_solver='degensac',
        ransac_thres=2,
        lprint_=print,
        debug=False,
):
    np.set_printoptions(precision=4)

    assert task == 'homography'
    lprint_(f'\n>>>>Eval hpatches: task={task} method={method} scale_H={scale_H} rthres={ransac_thres}')
    # Homography

    inlier_ratio = []
    h_failed = 0
    dists_ss = []
    dists_sp = []
    dists_sa = []
    image_num = 0
    failed = 0
    inaccurate = 0
    first_ransac_num = 0
    first_match_num = 0
    first_match = 0

    match_failed = 0
    n_matches = []
    match_time = []
    start_time = time.time()
    for pair_idx, pair_file in tqdm(enumerate(match_pairs), total=len(match_pairs), smoothing=.5):
        if debug and pair_idx > 10:
            break
        file_name = pair_file.replace('.txt', '')
        gt_file = os.path.join(gt_dir, pair_file)

        refer = file_name.split('_')[2] + '_' + file_name.split('_')[3]
        query = file_name.split('_')[2] + '_' + file_name.split('_')[4]
        im1_path = os.path.join(im_dir, query + '.jpg')
        im2_path = os.path.join(im_dir, refer + '.jpg')
        # Eval on composed pairs within seq
        image_num += 1
        category = file_name.split('_')[2][0]
        # Predict matches
        try:
            t0 = time.time()
            match_res = matcher(im1_path, im2_path)
            if len(match_res) > 5:
                first_match_num += match_res[-2]
                first_ransac_num += match_res[-1]
                first_match += 1
            match_time.append(time.time() - t0)
            matches, p1s, p2s = match_res[0:3]
        except Exception as e:
            print(str(e))
            p1s = p2s = matches = []
            match_failed += 1
        n_matches.append(len(matches))

        if 'homography' in task:
            try:

                if 'cv' in h_solver:
                    H_pred, inliers = cv2.findHomography(matches[:, :2], matches[:, 2:4], cv2.RANSAC, ransac_thres)
                else:
                    H_pred, inliers = pydegensac.findHomography(matches[:, :2], matches[:, 2:4], ransac_thres)
                if scale_H:
                    scale = match_res[4]

                    # Scale gt homoragphies
                    H_scale_im1 = scale_homography(1/scale[0], 1/scale[1])
                    H_scale_im2 = scale_homography(1/scale[2], 1/scale[3])
                    H_pred = np.linalg.inv(H_scale_im2) @ H_pred @ H_scale_im1

            except:
                H_pred = None

            if H_pred is None:
                avg_dist = big_num = 1e6
                irat = 0
                h_failed += 1
                failed += 1
                inliers = []
            else:
                points_gd = np.loadtxt(gt_file)
                raw = np.zeros([len(points_gd), 2])
                dst = np.zeros([len(points_gd), 2])
                raw[:, 0] = points_gd[:, 2]
                raw[:, 1] = points_gd[:, 3]
                dst[:, 0] = points_gd[:, 0]
                dst[:, 1] = points_gd[:, 1]
                # if scale_H:
                #     # scale = (wo / wt, ho / ht) for im1 & im2
                #     scale = match_res[4]
                #
                #     # Scale gt homoragphies
                #     H_scale_im1 = scale_homography(scale[1], scale[0])
                #     H_scale_im2 = scale_homography(scale[3], scale[2])
                #     H_pred = np.linalg.inv(H_scale_im2) @ H_pred @ H_scale_im1
                dst_pred = cv2.perspectiveTransform(raw.reshape(-1, 1, 2), H_pred).squeeze()
                dis = (dst - dst_pred) ** 2
                dis = np.sqrt(dis[:, 0] + dis[:, 1])
                avg_dist = dis.mean()
                irat = np.mean(inliers)
                mae = dis.max()
                mee = np.median(dis)
                if mae > 50 or mee > 20:
                    inaccurate += 1

                # if avg_dist > 0:
                #     kp0 = p1s
                #     kp1 = p2s
                #     if scale_H:
                #         # scale = (wo / wt, ho / ht) for im1 & im2
                #         scale = match_res[4]
                #         sc1 = scale[:2]
                #         sc2 = scale[2:]
                #         kp0 = sc1 * p1s
                #         kp1 = sc2 * p2s
                #     kp0, kp1 = torch.from_numpy(kp0), torch.from_numpy(kp1)
                #
                #     print(im1_path, im2_path, avg_dist)
                #     # if im2_path == './data/datasets/hpatches-sequences-release/i_bridger/6.ppm':
                #     # import matplotlib.pyplot as plt
                #     # im1 = cv2.imread(im1_path, 0)
                #     # im2 = cv2.imread(im2_path, 0)
                #     # w = im1.shape[1]
                #     # plt.figure(dpi=200)
                #     # kp0_kp = [cv2.KeyPoint(int(k[0]), int(k[1]), 30) for k in kp0]
                #     # kp1_kp = [cv2.KeyPoint(int(k[0]), int(k[1]), 30) for k in kp1]
                #     # matches = [cv2.DMatch(_trainIdx=i, _queryIdx=i, _distance=1, _imgIdx=-1) for i in range(len(kp0))]
                #     # show = cv2.drawMatches(im1, kp0_kp,
                #     #                        im2, kp1_kp, matches, None)
                #     # plt.imshow(show)
                #     # plt.title(f'error:{avg_dist}')
                #     #
                #     # plt.scatter(dst.squeeze()[:, 0]+w, dst.squeeze()[:, 1], s=2, c='g')
                #     # plt.scatter(dst_pred.squeeze()[:, 0]+w, dst.squeeze()[:, 1], s=2, c='r')
                #     # plt.show()
            inlier_ratio.append(irat)

            if category == 'S':
                dists_ss.append(avg_dist)
            if category == 'P':
                dists_sp.append(avg_dist)
            if category == 'A':
                dists_sa.append(avg_dist)

    lprint_(
        f'>>Finished, pairs={len(match_time)} match_failed={match_failed} matches={np.mean(n_matches):.1f} match_time={np.mean(match_time):.2f}s')
    print('----------------------------------------------------------------------------------------')
    # print(f'first_matches_num: {first_match_num / first_match}, first_ransac_num: {first_ransac_num / first_match}')
    print('----------------------------------------------------------------------------------------')
    lprint_('==== Homography Estimation ====')
    lprint_(
        f'Hest solver={h_solver} est_failed={h_failed} ransac_thres={ransac_thres} inlier_rate={np.mean(inlier_ratio):.2f}')
    mauc = eval_summary_homography(dists_ss, dists_sp, dists_sa)

    print('-' * 40)
    print(f"Failed:{'%.2f' % (100 * failed / image_num)}%, Inaccurate:{'%.2f' % (100 * inaccurate / image_num)}%, "
          f"Acceptable:{'%.2f' % (100 * (image_num - inaccurate - failed) / image_num)}%")

    print('-' * 40)

    return mauc