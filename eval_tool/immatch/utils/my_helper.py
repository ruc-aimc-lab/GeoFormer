import os
import numpy as np
import glob
import time
import pydegensac
import cv2
import torch
from tqdm import tqdm
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

from eval_tool.immatch.utils.hpatches_helper import cal_error_auc


def compute_auc(a_error):

    a_error = np.array(a_error)

    limit = 10
    ga_error = np.zeros(limit + 1)

    accum_a = 0

    for i in range(1, limit + 1):
        ga_error[i] = np.sum(a_error < i) * 100 / len(a_error)

        accum_a = accum_a + ga_error[i]

    auc_a = accum_a / (limit * 100)
    return {'auc': auc_a}


def cal_reproj_dists(p1s, p2s, homography):
    '''Compute the reprojection errors using the GT homography'''

    p1s_h = np.concatenate([p1s, np.ones([p1s.shape[0], 1])], axis=1)  # Homogenous
    p2s_proj_h = np.transpose(np.dot(homography, np.transpose(p1s_h)))
    p2s_proj = p2s_proj_h[:, :2] / p2s_proj_h[:, 2:]
    dist = np.sqrt(np.sum((p2s - p2s_proj) ** 2, axis=1))
    return dist


def eval_summary_homography(dists_sa, thres):

    # Compute aucs
    auc_sa = cal_error_auc(dists_sa, thresholds=thres)

    summary = f'Hest AUC: a={auc_sa}\n\n'
    print(summary)
    return auc_sa[-1]

def scale_homography(sw, sh):
    return np.array([[sw,  0, 0],
                     [ 0, sh, 0],
                     [ 0,  0, 1]])

def eval_homography_my(
        matcher,
        match_pairs,
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
    dists_all = []
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
    for pair_idx, (query, refer, gd) in tqdm(enumerate(match_pairs), total=len(match_pairs), smoothing=.5):
        if debug and pair_idx > 10:
            break

        im1_path = query
        im2_path = refer
        # Eval on composed pairs within seq
        image_num += 1

        # Predict matches
        try:
            t0 = time.time()
            if im1_path == './data/datasets/copy/query/674_2.jpg':
                a=1
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
                from PIL import Image
                im = Image.open(im1_path)
                w1, h1 = im.size
                im = Image.open(im2_path)
                w2, h2 = im.size
                points_gd = np.loadtxt(gd)
                raw = np.zeros([len(points_gd), 2])
                dst = np.zeros([len(points_gd), 2])
                raw[:, 0] = points_gd[:, 0] * w1
                raw[:, 1] = points_gd[:, 1] * h1
                dst[:, 0] = points_gd[:, 2] * w2
                dst[:, 1] = points_gd[:, 3] * h2
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
                if mae > 10 or mee > 5:
                    inaccurate += 1

                if avg_dist > 100000:
                    kp0 = p1s
                    kp1 = p2s
                    if scale_H:
                        # scale = (wo / wt, ho / ht) for im1 & im2
                        scale = match_res[4]
                        sc1 = scale[:2]
                        sc2 = scale[2:]
                        kp0 = sc1 * p1s
                        kp1 = sc2 * p2s
                    kp10, kp1 = torch.from_numpy(kp0), torch.from_numpy(kp1)

                    print(im1_path, im2_path, avg_dist)
                    # if im2_path == './data/datasets/hpatches-sequences-release/i_bridger/6.ppm':
                    import matplotlib.pyplot as plt
                    im1 = cv2.imread(im1_path, 0)
                    im2 = cv2.imread(im2_path, 0)
                    w = im1.shape[1]
                    plt.figure(dpi=200)
                    kp0_kp = [cv2.KeyPoint(int(k[0]), int(k[1]), 30) for k in kp0]
                    kp1_kp = [cv2.KeyPoint(int(k[0]), int(k[1]), 30) for k in kp1]
                    matches = [cv2.DMatch(_trainIdx=i, _queryIdx=i, _distance=1, _imgIdx=-1) for i in range(len(kp0))]
                    show = cv2.drawMatches(im1, kp0_kp,
                                           im2, kp1_kp, matches, None)
                    plt.imshow(show)
                    plt.title(f'error:{avg_dist}')

                    plt.scatter(dst.squeeze()[:, 0]+w, dst.squeeze()[:, 1], s=2, c='g')
                    plt.scatter(dst_pred.squeeze()[:, 0]+w, dst.squeeze()[:, 1], s=2, c='r')
                    plt.show()
            inlier_ratio.append(irat)

            dists_all.append(avg_dist)

    lprint_(
        f'>>Finished, pairs={len(match_time)} match_failed={match_failed} matches={np.mean(n_matches):.1f} match_time={np.mean(match_time):.2f}s')
    if first_match != 0:
        print('----------------------------------------------------------------------------------------')
        # print(f'first_matches_num: {first_match_num / first_match}, first_ransac_num: {first_ransac_num / first_match}')
        print('----------------------------------------------------------------------------------------')

    lprint_('==== Homography Estimation ====')
    lprint_(
        f'Hest solver={h_solver} est_failed={h_failed} ransac_thres={ransac_thres} inlier_rate={np.mean(inlier_ratio):.2f}')
    mauc = eval_summary_homography(dists_all, [3, 5, 10])

    print('-' * 40)
    print(f"Failed:{'%.2f' % (100 * failed / image_num)}%, Inaccurate:{'%.2f' % (100 * inaccurate / image_num)}%, "
          f"Acceptable:{'%.2f' % (100 * (image_num - inaccurate - failed) / image_num)}%")

    print('-' * 40)

    return mauc

def eval_cls_my(
        matcher,
        match_pairs,
        method='',
        task='class',
        scale_H=False,
        h_solver='degensac',
        ransac_thres=2,
        lprint_=print,
        debug=False,
):
    np.set_printoptions(precision=4)

    assert task == 'class'
    lprint_(f'\n>>>>Eval my: task={task} method={method} scale_H={scale_H} rthres={ransac_thres}')
    # Homography

    inliers_num_list = []
    classes = []


    match_failed = 0
    n_matches = []
    match_time = []
    start_time = time.time()
    for pair_idx, (query, refer, lb) in tqdm(enumerate(match_pairs), total=len(match_pairs), smoothing=.5):
        if debug and pair_idx > 10:
            break
        # if lb == '1':
        #     continue
        im1_path = query
        im2_path = refer
        # Eval on composed pairs within seq

        # Predict matches
        try:
            t0 = time.time()
            if im1_path == './data/datasets/copy/query/674_2.jpg':
                a=1
            match_res = matcher(im1_path, im2_path)
            match_time.append(time.time() - t0)
            matches, p1s, p2s = match_res[0:3]
        except Exception as e:
            print(str(e))
            p1s = p2s = matches = []
            match_failed += 1
        n_matches.append(len(matches))

        try:

            if 'cv' in h_solver:
                H_pred, inliers = cv2.findHomography(matches[:, :2], matches[:, 2:4], cv2.RANSAC, ransac_thres)
            else:
                H_pred, inliers = pydegensac.findHomography(matches[:, :2], matches[:, 2:4], ransac_thres)


        except:
            H_pred = None

        if H_pred is None:
            inliers_num = 0
        else:
            inliers_num = inliers.sum()
        # if (int(lb) == 0 and inliers_num > 4) or (int(lb) == 1 and inliers_num <= 10):
        # print(im1_path, im2_path, avg_dist)
        #     import matplotlib.pyplot as plt
        #     kp0 = p1s
        #     kp1 = p2s
        #
        #     # scale = (wo / wt, ho / ht) for im1 & im2
        #     scale = match_res[4]
        #     sc1 = scale[:2]
        #     sc2 = scale[2:]
        #     kp0 = sc1 * p1s
        #     kp1 = sc2 * p2s
        #     im1 = cv2.imread(im1_path, 0)
        #     im2 = cv2.imread(im2_path, 0)
        #     w = im1.shape[1]
            # plt.figure(dpi=200)
            # kp0_kp = [cv2.KeyPoint(int(k[0]), int(k[1]), 30) for k in kp0]
            # kp1_kp = [cv2.KeyPoint(int(k[0]), int(k[1]), 30) for k in kp1]
            # matches = [cv2.DMatch(_trainIdx=i, _queryIdx=i, _distance=1, _imgIdx=-1) for i in range(len(kp0))]
            # show = cv2.drawMatches(im1, kp0_kp,
            #                        im2, kp1_kp, matches, None)
            # plt.imshow(show)
            # plt.show()
        inliers_num_list.append(inliers_num)
        classes.append(int(lb))

    inliers = np.array(inliers_num_list)
    classes = np.array(classes)

    fpr, tpr, threshold = roc_curve(classes, inliers)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, threshold)(eer)

    print('--------------------------------------------------')
    print('EER: %.2f%%, threshold: %d' % (eer * 100, thresh))

    return eer


def eval_my(
        matcher,
        match_pairs,
        method='',
        task='homography',
        scale_H=False,
        h_solver='degensac',
        ransac_thres=2,
        lprint_=print,
        debug=False,
):
    if task == 'homography':
        return eval_homography_my(
            matcher,
            match_pairs,
            method=method,
            task=task,
            scale_H=scale_H,
            h_solver=h_solver,
            ransac_thres=ransac_thres,
            lprint_=lprint_,
            debug=debug,
        )
    else:
        return eval_cls_my(
            matcher,
            match_pairs,
            method=method,
            task=task,
            scale_H=scale_H,
            h_solver=h_solver,
            ransac_thres=ransac_thres,
            lprint_=lprint_,
            debug=debug,
        )