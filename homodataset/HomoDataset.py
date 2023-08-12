# -*- coding: utf-8 -*-
import os
import random
import time
import kornia
import imgaug.augmenters as iaa
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
from utils.homography import sample_homography, compute_valid_mask

from warnings import filterwarnings

from utils.preprocess_utils import resize_aspect_ratio, get_perspective_mat, scale_homography

filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')


class HomoDataset(Dataset):
    def __init__(self, img_dir, size=(640, 480), st=32, rank=0, word_size=None):
        self.img_dir = img_dir
        self.size = size
        imgs = []
        self.st = st
        # imgs = os.listdir(img_dir)
        for curDir, dirs, files in os.walk(img_dir):
            fs = [os.path.join(curDir, x) for x in files]

            # fs = os.listdir(image_path)

            for i in fs:
                (path, filename) = os.path.split(i)

                if (i.endswith('.jpg') or i.endswith('.ppm')):
                    imgs.append(i)

        self.data = imgs[:]
        if word_size is not None:
            bz = int(len(self.data) // word_size)
            start = rank * bz
            end = start + bz if start + bz < len(self.data) else len(self.data)
            # print(start, end)
            self.data = self.data[start:end]

        # self.num_samples = len(self.data)

        self.model_image_height, self.model_image_width = size
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])

        import albumentations as alb
        self.aug_list = [
            alb.OneOf([alb.RandomBrightness(limit=0.2, p=0.8), alb.RandomContrast(limit=0.3, p=0.6)], p=0.5),
            alb.OneOf([alb.MotionBlur(p=0.5), alb.GaussNoise(p=0.6)], p=0.5),
            ]
        self.aug_func = alb.Compose(self.aug_list, p=0.65)
        self.apply_aug = True
        self.config = {
            'apply_color_aug': True,  # whether to apply photometric distortions
            'image_height': size[0],
            'image_width': size[1],
            'augmentation_params':{
                'patch_ratio': 0.8,
                'translation': 0.2,  # translation component range
            }
        }
        self.aug_params = self.config['augmentation_params']

    def __len__(self):
        return len(self.data)


    def apply_augmentations(self, image1, image2):
        image1_dict = {'image': image1}
        image2_dict = {'image': image2}
        result1, result2 = self.aug_func(**image1_dict), self.aug_func(**image2_dict)
        return result1['image'], result2['image']

    def get_pair(self, file_path):
        resize = True
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        height, width = image.shape[0:2]
        homo_matrix = None
        while homo_matrix is None:
            homo_matrix = get_perspective_mat(self.aug_params['patch_ratio'], width, height,
                                              self.aug_params['translation'])
            try:
                torch.inverse(torch.from_numpy(homo_matrix))
            except:
                homo_matrix = None
        warped_image = cv2.warpPerspective(image.copy(), homo_matrix, (width, height))

        if self.st == 0:
            if height > width:
                self.config['image_width'], self.config['image_height'] = self.size[1], self.size[0]
            else:
                self.config['image_width'], self.config['image_height'] = self.size[0], self.size[1]
        else:
            if height > width:
                self.config['image_height'] = self.size[0]
                self.config['image_width'] = int((self.config['image_height'] / height * width) // self.st * self.st)
            else:
                self.config['image_width'] = self.size[1]
                self.config['image_height'] = int((self.config['image_width'] / width * height) // self.st * self.st)

        if resize:
            orig_resized = cv2.resize(image, (self.config['image_width'], self.config['image_height']))
            warped_resized = cv2.resize(warped_image, (self.config['image_width'], self.config['image_height']))
        else:
            orig_resized = image
            warped_resized = warped_image
        if self.apply_aug:
            orig_resized, warped_resized = self.apply_augmentations(orig_resized, warped_resized)
        homo_matrix = scale_homography(homo_matrix, height, width, self.config['image_height'],
                                       self.config['image_width']).astype(np.float32)
        orig_resized = np.expand_dims(orig_resized, 0).astype(np.float32) / 255.0
        warped_resized = np.expand_dims(warped_resized, 0).astype(np.float32) / 255.0
        return orig_resized, warped_resized, homo_matrix

    def get_images_labels(self, index):
        is_neg = False
        file_path = self.data[index]
        # sample = self.data[index]
        # file_path = os.path.join(self.img_dir, sample)
        image0, image1, homography = self.get_pair(file_path)
        homography = torch.from_numpy(homography).unsqueeze(0)
        if np.random.uniform(0, 1) < 0.:  # negative
            is_neg = True
            neg_path = self.get_neg_sample(index, False)
            neg0, neg1, _ = self.get_pair(neg_path)
            image1 = neg0
            if np.random.uniform(0, 1) < 0.3:
                image1 = neg1
            valid_mask_left = torch.zeros([1, image0.shape[1], image0.shape[2]])
            valid_mask_right = torch.zeros([1, image1.shape[1], image1.shape[2]])
            homography = torch.eye(3).unsqueeze(0)
            return image0, image1, homography, valid_mask_left, valid_mask_right, is_neg
        valid_mask_right = compute_valid_mask(image0.shape[1:], homography)
        valid_mask_left = kornia.geometry.transform.warp_perspective(valid_mask_right.unsqueeze(0),
                                                                     torch.inverse(homography),
                                                                     image1.shape[1:],
                                                                     align_corners=True)[0]
        # valid_mask_left = valid_mask.clone()
        # valid_mask_left[:] = 1
        # valid_mask_right = valid_mask
        if np.random.uniform(0, 1) < 0.5:
            homography = torch.inverse(homography)
            tmp = image1
            image1 = image0
            image0 = tmp
            valid_mask = valid_mask_right
            valid_mask_right = valid_mask_left
            valid_mask_left = valid_mask

        return image0, image1, homography, valid_mask_left, valid_mask_right, is_neg
    def __getitem__(self, index: int):
        image0, image1, homography, valid_mask_left, valid_mask_right, is_neg = self.get_images_labels(index)

        name = self.data[index]
        name = os.path.split(name)[-1]
        data = {
            'image0': image0,  # (1, h, w)
            'image1': image1,
            'H_0to1': homography[0],  # (1, 3, 3)
            'H_1to0': torch.inverse(homography)[0],
            'valid_mask_left': valid_mask_left,
            'valid_mask_right': valid_mask_right,
            'is_negs': is_neg,
            'dataset_name': 'Oxford',
            'pair_id': index,
            'pair_names': (name + '_0', name + '_1'),
        }
        return data
        # return image0, image1, homography, valid_mask_left, valid_mask_right, is_neg

    # def __getitem__(self, idx):
    #     sample = self.data[idx]
    #
    #     raw_path = os.path.join(self.img_dir, sample)
    #     raw_img = cv2.imread(raw_path)
    #     model_h, model_w = self.model_image_height, self.model_image_width
    #     # if seg_img.shape[0] != h or seg_img.shape[1] != w:
    #     is_neg = False
    #     raw_img = cv2.resize(raw_img, (model_w, model_h))
    #     h, w, _ = raw_img.shape
    #
    #     valid_mask = torch.zeros([h, w]).unsqueeze(0)
    #     image0, image1, homography = None, None, None
    #     if random.random() < 0.3:
    #         image0 = raw_img
    #         image1 = self.get_neg_sample(idx)
    #         image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
    #         image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    #         # image0 = np.expand_dims(image0, axis=0).astype(np.uint8)
    #         # image1 = np.expand_dims(image1, axis=0).astype(np.uint8)
    #         # image0, image1 = self.seq(images=image0)[0], \
    #         #     self.seq(images=image1)[0]
    #         image0, image1 = self.apply_augmentations(image0, image1)
    #
    #         image0, image1 = self.transforms(image0), self.transforms(image1)
    #         homography = torch.eye(3).unsqueeze(0)
    #         valid_mask_left = valid_mask.clone()
    #         valid_mask_right = valid_mask
    #         is_neg = True
    #         return image0, image1, homography, valid_mask_left, valid_mask_right, is_neg
    #     while (valid_mask > 0).sum() < 10000:
    #         image0, image1, homography, valid_mask = self.get_pos_sample(raw_img)
    #
    #     # image0 = np.expand_dims(image0, axis=0).astype(np.uint8)
    #     # image1 = np.expand_dims(image1, axis=0).astype(np.uint8)
    #     # back_img = self.get_neg_sample(idx)
    #     # back_img = cv2.cvtColor(back_img, cv2.COLOR_BGR2GRAY)
    #     # back_img = np.expand_dims(back_img, axis=0).astype(np.float32) / 255
    #     # back_img[valid_mask == 1] = 0
    #     # image1 = image1 + back_img
    #     # image0, image1 = self.seq(images=image0)[0], \
    #     #     self.seq(images=image1)[0]
    #     image0, image1 = self.apply_augmentations(image0, image1)
    #
    #     image0, image1 = self.transforms(image0), self.transforms(image1)
    #     valid_mask_left = valid_mask.clone()
    #     valid_mask_left[:] = 1
    #     valid_mask_right = valid_mask
    #     if random.random() < 0.5:
    #         homography = torch.inverse(homography)
    #         tmp = image1
    #         image1 = image0
    #         image0 = tmp
    #         valid_mask_right = valid_mask_left
    #         valid_mask_left = valid_mask
    #
    #     return image0, image1, homography, valid_mask_left, valid_mask_right, is_neg

    def on_the_fly(self, query, mask=None, mode=0):
        m = mask
        image_shape = query.shape[:2]
        homography = sample_homography(image_shape, 'cpu', mode=mode)
        valid_mask = compute_valid_mask(image_shape, homography)

        r = cv2.warpPerspective(query, homography.squeeze().numpy(), tuple(image_shape[::-1]))
        if mask is not None:
            m = cv2.warpPerspective(mask, homography.squeeze().numpy(), tuple(image_shape[::-1]))
            m[m < 10] = 0
        r[r < 10] = 0

        return r, m, homography, valid_mask

    def get_pos_sample(self, raw_img):

        image1, _, homography, valid_mask = self.on_the_fly(raw_img)

        # 构成pair
        # image0 = np.where(seg_img > 0, raw_img, 0)
        image0 = raw_img
        # image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
        # image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        return image0, image1, homography, valid_mask
        # pos_pair = cv2.hconcat([letterbox(image0, self.size, self.size), letterbox(image1, self.size, self.size)])
        # return pos_pair

    # def get_neg_sample(self, raw_img, seg_img, raw_url):
    #     # 负样本生成方式
    #     # 方式一：随机图
    #     # 方式二：同商品图片
    #     # 方式三：抠图 + 非仿射的透视变换
    #     mode = random.choice(self.sample_mode)
    #     if mode == 'easy':
    #         right_img = self.get_rand_img()
    #     elif mode == 'normal':
    #         pid = self.url2pid[raw_url]
    #         urls = self.pid2url[pid]
    #         valid_urls = [url for url in urls if url != raw_url]
    #         if len(valid_urls) > 0:
    #             right_img = self.get_rand_img(random.choice(valid_urls))
    #         else:
    #             right_img = self.get_rand_img()
    #     else:
    #         valid_connect_stats, mask_count = cal_connect_info(seg_img)
    #         # 抠图
    #         obj_img, obj_mask = get_obj(raw_img, seg_img, valid_connect_stats, mask_count)
    #         obj_img = np.where(obj_mask > 63, obj_img, 0)
    #         # 非仿射的透视变换
    #         obj_img = rand_transform(obj_img)
    #         # 嵌入整图
    #         back_img = self.get_rand_img()
    #         right_img = mosaic_obj(obj_img, back_img)
    #
    #     # 构成pair
    #     left_img = np.where(seg_img > 0, raw_img, 0)
    #     neg_pair = cv2.hconcat([letterbox(left_img, self.size, self.size), letterbox(right_img, self.size, self.size)])
    #     return neg_pair


