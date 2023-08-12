# -*- coding: utf-8 -*-
import cv2
import imghdr
import imageio
import requests
import numpy as np

def download_image(url, mode='binary', size=None, data_format=cv2.IMREAD_COLOR):
    img_binary, img = None, None
    req_count = 0
    while(req_count < 1):
        try:
            ret = requests.get(url, timeout=10)
            if ret.status_code == 200:
                img_binary = ret.content
                if mode == 'ndarray':
                    if imghdr.what(None, img_binary) != 'gif':
                        img = np.asarray(bytearray(img_binary), dtype="uint8")
                        img = cv2.imdecode(img, data_format)
                    else:
                        gifs = imageio.mimread(img_binary)
                        img = cv2.cvtColor(gifs[0], cv2.COLOR_RGB2BGR)
                    # 按图像高作为参照，对图像进行等比例缩放（避免图像失真）
                    if size is not None:
                        h, w = img.shape[0], img.shape[1]
                        w_new = int(w * size / h)
                        img = cv2.resize(img, (w_new, size))
            break
        except Exception as e:
            req_count += 1
            print(e)
    if mode == 'ndarray':
        return img
    return img_binary


def image_binary2ndarray(img_binary, size=None):
    try:
        if imghdr.what(None, img_binary) != 'gif':
            img = np.asarray(bytearray(img_binary), dtype="uint8")
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        else:
            gifs = imageio.mimread(img_binary)
            img = cv2.cvtColor(gifs[0], cv2.COLOR_RGB2BGR)
        if size is not None:
            h, w = img.shape[0], img.shape[1]
            w_new = int(w * size / h)
            img = cv2.resize(img, (w_new, size))
        return img
    except:
        return None

def image_ndarray2binary(img):
    b_img = cv2.imencode('.jpg', img)[1].tobytes()
    return b_img
