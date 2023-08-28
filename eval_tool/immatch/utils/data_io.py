from PIL import Image
import cv2
import numpy as np
import torchvision.transforms as transforms

from utils.common_utils import pre_processing


def lprint(ms, log=None):
    '''Print message on console and in a log file'''
    print(ms)
    if log:
        log.write(ms+'\n')
        log.flush()

def resize_im(wo, ho, imsize=None, dfactor=1, value_to_scale=max, aspan=False):
    wt, ht = wo, ho
    if imsize and (aspan or value_to_scale(wo, ho) > imsize) and imsize > 0:
        scale = imsize / value_to_scale(wo, ho)
        ht, wt = int(round(ho * scale)), int(round(wo * scale))

    # Make sure new sizes are divisible by the given factor
    wt, ht = map(lambda x: int(x // dfactor * dfactor), [wt, ht])

    scale = (wo / wt, ho / ht)
    return wt, ht, scale

def read_im(im_path, imsize=None, dfactor=1, value_to_scale=min):
    im = Image.open(im_path)
    im = im.convert('RGB')

    # Resize
    wo, ho = im.width, im.height
    wt, ht, scale = resize_im(wo, ho, imsize=imsize, dfactor=dfactor, value_to_scale=value_to_scale)
    im = im.resize((wt, ht), Image.BICUBIC)
    return im, scale

def read_im_gray(im_path, imsize=None):
    im, scale = read_im(im_path, imsize)
    return im.convert('L'), scale

def load_gray_scale_tensor(im_path, device, imsize=None, dfactor=1, value_to_scale=max):
    im_rgb, scale = read_im(im_path, imsize, dfactor=dfactor, value_to_scale=value_to_scale)
    gray = np.array(im_rgb.convert('L'))
    gray = transforms.functional.to_tensor(gray).unsqueeze(0).to(device)
    return gray, scale

def load_gray_scale_tensor_cv(im_path, device, imsize=None, dfactor=1, enhanced=False, value_to_scale=min, aspan=False):
    # Used for LoFTR

    im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)

    ho, wo = im.shape
    wt, ht, scale = resize_im(wo, ho, imsize=imsize, dfactor=dfactor, value_to_scale=value_to_scale, aspan=aspan)
    # if aspan and min(wt, ht) < 400:
    #     wt, ht = 2*wt, 2*ht
    #     new_scale = (scale[0]*0.5, scale[1]*0.5)
    #     scale = new_scale
    im = cv2.resize(im, (wt, ht))

    im = transforms.functional.to_tensor(im).unsqueeze(0).to(device)
    return im, scale

def load_im_tensor(im_path, device, imsize=None, normalize=True,
                   with_gray=False, raw_gray=False, dfactor=1):
    im_rgb, scale = read_im(im_path, imsize, dfactor=dfactor)

    # RGB  
    im = transforms.functional.to_tensor(im_rgb)
    if normalize:
        im = transforms.functional.normalize(
            im , mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    im = im.unsqueeze(0).to(device)
    
    if with_gray:
        # Grey
        gray = np.array(im_rgb.convert('L'))
        if not raw_gray:
            gray = transforms.functional.to_tensor(gray).unsqueeze(0).to(device)
        return im, gray, scale
    return im, scale
