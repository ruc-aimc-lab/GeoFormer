from utils.homography import homographic_aug_pipline
import torch

def on_the_fly(query):
    labels = []
    homographies = []
    refer_tmp = []
    r_valid_masks = []
    for q in query:
        q = q.unsqueeze(0)
        r, r_valid_mask, homography = homographic_aug_pipline(q, device=q.device)
        refer_tmp.append(r)
        homographies.append(homography)
        r_valid_masks.append(r_valid_mask)
        labels.append(1)
    refer = torch.cat(refer_tmp)
    homographies = torch.cat(homographies)
    r_valid_masks = torch.cat(r_valid_masks)
    return refer, homographies, r_valid_masks, labels