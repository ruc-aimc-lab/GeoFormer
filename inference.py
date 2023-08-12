
import torch
import cv2
import numpy as np
from torchvision import transforms

# loftr_config = {
#     "nms_size": 10,
#     "nms_thresh": 0.1,
#     "cont_thresh": 0.8,
#     "geo_thresh": 0.5,
#     "image_shape": (640, 480),
#     'layer_names': ['self', 'cross']*4,
#     "d_model": 256,
#     'initial_dim': 128,
#     'block_dims': [128, 256, 256],
#     'nhead': 4,
#     'max_num': 1024,
#     'sinkhorn_iterations': 5,
#     'is_train': True
# }
my_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            # transforms.Normalize(0.5, 0.5)
        ])
# model = SDMatchNet(loftr_config)
# state_dict = torch.load('save/saved_model.pth', map_location=torch.device('cpu'))
# model.load_state_dict(state_dict)
# h, w = loftr_config['image_shape']
# a = cv2.imread('samples/1.jpg')
# b = cv2.imread('samples/2.jpg')
# a = cv2.resize(a, (w, h))
# b = cv2.resize(b, (w, h))

def inference(model, query, refer, seg=None, device='cpu', deal_data=True):
    if deal_data:
        query = my_transforms(query).unsqueeze(0).to(device)
        refer = my_transforms(refer).unsqueeze(0).to(device)

    # model.eval()
    if seg is not None:
        kp0, kp1, raw_kp0, raw_kp1, fine_kp0, fine_kp1 = model(query, refer, seg)
    else:
        kp0, kp1, raw_kp0, raw_kp1, fine_kp0, fine_kp1 = model(query, refer)

    # try:
    #     # kp0, kp1 = coarse_results[0]['coarse_kps0'].cpu(), coarse_results[0]['coarse_kps1'].cpu()
    #     # raw_kp0, raw_kp1 = coarse_results[0]['raw_kps0'].cpu(), coarse_results[0]['raw_kps1'].cpu()
    #     # fine_kp0, fine_kp1 = fine_results[0]['fine_kps0'].cpu(), fine_results[0]['fine_kps1'].cpu()
    #     kp0, kp1 = coarse_results[0]['coarse_kps0'].cpu(), coarse_results[0]['coarse_kps1'].cpu()
    #     raw_kp0, raw_kp1 = coarse_results[0]['raw_kps0'].cpu(), coarse_results[0]['raw_kps1'].cpu()
    #     fine_kp0, fine_kp1 = fine_results[0]['fine_kps0'].cpu(), fine_results[0]['fine_kps1'].cpu()
    # except Exception:
    #     kp0 = kp1 = []
    #     raw_kp0, raw_kp1 = [], []
    #     fine_kp0 = fine_kp1 = []


    # kp0 = [cv2.KeyPoint(int(k[0]), int(k[1]), 30) for k in kp0]
    # kp1 = [cv2.KeyPoint(int(k[0]), int(k[1]), 30) for k in kp1]
    # matches = [cv2.DMatch(_trainIdx=i, _queryIdx=i, _distance=1, _imgIdx=-1) for i in range(len(kp0))]

    fine_kp0 = [cv2.KeyPoint(int(k[0]), int(k[1]), 30) for k in fine_kp0]
    fine_kp1 = [cv2.KeyPoint(int(k[0]), int(k[1]), 30) for k in fine_kp1]
    fine_matches = [cv2.DMatch(_trainIdx=i, _queryIdx=i, _distance=1, _imgIdx=-1) for i in range(len(fine_kp0))]

    kp0 = [cv2.KeyPoint(int(k[0]), int(k[1]), 30) for k in kp0]
    kp1 = [cv2.KeyPoint(int(k[0]), int(k[1]), 30) for k in kp1]

    raw_kp0 = [cv2.KeyPoint(int(k[0]), int(k[1]), 30) for k in raw_kp0]
    raw_kp1 = [cv2.KeyPoint(int(k[0]), int(k[1]), 30) for k in raw_kp1]

    return fine_kp0, fine_kp1, fine_matches, raw_kp0, raw_kp1, kp0, kp1
    #
    src_pts = np.float32([kp0[m.queryIdx].pt for m in fine_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.trainIdx].pt for m in fine_matches]).reshape(-1, 1, 2)
    if len(src_pts) > 4:
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        ransac_matches = np.array(fine_matches)[mask.ravel()==1]

    if vis:
        show = cv2.drawMatches((query[0][0].numpy() * 255).astype(np.uint8), kp0,
                               (refer[0][0].numpy() * 255).astype(np.uint8), kp1, matches, None)
        plt.imshow(show)
        plt.show()

        show = cv2.drawMatches((query[0][0].numpy() * 255).astype(np.uint8), fine_kp0,
                               (refer[0][0].numpy() * 255).astype(np.uint8), fine_kp1, fine_matches, None)
        plt.imshow(show)
        plt.show()

        if len(src_pts) > 4:
            show = cv2.drawMatches((query[0][0].numpy() * 255).astype(np.uint8), fine_kp0,
                                   (refer[0][0].numpy() * 255).astype(np.uint8), fine_kp1, ransac_matches, None)
            plt.imshow(show)
            plt.show()