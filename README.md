# GeoFormer for Homography Estimation

[//]: # ([![PWC]&#40;https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/semi-supervised-keypoint-detector-and/image-registration-on-fire&#41;]&#40;https://paperswithcode.com/sota/image-registration-on-fire?p=semi-supervised-keypoint-detector-and&#41;)

This is the official source code of our ICCV2023 paper: [Geometrized Transformer for Self-Supervised Homography Estimation](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Geometrized_Transformer_for_Self-Supervised_Homography_Estimation_ICCV_2023_paper.html).

![illustration](./image/fig-model.jpg)

## Environment
We used Anaconda to setup a deep learning workspace that supports PyTorch. Run the following script to install all the required packages.

``` conda
conda create -n GeoFormer python==3.8 -y
conda activate GeoFormer
git clone https://github.com/ruc-aimc-lab/GeoFormer.git
cd GeoFormer
pip install -r requirements.txt
```

## Downloads

### Data
#### For Training
GeoFormer can be trained on the artificially synthesized dataset OxFord-Paris, as well as on Megadepth dataset with depth labels included. 
You need to organize the data according to the specified format.
+ OxFord-Paris, you can directly download the [OxFord 5K](https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images.tgz) and [Paris 6K](http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_1.tgz) and extract them into one folder and move to the 'data/' directory;
+ Megadepth, you need to follow the process outlined in the [LoFTR](https://github.com/zju3dv/LoFTR/blob/master/docs/TRAINING.md) project  to organize it from scratch.
The training data should be organized as follows.
```
data/
    datasets/
        Oxford-Paris/
            oxbuild_images/ 
                all_souls_000000.jpg
                ....
            paris/
                defense/
                eiffle/
                ....  
        Megadepth/
            index/
            train/
            test/

```
#### For Testing

+ FIRE: [https://projects.ics.forth.gr/cvrl/fire/](https://projects.ics.forth.gr/cvrl/fire/)
+ ISC-HE: [https://drive.google.com/drive/folders/1giglxwMGlOb3qE-5AQ60R4495KxRGTqb](https://drive.google.com/drive/folders/1giglxwMGlOb3qE-5AQ60R4495KxRGTqb)
+ Hpatches: [http://icvl.ee.ic.ac.uk/vbalnt/hpatches/](http://icvl.ee.ic.ac.uk/vbalnt/hpatches/)

The file organizations are as follows:

```
data/
    datasets/
        FIRE/
            Ground Truth/
            Images/
            Masks/
        
        hpatches-sequences-release/
            i_ajuntament/
            ....
            
        ISC-HE/
            gd/
            query/
            refer/
            index.txt
```
+ Note that the annotation file of `control_points_P37_1_2.txt` in `FIRE` dataset is incorrect, so it shall be excluded from evaluation.


### Models

You may skip the training stage and use our provided models for homography estimation.
+ [Google drive](https://drive.google.com/drive/folders/1giglxwMGlOb3qE-5AQ60R4495KxRGTqb)

Put the trained model geoformer.ckpt into `saved_ckpt/` folder.

The model config can be found in [model/geo_config.py](model/geo_config.py)

## Code

### Training 

Before training GeoFormer,
the parameters of the dataset can be modified in these two files. [train_config/homo_trainval_640.py](train_config/homo_trainval_640.py) file or [train_config/megadepth_trainval_640.py](train_config/megadepth_trainval_640.py) file 
And the training hyperparameters are adjusted here [train_config/loftr_ds_dense.py](train_config/loftr_ds_dense.py).


```
python -m lightning/train_depth_geoformer
or
python -m lightning/train_homo_geoformer
```

### Inference

#### Homography Estimation Performance

Our evaluation code is implemented based on the foundation of this benchmark.
+ [immatch](https://github.com/GrumpyZhou/image-matching-toolbox/tree/main/immatch)

The [eval_Hpatches.py](eval_Hpatches.py) code shows how homography estimation is performed on the Hpatches dataset.
```
python eval_Hpatches.py
```
If everything goes well, you shall see the following message on your screen:
```
==== Homography Estimation ====
Hest solver=cv est_failed=0 ransac_thres=3 inlier_rate=0.89
Hest Correct: a=[0.7    0.8845 0.9379 0.9603]
i=[0.8772 0.986  0.9965 0.9965]
v=[0.5288 0.7864 0.8814 0.9254]
Hest AUC: a=[0.5154 0.7206 0.7997 0.8768]
i=[0.7634 0.8913 0.9319 0.9642]
v=[0.2774 0.5572 0.6735 0.7931]
```

The other test sets can be evaluated similarly.
```
python eval_FIRE.py

python eval_ISC.py
```
---

#### One Pair Inference
If you just want to input one pair and obtain the matching results, you can just run the infer code [inference.py](inference.py).
## Citations
If you find this repository useful, please consider citing:
```
@inproceedings{liu2022SuperRetina,
  title={Geometrized Transformer for Self-Supervised Homography Estimation},
  author={Jiazhen Liu and Xirong Li},
  booktitle={ICCV},
  year={2023}
}
```

## Contact
If you encounter any issue when running the code, please feel free to reach us either by creating a new issue in the GitHub or by emailing

+ Jiazhen Liu (liujiazhen@ruc.edu.cn)

