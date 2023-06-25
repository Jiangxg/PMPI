# PMPI: Patch-based Multiplane Images for learning View Synthesis

We present PMPI, an explicit and plane-based model for real-time view synthesis. Our PMPI
achieves more dense and effective samplings without adding computing resources by assembling patches around visible contents.

## PMPI model
![image](https://github.com/Jiangxg/PMPI/assets/41377695/e7f1edcc-6933-4ff4-91e7-1245b3d2a980)

## Our method for learning PMPI
![image](https://github.com/Jiangxg/PMPI/assets/41377695/0a80d94c-1bac-46e1-b562-b009656c3d48)


## Table of contents
-----
  * [Installation](#Installation)
  * [Dataset](#Dataset)
  * [Training](#Training)
  * [Rendering](#Rendering)
  * [Citation](#citation)
------

## Installation
We provide `environment.yml` to help you setup a conda environment. 

```shell
conda env create -f environment.yml
```

## Dataset
### NeRF's  real forward-facing dataset
**Download:** [Undistorted front facing dataset](https://vistec-my.sharepoint.com/:f:/g/personal/pakkapon_p_s19_vistec_ac_th/ErjPRRL9JnFIp8MN6d1jEuoB3XVoxJkffPjfoPyhHkj0dg?e=qIunN0)

For real forward-facing dataset, NeRF is trained with the raw images, which may contain lens distortion. But we use the undistorted images provided by COLMAP.

However, you can try running other scenes from [Local lightfield fusion](https://github.com/Fyusion/LLFF) (Eg. [airplant](https://github.com/Fyusion/LLFF/blob/master/imgs/viewer.gif)) without any changes in the dataset files. In this case, the images are not automatically undistorted.

### Using your own images.

Running PMPI on your own images. You need to install [COLMAP](https://colmap.github.io/) on your machine.

Then, put your images into a directory following this structure
```
<scene_name>
|-- images
     | -- image_name1.jpg
     | -- image_name2.jpg
     ...
```

The training code will automatically prepare a scene for you. You may have to tune `planes.txt` to get better reconstruction (see [dataset explaination](https://vistec-my.sharepoint.com/:t:/g/personal/pakkapon_p_s19_vistec_ac_th/EYBtE-X95pFLscoLFehUMtQBjrrYKQ9mxVEzKzNlDuoZLw?e=bODHZ4))


## Training

python train_progressive.py -scene data/forward/fortress_undistort -model_dir model/fortress_192layers_c2f -epochs 4000 -before_decay 0 -n_max 384 -ray 4 -cv2resize -layers 192 -sublayers 1 -size_patch 36 -offset 216 -tb_toc 100 -gamma 0.0006 -all_gpu

```shell
python train.py -scene ${PATH_TO_SCENE} -model_dir ${MODEL_TO_SAVE_CHECKPOINT} -epochs 4000 -before_decay 0 -n_max 384 -ray 4  -layers 192 -sublayers 1 -size_patch 36 -offset 216  -gamma 0.0006 -all_gpu
```

This implementation uses [scikit-image](https://scikit-image.org/) to resize images during training by default. The results and scores in the paper are generated using OpenCV's resize function. If you want the same behavior, please add `-cv2resize` argument.

Note that this code is tested on two Nvidia V100 32GB.

For a GPU/GPUs with less memory (e.g., a single RTX 3090), you can run using the following command:
```shell
python train.py -scene ${PATH_TO_SCENE} -model_dir ${MODEL_TO_SAVE_CHECKPOINT} -epochs 4000 -before_decay 0 -n_max 384 -ray 4  -layers 128 -sublayers 1 -size_patch 36 -offset 216  -gamma 0.0006
```
Note that when your GPU runs out of memeory, you can try reducing the number of layers, sublayers, and sampled rays.

## Citation


## Acknowledge
Our repo is developed based on [NeX](https://github.com/nex-mpi/nex-code/) and [NSVF](https://github.com/facebookresearch/NSVF)
