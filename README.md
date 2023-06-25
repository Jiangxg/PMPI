# PMPI: Patch-based Multiplane Images for learning View Synthesis

We present PMPI, an explicit and plane-based model for real-time view synthesis. Our PMPI
achieves more dense and effective samplings without adding computing resources by assembling patches around visible contents.

## PMPI model
![image](https://github.com/Jiangxg/PMPI/assets/41377695/e7f1edcc-6933-4ff4-91e7-1245b3d2a980)

## Our method for learning PMPI
![image](https://github.com/Jiangxg/PMPI/assets/41377695/0a80d94c-1bac-46e1-b562-b009656c3d48)


## Table of contents
-----
  * [Getting started](#Getting-started)
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

Running NeX on your own images. You need to install [COLMAP](https://colmap.github.io/) on your machine.

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
python train_progressively.py -scene data/fern_undistort -model_dir model/fern_32 -layers 32 -sublayers 1 -cv2resize -hidden 256 -tb_toc 20 -ray2000 
Run with the paper's config
```shell
python train.py -scene ${PATH_TO_SCENE} -model_dir ${MODEL_TO_SAVE_CHECKPOINT} -layer ${NUM_OF_LAYERS} -sublayer ${NUM_OF_SUBLAYERS} -cv2resize
```

This implementation uses [scikit-image](https://scikit-image.org/) to resize images during training by default. The results and scores in the paper are generated using OpenCV's resize function. If you want the same behavior, please add `-cv2resize` argument.

Note that this code is tested on an Nvidia V100 32GB and 4x RTX 2080Ti GPU.

For a GPU/GPUs with less memory (e.g., a single RTX 2080Ti), you can run using the following command:
```shell
python train.py -scene ${PATH_TO_SCENE} -model_dir ${MODEL_TO_SAVE_CHECKPOINT} -http -layers 12 -sublayers 6 -hidden 256
```
Note that when your GPU runs ouut of memeory, you can try reducing the number of layers, sublayers, and sampled rays.

## Rendering

To generate a WebGL viewer and a video result.
```shell
python train.py -scene ${scene} -model_dir ${MODEL_TO_SAVE_CHECKPOINT} -predict -http
```

### Video rendering

To generate a video that matches the real forward-facing rendering path, add `-nice_llff` argument, or `-nice_shiny` for shiny dataset



## Citation

```
@inproceedings{Wizadwongsa2021NeX,
    author = {Wizadwongsa, Suttisak and Phongthawee, Pakkapon and Yenphraphai, Jiraphon and Suwajanakorn, Supasorn},
    title = {NeX: Real-time View Synthesis with Neural Basis Expansion},
    booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, 
    year = {2021},
}
```

## Visit us 🦉
[![Vision & Learning Laboratory](https://i.imgur.com/hQhkKhG.png)](https://vistec.ist/vision) [![VISTEC - Vidyasirimedhi Institute of Science and Technology](https://i.imgur.com/4wh8HQd.png)](https://vistec.ist/)
