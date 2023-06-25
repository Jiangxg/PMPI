# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

''' Modified based on: https://github.com/facebookresearch/NSVF '''
from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import os, sys
import torch
import torch.nn.functional as F
from torch.autograd import Function
import torch.nn as nn
import sys
import numpy as np

try:
    import builtins
except:
    import __builtin__ as builtins

try:
    import aabb._ext as _ext
except ImportError:
    pass
    # raise ImportError(
    #     "Could not import _ext module.\n"
    #     "Please see the setup instructions in the README"
    # )

'''
class AABBRayIntersect(Function):
    @staticmethod
    def forward(ctx, n_max, patches, ray_start, ray_dir, ray):
        # points is a tensor
        # train NSVF using a batch size of 4 images and for each image we sample 2048 rays
        # HACK: speed-up ray-voxel intersection by batching...

        # ray_start: S, N, 3 ---> S * G, K, 3

        sel = ray_dir.shape[0]
        rays = 2000 * ray

        #ray_dir (sel, 2) --> (2000 * ray, 2)
        if rays > sel:
            ray_dir = torch.cat([ray_dir, ray_dir[:rays-sel, :]], 0)

        # reshape 是否改变了内存顺序？
        ray_dir = ray_dir.reshape(2000, ray, 2).contiguous()
        patches = patches.expand(2000, *patches.size()[1:]).contiguous()
        ray_start = ray_start.expand(2000, *ray_start.size()[1:]).contiguous()

        # patches: [2000, num, 4] 最好是按照深度升序排列
        # ray_start: [2000, 3, 1]
        # ray_dir: [2000, ray, 2]

        #points: tensor [2000, 4, n_max, 3] 最后一维包含了光线与patch交点的参考相机坐标系坐标
        points = _ext.aabb_intersect(
            ray_start.float(), ray_dir.float(), patches.float(), n_max)
        
        #points: 8000, n_max, 3
        points = points.type_as(ray_start).reshape(-1, points.shape[2], 3)

        #points: sel, n_max, 3
        if rays > sel:
            points = points[:sel, :, :]
        
        #????
        #ctx.mark_non_differentiable(points)

        return points

    @staticmethod
    def backward(ctx, a, b, c):
        return None, None, None, None, None

aabb_ray_intersect = AABBRayIntersect.apply

'''

def aabb_ray_intersect(n_max, patches, ray_start, ray_dir, ray):
    sel = ray_dir.shape[0]
    rays = 2000 * ray

    #ray_dir (sel, 2) --> (2000 * ray, 2)
    if rays > sel:
        ray_dir = torch.cat([ray_dir, torch.ones(rays-sel, 2).cuda()], 0)

    # reshape 是否改变了内存顺序？
    ray_dir = ray_dir.reshape(2000, ray, 2).contiguous()
    patches = patches.expand(2000, *patches.size()[1:]).contiguous()
    ray_start = ray_start.expand(2000, *ray_start.size()[1:]).contiguous()

    # patches: [2000, num, 4] 最好是按照深度升序排列
    # ray_start: [2000, 3, 1]
    # ray_dir: [2000, ray, 2]

    #points: tensor [2000, 4, n_max, 3] 最后一维包含了光线与patch交点的参考相机坐标系坐标
    points = _ext.aabb_intersect(
        ray_start.float(), ray_dir.float(), patches.float(), n_max)
        
    #points: 8000, n_max, 3
    points = points.type_as(ray_start).reshape(-1, points.shape[2], 3)

    #points: sel, n_max, 3
    if rays > sel:
        points = points[:sel, :, :]

    return points