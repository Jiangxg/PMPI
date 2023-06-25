# Copyright (c) 2021 VISTEC - Vidyasirimedhi Institute of Science and Technology
# Distribute under MIT License
# Authors:
#  - Suttisak Wizadwongsa <suttisak.w_s19[-at-]vistec.ac.th>
#  - Pakkapon Phongthawee <pakkapon.p_s19[-at-]vistec.ac.th>
#  - Jiraphon Yenphraphai <jiraphony_pro[-at-]vistec.ac.th>
#  - Supasorn Suwajanakorn <supasorn.s[-at-]vistec.ac.th>
# -*- coding:utf-8 -*-


#逐步缩小patch的尺寸

from __future__ import division
from __future__ import print_function

import argparse
import getpass
import struct
import time

import torch as pt
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter

import os, sys, json
import numpy as np
from skimage import io, transform
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

import scipy.io as io

from utils.utils import *
from utils.mpi_utils import *
from utils.mlp import *
from utils.colmap_runner import colmapGenPoses
from utils.colmap_runner import colmapGenPoses_modified

from aabb import aabb_ray_intersect

parser = argparse.ArgumentParser()

#training schedule
parser.add_argument('-epochs', type=int, default=7000, help='total epochs to train')
parser.add_argument('-steps', type=int, default=-1, help='total steps to train. In our paper, we proposed to use epoch instead.')
parser.add_argument('-tb_saveimage', type=int, default=50, help='write an output image to tensorboard for every <tb_saveimage> epochs')
parser.add_argument('-tb_savempi', type=int, default=200, help='generate MPI (WebGL) and measure PSNR/SSIM of validation image for every <tb_savempi> epochs')
parser.add_argument('-checkpoint', type=int, default=1000, help='save checkpoint for every <checkpoint> epochs. Be aware that! It will replace the previous checkpoint.')
parser.add_argument('-tb_toc',type=int, default=500, help="print output to terminal for every tb_toc epochs")

#lr schedule
parser.add_argument('-lrc', type=float, default=10, help='the number of times of lr using for learning rate of explicit basis (k0).')
parser.add_argument('-lr', type=float, default=1e-3, help='learning rate of a multi-layer perceptron')
parser.add_argument('-decay_epoch', type=int, default=1333, help='the number of epochs for decay learning rate')
parser.add_argument('-decay_rate', type=float, default=0.1, help='ratio of decay rate at every <decay_epoch> epochs')
parser.add_argument('-before_decay', type=int, default=3000, help='the number of epoches before decay learning rate')
parser.add_argument('-n_max', type=int, default=40, help='the max number of points in each ray')

#network (First MLP)
parser.add_argument('-ray', type=int, default=4, help='the 1/2000 number of sampled ray that is used to train in each step')
parser.add_argument('-hidden', type=int, default=384, help='the number of hidden node of the main MLP')
parser.add_argument('-mlp', type=int, default=4, help='the number of hidden layer of the main MLP')
parser.add_argument('-pos_level', type=int, default=10, help='the number of positional encoding in terms of image size. We recommend to set 2^(pos_level) > image_height and image_width')
parser.add_argument('-depth_level', type=int, default=8,help='the number of positional encoding in terms number of plane. We recommend to set 2^(depth_level) > layers * subplayers')
parser.add_argument('-lrelu_slope', type=float, default=0.01, help='slope of leaky relu')
parser.add_argument('-sigmoid_offset', type=float, default=5, help='sigmoid offset that is applied to alpha before sigmoid')

#basis (Second MLP)
parser.add_argument('-basis_hidden', type=int, default=64, help='the number of hidden node in the learned basis MLP')
parser.add_argument('-basis_mlp', type=int, default=1, help='the number of hidden layer in the learned basis MLP')
parser.add_argument('-basis_order', type=int, default=3, help='the number of  positional encoding in terms of viewing angle')
parser.add_argument('-basis_out', type=int, default=8, help='the number of coeffcient output (N in equation 3 under seftion 3.1)')

#loss
parser.add_argument('-gradloss', type=float, default=0.05, help='hyperparameter for grad loss')
parser.add_argument('-tvc', type=float, default=0.03, help='hyperparameter for total variation regularizer')

#training and eval data
parser.add_argument('-scene', type=str, default="", help='directory to the scene')
parser.add_argument('-ref_img', type=str, default="",  help='reference image, camera parameter of reference image is use to create MPI')
parser.add_argument('-dmin', type=float, default=-1, help='first plane depth')
parser.add_argument('-dmax', type=float, default=-1, help='last plane depth')
parser.add_argument('-invz', action='store_true', help='place MPI with inverse depth')
parser.add_argument('-scale', type=float, default=-1, help='scale the MPI size')
parser.add_argument('-llff_width', type=int, default=1008, help='if input dataset is LLFF it will resize the image to <llff_width>')
parser.add_argument('-deepview_width', type=int, default=800, help='if input dataset is deepview dataset, it will resize the image to <deepview_width>')
parser.add_argument('-train_ratio', type=float, default=0.875, help='ratio to split number of train/test (in case dataset doesn\'t specify how to split)')
parser.add_argument('-random_split', action='store_true', help='random split the train/test set. (in case dataset doesn\'t specify how to split)')
parser.add_argument('-num_workers', type=int, default=8, help='number of pytorch\'s dataloader worker')
parser.add_argument('-cv2resize', action='store_true', help='apply cv2.resize instead of skimage.transform.resize to match the score in our paper (see note in github readme for more detail) ')

#PMPI
parser.add_argument('-offset', type=int, default=200, help='the offset (padding) of the MPI.')
parser.add_argument('-layers', type=int, default=16, help='the number of plane that stores base color')
parser.add_argument('-sublayers', type=int, default=12, help='the number of plane that share the same texture. (please refer to coefficient sharing under section 3.4 in the paper)')
parser.add_argument('-size_patch', type=int, default=36, help='the size of each patch')
parser.add_argument('-gamma', type=float, default=0, help='the threshold of k0 to determine whether to prune the patch or not')

#predict
parser.add_argument('-no_eval', action='store_true', help='do not measurement the score (PSNR/SSIM/LPIPS) ')
parser.add_argument('-no_csv', action='store_true', help="do not write CSV on evaluation")
parser.add_argument('-no_video', action='store_true', help="do not write the video on prediction")
parser.add_argument('-no_webgl', action='store_true', help='do not predict webgl (realtime demo) related content.')
parser.add_argument('-predict', action='store_true', help='predict validation images')
parser.add_argument('-eval_path', type=str, default='runs/evaluation/', help='path to save validation image')
parser.add_argument('-web_path', type=str, default='runs/html/', help='path to output real time demo')
parser.add_argument('-web_width', type=int, default=16000, help='max texture size (pixel) of realtime demo. WebGL on Highend PC is support up to 16384px, while mobile phone support only 4096px')
parser.add_argument('-http', action='store_true', help='serve real-time demo on http server')
parser.add_argument('-render_viewing', action='store_true', help='genereate view-dependent-effect video')
parser.add_argument('-render_nearest', action='store_true', help='genereate nearest input video')
parser.add_argument('-render_depth', action='store_true', help='generate depth')

# render path
parser.add_argument('-nice_llff', action='store_true', help="generate video that its rendering path matches real-forward facing dataset")
parser.add_argument('-nice_shiny', action='store_true', help="generate video that its rendering path matches shiny dataset")


#training utility
parser.add_argument('-model_dir', type=str, default="scene", help='model (scene) directory which store in runs/<model_dir>/')
parser.add_argument('-pretrained', type=str, default="", help='location of checkpoint file, if not provide will use model_dir instead')
parser.add_argument('-restart', action='store_true', help='delete old weight and retrain')
parser.add_argument('-clean', action='store_true', help='delete old weight without start training process')

# dataset difference
parser.add_argument('-space', action='store_true', help='is space datasets?')


#miscellaneous
parser.add_argument('-all_gpu',action='store_true',help="In multiple GPU training, We don't train MLP (data parallel) on the first GPU. This make training slower but we can utilize more VRAM on other GPU.")

args = parser.parse_args()

def totalVariation(images):
  pixel_dif1 = images[:, :, 1:, :] - images[:, :, :-1, :]
  pixel_dif2 = images[:, :, :, 1:] - images[:, :, :, :-1]
  sum_axis = [1, 2, 3]

  tot_var = (
      pt.sum(pt.abs(pixel_dif1), dim=sum_axis) +
      pt.sum(pt.abs(pixel_dif2), dim=sum_axis))

  return tot_var / (images.shape[2]-1) / (images.shape[3]-1)

def cumprod_exclusive(x):
  # x: n_max, sel, 1
  cp = pt.cumprod(x, 0)
  cp = pt.roll(cp, 1, 0)
  cp[0] = 1.0
  return cp

def getWarp3d(warped, depths, sfm, interpolate = False):
  '''
  if not interpolate:
    depths = pt.repeat_interleave(pt.linspace(-1, 1, args.layers), args.sublayers).view(1, -1, 1, 1, 1).cuda()
  else:
    depths = pt.linspace(-1, 1, args.layers * args.sublayers).view(1, -1, 1, 1, 1).cuda()
  '''
  # warped: sel, n_max, 2
  # depths: sel, n_max, 1
  
  if sfm.invz:
      depths_normalized = (1- 2* (1 / depths - 1 / sfm.dmax) / (1 / sfm.dmin - 1 / sfm.dmax))
  else:
      depths_normalized = (2 * (depths - sfm.dmin) / (sfm.dmax - sfm.dmin) - 1)
  
  # 1, 1, sel, n_max, 3
  warp3d = pt.cat((warped, depths_normalized), -1)[None, None]

  # 1, n_max, sel, 1, 3
  warp3d = warp3d.permute([0, 3, 2, 1, 4])

  return warp3d

def normalized(v, dim):
  return v / (pt.norm(v, dim=dim, keepdim=True) + 1e-7)

def getMpi_c_grid3d(mpi_c_grid2d, depths_patch, sfm):
  '''
  Args:
    mpi_c_grid: n, mpi_h, mpi_w, 2
    depths_patch: n, mpi_h/p_h, mpi_w/p_w
  Returns:
    mpi_c_grid3d: 1, n, mpi_h, mpi_w, 3
  '''
  if sfm.invz:
    depths_normalized = 1 - 2 *(1/depths_patch - 1/sfm.dmax)/(1/sfm.dmin - 1/sfm.dmax)
  else:
    depths_normalized = 2 * (depths_patch - sfm.dmin) / (sfm.dmax - sfm.dmin) - 1

  #depth_pixel: n, mpi_h, mpi_w, 1

  depths_pixel = pt.repeat_interleave(pt.repeat_interleave(depths_normalized, args.size_patch, dim=1), args.size_patch, dim=2)[..., None]
  print("depths_pixel.size():")
  print(depths_pixel.size())

  #mpi_c_grid3d: 1, n, mpi_h, mpi_w, 3
  mpi_c_grid3d = pt.cat((mpi_c_grid2d, depths_pixel), -1)[None]
  
  return mpi_c_grid3d.cuda()
  
class Basis(nn.Module):
  def __init__(self, shape, out_view):
    super().__init__()
    #choosing illumination model
    self.order = args.basis_order

    # network for learn basis
    self.seq_basis = nn.DataParallel(
      ReluMLP(
        args.basis_mlp, #basis_mlp
        args.basis_hidden, #basis_hidden
        self.order * 4,
        args.lrelu_slope,
        out_node = args.basis_out, #basis_out
      )
    )
    print('Basis Network:',self.seq_basis)

    # positional encoding pre compute
    self.pos_freq_viewing = pt.Tensor([(2 ** i) for i in range(self.order)]).view(1, 1, -1).cuda()

  def forward(self, ray_dir, coeff = None):
    # ray_dir: sel, 2
    # coeff: sel, n_max, args.basis_out * 3
    # vi, xy = get_viewing_angle(PMPI_clamped, sfm, feature, ref_coords)
    sel = ray_dir.shape[0]
    ray_dir_homo = pt.cat((ray_dir, pt.ones(sel, 1).cuda()), 1)
    
    # sel, 3
    vi = normalized(ray_dir_homo, 1)

    # positional encoding for learn basis
    # sel, 2, pos_freq
    hinv_xy = vi[...,  :2, None] * self.pos_freq_viewing

    #sel, 1, 2*pos_freq
    big = pt.reshape(hinv_xy, [sel, 1, hinv_xy.shape[-2] * hinv_xy.shape[-1]])
    vi = pt.cat([pt.sin(0.5*np.pi*big), pt.cos(0.5*np.pi*big)], -1)

    # sel, 1, basis_out
    out2 = self.seq_basis(vi)
    out2 = pt.tanh(out2)

    # sel, 1, 1, basis_out
    vi = out2.view(sel, 1, 1, -1)

    # sel, n_max, 3, basis_out
    coeff = coeff.view(coeff.shape[0], coeff.shape[1], 3,  -1)
    coeff = pt.tanh(coeff)

    # n_max, 3, sel, 1
    illumination = pt.sum(coeff * vi,-1).permute([1, 2, 0])[..., None]

    return illumination

def get_viewing_angle(PMPI_clamped, sfm, feature, ref_coords):
  camera = sfm.ref_rT.t() @ feature["center"][0] + sfm.ref_t

  # (2n, rays, 2) -> (2n, 2, rays)
  coords = ref_coords.permute([0, 2, 1])
  # (n, 2, rays) -> (n, 3, rays)
  coords = pt.cat([coords, pt.ones_like(coords[:, :1])], 1)

  # coords: (n, 3, rays)
  # PMPI_clamped: (n, rays)
  # xyz: (n, 3, rays)
  xyz = coords * PMPI_clamped[:, None]

  ki = pt.tensor([[feature['fx'][0], 0, feature['px'][0]],
                  [0, feature['fy'][0], feature['py'][0]],
                  [0, 0, 1]], dtype=pt.float).inverse().cuda()

  xyz = ki @ xyz

  # camera: (3, 1) -> (1, 3, 1)
  # xyz: (n, 3, rays)
  # viewing_angle: (n, 3, rays)
  # viewing_angle = normalized(camera[None].cuda() - xyz, 1)
  inv_viewing_angle = normalized(xyz - camera[None].cuda(), 1)

  view = inv_viewing_angle.permute([0, 2, 1])
  xyz = xyz.permute([0, 2, 1])
  return view[:,:,None], xyz[:,:,None]

def get_depths_patch(sfm, float_drange):
  n = args.layers * args.sublayers
  if sfm.invz:
    planes_inverse = pt.Tensor(np.linspace(1/sfm.dmin, 1/sfm.dmax, n)).view(-1, 1, 1)
    PMPI_inverse = (planes_inverse - 1/sfm.dmax)/(1/sfm.dmin - 1/sfm.dmax) * (1/float_drange[0,...] - 1/float_drange[1,...]) + 1/float_drange[1,...]
    depths_patch = 1/PMPI_inverse
  else:
    planes = pt.Tensor(np.linspace(sfm.dmin, sfm.dmax, n)).view(-1, 1, 1)
    depths_patch = (planes - sfm.dmin) / (sfm.dmax - sfm.dmin) * (float_drange[1,...] - float_drange[0,...]) + float_drange[0,...]
  return depths_patch  

def get_patches(sfm, depths_patch, PMPI_center):
  # depths_patch: (n, mpi_h/p_h, mpi_w/p_w)
  # PMPI_center: [mpi_h/p_h, mpi_w/p_w, 2]

  #patches_z: (n, mpi_h/p_h, mpi_w/p_w, 1)
  n = depths_patch.shape[0]
  patches_z = depths_patch[..., None]

  #PMPI_center_homo: [mpi_h/p_h, mpi_w/p_w, 3, 1]
  PMPI_center_homo = pt.cat((PMPI_center, pt.ones(PMPI_center.shape[0], PMPI_center.shape[1], 1)), -1)[..., None]
  tt = sfm.ref_cam
  ref_k_inverse = pt.tensor( [[tt['fx'], 0, tt['px']],
                      [0, tt['fy'], tt['py']],
                      [0,        0,       1]]).inverse()

  # 1, mpi_h/p_h, mpi_w/p_w, 3, 1
  patches_xyz = (ref_k_inverse @ PMPI_center_homo)[None]

  # n, mpi_h/p_h, mpi_w/p_w, 3, 1
  patches_xyz = pt.repeat_interleave(patches_xyz, n, dim=0)

  patches_xyz[:, :, :, :2, 0] = pt.div(patches_xyz[:, :, :, :2, 0] * patches_z,  patches_xyz[:, :, :, 2:3, 0])
  patches_xyz[:, :, :, 2:3, 0] = patches_z


  point_x1 = (ref_k_inverse @ pt.tensor([args.size_patch/2, 0, 1], dtype=pt.float).view(3,1))
  point_x2 = (ref_k_inverse @ pt.tensor([0, 0, 1], dtype=pt.float).view(3, 1))
  point_x1 = point_x1 / point_x1[2]
  point_x2 = point_x2 / point_x2[2]

  #n, mpi_h/p_h, mpi_w/p_w, 1, 1
  patches_half = pt.tensor([point_x1[0] - point_x2[0]], dtype=pt.float).view(1, 1, 1, 1, 1)
  patches_half = pt.repeat_interleave(patches_half, patches_xyz.shape[0], dim=0)
  patches_half = pt.repeat_interleave(patches_half, patches_xyz.shape[1], dim=1)
  patches_half = pt.repeat_interleave(patches_half, patches_xyz.shape[2], dim=2)
  patches_half = patches_half * patches_xyz[:, :, :, 2:3, :]


  '''
  # (n, 1, 1, 1, 1)
  patches_half = ((patches_xyz[:, 0, 1, 0, 0] - patches_xyz[:, 0, 0, 0, 0]) / 2).view(-1, 1, 1, 1, 1)
  print("half_x:{}".format(patches_xyz[:, 0, 1, 0, 0] - patches_xyz[:, 0, 0, 0, 0]))
  print("half_x:{}".format(patches_xyz[:, 0, 30, 0, 0] - patches_xyz[:, 0, 29, 0, 0]))
  print("half_y:{}".format(patches_xyz[:, 1, 0, 1, 0] - patches_xyz[:, 0, 0, 1, 0]))
  print("half_y:{}".format(patches_xyz[:, 30, 0, 1, 0] - patches_xyz[:, 29, 0, 1, 0]))
  '''

  # (n, mpi_h/p_h, mpi_w/p_w, 4, 1)
  # num, 4
  patches_xyz = pt.cat((patches_xyz, patches_half), 3).view(-1, 4)

  # num, 1
  _, indice = pt.sort(patches_xyz[:, 2:3], 0)
  indice = pt.repeat_interleave(indice, 4, 1)
  patches_xyz_ordered = pt.gather(patches_xyz, 0, indice)

  return patches_xyz_ordered[None]

def get_ray_dir(sfm, selection, feature, output_shape):
  # camera: 1, 3
  # camera = (sfm.ref_rT.t() @ feature["center"][0] + sfm.ref_t).view(1, 3).cuda()

  selection = selection.cuda()
  # coords: (sel, 3)
  coords = pt.stack([selection % output_shape[1], selection // output_shape[1],
                    pt.ones_like(selection)], -1).float()
  
  Rs = pt.transpose(sfm.ref_rT, 0, 1)
  RtT = pt.transpose(feature['r'][0], 0, 1)
  
  fx = feature['fx'][0]
  fy = feature['fy'][0]
  px = feature['px'][0]
  py = feature['py'][0]

  kt_inverse = pt.tensor([[fx, 0, px],
                  [0, fy, py],
                  [0, 0, 1]], dtype=pt.float).inverse()

  H = Rs @ RtT @ kt_inverse

  # sel, 3
  dir_vec = coords @ pt.transpose(H, 0, 1).cuda()

  #dir_vec = coords_reference - camera

  # sel, 2
  ray_dir = dir_vec[:, :2] / dir_vec[:, 2:]
  return ray_dir

def points_to_warped(sfm, points, input_shape):
  #points: sel, n_max, 3
  tt = sfm.ref_cam
  ref_k = pt.tensor( [[tt['fx'], 0, tt['px']],
                      [0, tt['fy'], tt['py']],
                      [0,        0,       1]]).cuda()

  #sel, n_max, 2  image (x,y) coords
  warped = points @ pt.transpose(ref_k, 0, 1).cuda()
  warped = warped[:, :, :2] / warped[:, :, 2:3]

  # normalized to [-1, 1]
  # input_shape[1]: mpi_w  input_shape[0]: mpi_h
  scale = pt.tensor([input_shape[1] - 1, input_shape[0] - 1]).cuda()

  warped[..., 0:1] = ((warped[..., 0:1] + sfm.offset) / scale[0].view(1, 1, 1)) * 2 - 1
  warped[..., 1:2] = ((warped[..., 1:2] + sfm.offset) / scale[1].view(1, 1, 1)) * 2 - 1

  return warped

class Network(nn.Module):
  def __init__(self, PMPI_center, shape, sfm):
    super(Network, self).__init__()

    #shape: [args.layers, 4, mpi_h, mpi_w]
    #self.shape: [mpi_h, mpi_w]
    self.shape = [shape[2], shape[3]]
    total_cuda = pt.cuda.device_count()
    mlp_first_device = 1 if (not args.all_gpu) and total_cuda > 1 else 0

    print('mlp_first_device:{}'.format(mlp_first_device))
    print('total_cuda:{}'.format(total_cuda))
    mlp_devices = list(range(mlp_first_device, total_cuda))

    print('mlp_device:{}'.format(mlp_devices))
    # mpi_c (k0) as an explicit
    #mpi_c: [16, 3, mpi_h, mpi_w]
    mpi_c = pt.empty((16, 3, shape[2], shape[3]), device='cuda:0').uniform_(-1, 1)
    self.mpi_c = nn.Parameter(mpi_c)

    #only optimize sfm.dmin and sfm.dmax
    float_drange = pt.zeros(2, shape[2]//args.size_patch, shape[3]//args.size_patch)
    float_drange[0, :, :] = sfm.dmin
    float_drange[1, :, :] = sfm.dmax
    self.float_drange = nn.Parameter(float_drange)
    self.PMPI_center = PMPI_center

    # depths_patch: (n, mpi_h/p_h, mpi_w/p_w)
    # 占用显存，可以优化
    self.depths_patch = nn.Parameter(get_depths_patch(sfm, self.float_drange))

    # [1, num, 4] num = n * mpi_h/p_h * mpi_w/p_w   xp, yp, zp, l_half
    self.patches = nn.Parameter(get_patches(sfm, self.depths_patch, self.PMPI_center).cuda())

    print("PMPI_center:{}".format(self.PMPI_center.size()))
    print("PMPI_center_x:{}".format(self.PMPI_center[0, :, 0]))
    print("PMPI_center_y:{}".format(self.PMPI_center[:, 0, 1]))

    print("size of float_depth:", self.float_drange.size())

    # PMPI: [n, mpi_h/p_h, mpi_w/p_w]
    # PMPI_center: [mpi_h/p_h, mpi_w/p_w, 2]
    # PMPI_indice: [n, mpi_h/p_h, mpi_w/p_w]. the indice of each layer in each patch. range(0, n-1)
    #self.PMPI = PMPI

    #self.PMPI_indice = PMPI_indice.cuda()

    self.specular = Basis(shape, args.basis_out * 3).cuda()
    self.seq1 = nn.DataParallel(
      VanillaMLP(
        args.mlp,
        args.hidden,
        args.pos_level,
        args.depth_level,
        args.lrelu_slope,
        out_node = 1 + args.basis_out * 3,
        first_gpu = mlp_first_device
      ),
      device_ids = mlp_devices
    )

    self.seq1 = self.seq1.cuda("cuda:{}".format(mlp_first_device))
    self.pos_freq = pt.Tensor([0.5 * np.pi * (2 ** i) for i in range(args.pos_level)] * 2).view(1, 1, 2, -1).cuda()
    self.depth_freq = pt.Tensor([0.5 * np.pi * (2 ** i) for i in range(args.depth_level)]).view(1, 1, -1).cuda()
    
    #self.z_coords = pt.linspace(-1, 1, args.layers * args.sublayers).view(-1, 1, 1, 1).cuda()
    if args.render_depth:
      self.rainbow_mpi = np.zeros((shape[0], 3, shape[2], shape[3]), dtype=np.float32)
      for i,s in enumerate(np.linspace(1, 0, shape[0])):
        color = Rainbow(s)
        for c in range(3):
          self.rainbow_mpi[i,c] = color[c]
      self.rainbow_mpi = pt.from_numpy(self.rainbow_mpi).to('cuda:0')
    else:
      self.rainbow_mpi = None

    if sfm.dmin < 0 or sfm.dmax < 0:
      raise ValueError("invalid dmin dmax")

    # self.planes = getPlanes(sfm, args.layers * args.sublayers)
    print('Mpi Size: {}'.format(self.mpi_c.shape))
    print('All combined layers: {}'.format(args.layers * args.sublayers))

    print('Layer of MLP: {}'.format(args.mlp + 2))
    print('Hidden Channel of MLP: {}'.format(args.hidden))
    print('Main Network',self.seq1)


  def forward(self, sfm, feature, output_shape, selection):
    ''' Rendering
    Args:
      sfm: reference camera parameter
      feature: target camera parameter
      output_shape: [h, w]. Desired rendered image
      selection: [ray]. pixel to train
    Returns:
      output: [1, 3, rays, 1] rendered image
    '''
    # depths_patch: (n, mpi_h/p_h, mpi_w/p_w)
    # warp_temp: (n, mpi_h/p_h, mpi_w/p_w, sel, 1, 2)
    # ref_coords: [n, mpi_h/p_h, mpi_w/p_w, sel, 2]

    
    n_max = args.n_max
    # coords of traget viewpoint in reference coordinates syetem
    # (3, 1)
    ray_star = (sfm.ref_rT.t() @ feature["center"][0] + sfm.ref_t).cuda()

    # [sel, 2] attention: sel != 8000  !!!!!!
    ray_dir = get_ray_dir(sfm, selection, feature, output_shape)

    # sel, n_max, 3   coords in reference coordinate system
    points = aabb_ray_intersect(n_max, self.patches, ray_star[None], ray_dir, args.ray)


    # warped: sel, n_max, 2,    [mpi_h, mpi_w] normalized to [-1, 1], in [mpi_h, mpi_w]
    # depths: sel, n_max, 1
    warped = points_to_warped(sfm, points, self.shape)

    # 占用显存，可以优化
    depths = points[:, :, 2:3]

    sel = warped.shape[0]

    # vxy: (sel, n_max, pos_level*2)
    vxy = (warped[:, :, :, None] * self.pos_freq).view(sel, n_max, -1)

    # vz: (sel, n_max, depth_level)
    if sfm.invz:
      vz = (1- 2* (1 / depths - 1 / sfm.dmax) / (1 / sfm.dmin - 1 / sfm.dmax)) * self.depth_freq
    else:
      vz = (2 * (depths - sfm.dmin) / (sfm.dmax - sfm.dmin) - 1) * self.depth_freq

    # vxyz: (sel, n_max, 2 * pos_level + depth_level)
    vxyz = pt.cat([vxy, vz], -1)
    bigcoords = pt.cat([pt.sin(vxyz), pt.cos(vxyz)], -1)
    # (sel, n_max, out_node)
    out = self.seq1(bigcoords).cuda()

    node = 0
    #占用显存，可以优化
    # self.mpi_a = out[..., node:node + 1]
    
    # n_max, sel, 1
    mpi_a_sig  = pt.sigmoid(out[..., node:node + 1] - args.sigmoid_offset).permute([1, 0, 2])
    node += 1

    if args.render_depth:
      # generate Rainbow MPI instead of real mpi to visualize the depth
      # self.rainbow_mpi: n, 3, h, w   warp: (n, sel, 1, 2)
      # Need: N, C, Din, Hin, Win;  N, Dout, Hout, Wout, 3
      rainbow_3d = self.rainbow_mpi.permute([1, 0, 2, 3])[None]
      warp3d = getWarp3d(warped, depths, sfm)
      #samples: N, C, Dout, Hout, Wout
      samples = F.grid_sample(rainbow_3d, warp3d, align_corners=True)
      # (layers, out_node, rays, 1)
      rgb = samples[0].permute([1, 0, 2, 3])
    else:
      mpi_sig = pt.sigmoid(self.mpi_c)
      # mpi_sig: n, 3, h, w   warp: (n, sel, 1, 2)
      # Need: N, C, Din, Hin, Win;    N, Dout, Hout, Wout, 3
      # warp3d: [1, n, sel, 1, 2]

      # 1, 3, n, h, w
      mpi_sig3d = mpi_sig.permute([1, 0, 2, 3])[None]

      # warp3d: 1, n_max, sel, 1, 3
      warp3d = getWarp3d(warped, depths, sfm)

      #samples: 1, 3, n_max, sel, 1
      samples = F.grid_sample(mpi_sig3d, warp3d, align_corners=True)
      # n_max, 3, sel, 1
      rgb = samples[0].permute([1, 0, 2, 3])
      
      # 基函数需要进一步适配，需要立即适配，否则网络训练飞了
      # 先不考虑参数MPI参数共享
      # sel, n_max, args.basis_out * 3
      cof = out[..., node:]

      # n_max, 3, sel, 1
      self.illumination = self.specular(ray_dir, coeff = cof)
      
      # n_max, 3, sel, 1
      rgb = pt.clamp(rgb + self.illumination, 0.0, 1.0)

    # mpi_a_sig: n_max, 1, sel, 1
    # n_max, 1, sel, 1
    mpi_a_sig = mpi_a_sig[:, None]
    weight = cumprod_exclusive(1 - mpi_a_sig)

    # 1, 3, sel, 1
    output = pt.sum(weight * rgb * mpi_a_sig, dim=0, keepdim=True)

    return output

  def updateDrange(self, sfm):
    '''update self.float_drange iteratly
    '''
    #slef.mpi_c: (layers, 3, mpi_h, mpi_w)
    #self.float_drange: (2, mpi_h/p_h, mpi_w/p_w)

    # self.depths_patch: (n, mpi_h/p_h, mpi_w/p_w)
    #mpi_c_grid2d: 1, n, mpi_h, mpi_w, 2
    #mpi_c_grid3d: 1, n, mpi_h, mpi_w, 3

    n = args.layers * args.sublayers

    #mpi_h, mpi_w, 1  normalized
    mpi_c_grid2d_x = pt.repeat_interleave(pt.linspace(-1, 1, self.shape[1])[None], self.shape[0], dim=0)[..., None]

    mpi_c_grid2d_y = pt.repeat_interleave(pt.linspace(-1, 1, self.shape[0])[:, None], self.shape[1], dim=1)[..., None]

    #n, mpi_h, mpi_w, 2
    mpi_c_grid2d = pt.repeat_interleave(pt.cat((mpi_c_grid2d_x, mpi_c_grid2d_y), dim=2)[None], n, dim=0)    
    
    print("mpi_c_grid2d.size():")
    print(mpi_c_grid2d.size())
    
    print("depths_patch():")
    print(self.depths_patch.size())
    #mpi_c_grid3d: 1, n, mpi_h, mpi_w, 3      in cuda
    mpi_c_grid3d = getMpi_c_grid3d(mpi_c_grid2d, self.depths_patch, sfm)

    print("mpi_c_grid3d.size():")
    print(mpi_c_grid3d.size())

    # mpi_sig3d: 1, 3, layers, mpi_h, mpi_w      in cuda
    mpi_sig = pt.sigmoid(self.mpi_c)
    mpi_sig3d = mpi_sig.permute([1, 0, 2, 3])[None]
    
    #mpi_c_selected: 3, n, mpi_h, mpi_w
    mpi_c_selected = F.grid_sample(mpi_sig3d, mpi_c_grid3d, align_corners=True).view(3, n, self.shape[0], self.shape[1])

    print("mpi_c_selected.max():{}".format(mpi_c_selected.max()))
    print("mpi_c_selected.min():{}".format(mpi_c_selected.min()))
    #pruuner
    #3, n, mpi_h, mpi_w
    mpi_c_normalized = pt.pow(mpi_c_selected - 0.5, 2)
    #1, n, mpi_h, mpi_w
    mpi_c_se = (mpi_c_normalized[0, ...] + mpi_c_normalized[1, ...] + mpi_c_normalized[2, ...])[None]
  
    #1, n, mpi_h/p_h, mpi_w/p_w
    m = nn.MaxPool2d(args.size_patch, stride=args.size_patch)
    mpi_c_patch = m(mpi_c_se)

    print(mpi_c_se.size())
    print(mpi_c_patch.size())

    print("mpi_c_patch.min():{}".format(mpi_c_patch.min()))

    mask = pt.where(mpi_c_patch > args.gamma, mpi_c_patch, pt.zeros_like(mpi_c_patch))

    '''
    if epoch==2500:
      for i in range(mpi_c_patch.shape[2]):
        for j in range(mpi_c_patch.shape[3]):
          occupancy_index = pt.nonzero(mask[0, :, i, j])
          if occupancy_index.shape[0]>0:
            #print("update happened!!!")
            #update dmin
            index_min = occupancy_index[0,0]
            self.float_drange.data[0, i, j] = self.depths_patch[max(index_min - 1, 0), i, j]
            #print("index_min:{}".format(index_min))
            #print("updated dmin:{}".format(depths_patch[index_min, i, j]))

            #update dmax
            index_max = occupancy_index[-1, 0]
            self.float_drange.data[1, i, j] = self.depths_patch[min(index_max+1, n-1), i, j]
    '''
    for i in range(mpi_c_patch.shape[2]):
      for j in range(mpi_c_patch.shape[3]):
        occupancy_index = pt.nonzero(mask[0, :, i, j])
        if occupancy_index.shape[0]>0:
          #print("update happened!!!")
          #update dmin
          index_min = occupancy_index[0,0]
          self.float_drange.data[0, i, j] = self.depths_patch[max(index_min - 1, 0), i, j]
          #print("index_min:{}".format(index_min))
          #print("updated dmin:{}".format(depths_patch[index_min, i, j]))

          #update dmax
          index_max = occupancy_index[-1, 0]
          self.float_drange.data[1, i, j] = self.depths_patch[min(index_max+1, args.layers*args.sublayers-1), i, j]

    #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    #self.float_drange.data[0, :, :] = 1.44
    # update depths_patch and patches from self.float_drange
    # print("dmin.min():{}".format(self.float_drange[0, ...].max()))
    # depths_patch: (n, mpi_h/p_h, mpi_w/p_w)
    self.depths_patch.data = get_depths_patch(sfm, self.float_drange)

    # [1, num, 4] num = n * mpi_h/p_h * mpi_w/p_w   xp, yp, zp, l_half
    self.patches.data = get_patches(sfm, self.depths_patch, self.PMPI_center).cuda()

def get_mpi_a(model, sfm):
  '''get mpi_a in each layer
    Args:
      Args:
      model: Neural net model
      sfm: reference camera parameter
      m: target camera parameter
      dataloader
    Returns:
      None, save mpi_a images
  '''
  # size of mpi layers
  sh = int(sfm.ref_cam['height'] + sfm.offset * 2)
  sw = int(sfm.ref_cam['width'] + sfm.offset * 2)

  # image in space datasets is fixed to (800, 480)
  if args.space:
    sh = (480 + sfm.offset * 2)
    sw = (800 + sfm.offset * 2)
  

  #normalized to [-1, 1]
  y, x = pt.meshgrid([
    (pt.arange(0, sh, dtype=pt.float)) / (sh-1) * 2 - 1,
    (pt.arange(0, sw, dtype=pt.float)) / (sw-1) * 2 - 1])

  coords = pt.cat([x[:,:,None].cuda(), y[:,:,None].cuda()], -1)

  model.eval()
  with pt.no_grad():
    # depths: [n, sh, sw]
    n = args.layers * args.sublayers
    depths = pt.repeat_interleave(model.depths_patch, args.size_patch, 1)
    depths = pt.repeat_interleave(depths, args.size_patch, 2).cuda()

    #normalized to [-1, 1]
    if sfm.invz:
        depths_normalized = (1- 2* (1 / depths - 1 / sfm.dmax) / (1 / sfm.dmin - 1 / sfm.dmax))
    else:
        depths_normalized = (2 * (depths - sfm.dmin) / (sfm.dmax - sfm.dmin) - 1)

    mpi_a_path = os.path.join('runs/layers/', args.model_dir)
    os.makedirs(mpi_a_path,exist_ok=True)

    print(model.depths_patch[0,:, :,].max())
    print(depths[0,:, :,].max())

    z_coords = pt.linspace(-1, 1, args.layers * args.sublayers).view(-1, 1, 1, 1).cuda()

    for i in range(n):
      #coords [sh, sw, 2] --> [1, sh, sw, 2, 1]
      #vxy [1, sh, sw, 2, pos_lev] -->  [1, sh, sw, 2*pos_lev]
      #pos_freq: 1, 1, 2, pos_lev
      vxy = coords.view(sh, sw, 2, 1) * model.pos_freq
      vxy = vxy.view(1, sh, sw, -1)

      #vz: (1, sh, sw, depth_lev)
      vz = (depths_normalized[i, :, :][..., None] * model.depth_freq)[None]

      #vxyz [1, sh, sw, 2*pos_lev + depth_lev]
      vxyz = pt.cat([vxy, vz], -1)
      bigcoords = pt.cat([pt.sin(vxyz), pt.cos(vxyz)], -1)
      
      out = model.seq1(bigcoords)
      node = 0

      #[1, sh, sw, 1]
      mpi_a = pt.sigmoid(out[..., node:node + 1].cpu().detach() - args.sigmoid_offset).numpy()
      
      image_name = "alphaImage_{:04d}.png".format(i)
      filepath = os.path.join(mpi_a_path, image_name)
      print('alphaImage_{:04d}.png'.format(i))
      io.imsave(filepath, (mpi_a[0, :, :, 0] * 255).astype(np.uint8))


      #save alpha images in k0
      vz_k0 = (pt.ones(sh, sw, 1).cuda() * z_coords[i, :, :] * model.depth_freq)[None]
      vxyz_k0 = pt.cat([vxy, vz_k0], -1)
      bigcoords_k0 = pt.cat([pt.sin(vxyz_k0), pt.cos(vxyz_k0)], -1)
      
      out_k0 = model.seq1(bigcoords_k0)
      node = 0

      #[1, sh, sw, 1]
      mpi_a_k0 = pt.sigmoid(out_k0[..., node:node + 1].cpu().detach() - args.sigmoid_offset).numpy()
      
      image_name = "alphaImage_k0_{:04d}.png".format(i)
      filepath = os.path.join(mpi_a_path, image_name)
      print('alphaImage_k0_{:04d}.png'.format(i))
      io.imsave(filepath, (mpi_a_k0[0, :, :, 0] * 255).astype(np.uint8))

      #save k0
      #mpi_c: [args.layer, 3, mpi_h, mpi_w]
      #mpi_sig: [3, mpi_h, mpi_w]
      if i<16:
        mpi_sig = pt.sigmoid(model.mpi_c[i, ...].permute([1, 2, 0])).cpu().detach().numpy()
        k0_name = "k0_{:04d}.png".format(i)
        k0_path = os.path.join(mpi_a_path, k0_name)
        print('k0_{:04d}.png'.format(i))
        io.imsave(k0_path, (mpi_sig * 255).astype(np.uint8))

      #coords: [sh, sw, 2]
      #depth_normalized: [n, sh, sw]
      #grid_coords: [1, 1, sh, sw, 3]
      grid_coords = pt.cat((coords, depths_normalized[i][..., None]), 2)[None, None]
      
      # mpi_sig3d: 1, 3, layers, mpi_h, mpi_w      in cuda
      mpi_sig_all = pt.sigmoid(model.mpi_c).permute([1, 0, 2, 3])[None]

      #sh, sw, 3
      output_k0 = F.grid_sample(mpi_sig_all, grid_coords, align_corners=True).view(3, sh, sw).permute([1, 2, 0]).cpu().detach().numpy()
      output_k0_name = "out_k0_{:04d}.png".format(i)
      output_k0_path = os.path.join(mpi_a_path, output_k0_name)
      io.imsave(output_k0_path, (output_k0 * 255).astype(np.uint8))

def generateAlpha(model, dataset, dataloader, writer, runpath, suffix="", dataloader_train = None):
  ''' Prediction
    Args.
      model.   --> trained model
      dataset. --> valiade dataset
      writer.  --> tensorboard
  '''
  suffix_str = "/%06d" % suffix if isinstance(suffix, int) else "/"+str(suffix)
  get_mpi_a(model, dataset.sfm)
  

  if not args.no_eval and len(dataloader) > 0:
    out = evaluation(model,
                     dataset,
                     dataloader,
                     2000 * args.ray,
                     runpath + args.model_dir + suffix_str,
                     webpath=args.eval_path,
                     write_csv = not args.no_csv)
    if writer is not None and isinstance(suffix, int):
      for metrics, score in out.items():
        writer.add_scalar('METRICS/{}'.format(metrics), score, suffix)


def setLearningRate(optimizer, epoch):
  if epoch > args.before_decay:
    ds = int((epoch - args.before_decay) / args.decay_epoch)
    lr = args.lr * (args.decay_rate ** ds)
  else:
    lr = args.lr

  optimizer.param_groups[0]['lr'] = lr
  if args.lrc > 0:
    optimizer.param_groups[1]['lr'] = lr * args.lrc

  if epoch > 4000:
    ds = int(3500 / args.decay_epoch)
    lr = args.lr * (args.decay_rate ** ds)

    optimizer.param_groups[0]['lr'] = lr
    if args.lrc > 0:
      optimizer.param_groups[1]['lr'] = lr * args.lrc


def get_PMPI(shape, sfm):
  ''' create PMPI
    Args:
      depth_start: [mpi_h/p_h, mpi_w/p_w]. minimun depth of each patch from colmap
      sfm:  reference camera parameter
    Returns:
      PMPI: [layers, mpi_h/p_h, mpi_w/p_w]. depths of each patch
      PMPI_center: [mpi_h/p_h, mpi_w/p_w, 2]. center (x,y) of each patch (without padding)
  '''
  n = args.layers * args.sublayers

  #PMPI_center: [mpi_h/p_h, mpi_w/p_w, 2]
  PMPI_center = pt.empty(shape[0], shape[1], 2)
  PMPI_x_start = args.size_patch/2 - sfm.offset
  PMPI_y_start = args.size_patch/2 - sfm.offset
  PMPI_x_end = args.size_patch* shape[1] - args.size_patch/2 - sfm.offset
  PMPI_y_end = args.size_patch* shape[0] - args.size_patch/2 - sfm.offset
  PMPI_center[..., 0] = pt.repeat_interleave(pt.linspace(PMPI_x_start, PMPI_x_end, shape[1])[None], shape[0], dim=0)
  PMPI_center[..., 1] = pt.repeat_interleave(pt.linspace(PMPI_y_start, PMPI_y_end, shape[0])[...,None], shape[1], dim=1)

  return PMPI_center

def train():
  pt.manual_seed(1)
  np.random.seed(1)

  if args.restart or args.clean:
    os.system("rm -rf " + "runs/" + args.model_dir)
  if args.clean:
    exit()

  dpath = args.scene

  dataset = loadDataset(dpath)
  sampler_train, sampler_val, dataloader_train, dataloader_val = prepareDataloaders(
    dataset,
    dpath,
    random_split = args.random_split,
    train_ratio = args.train_ratio,
    num_workers = args.num_workers
  )

  mpi_h = int(dataset.sfm.ref_cam['height'] + dataset.sfm.offset * 2)
  mpi_w = int(dataset.sfm.ref_cam['width'] + dataset.sfm.offset * 2)

  # image in space datasets is fixed to (800, 480)
  if args.space:
    mpi_h = (480 + dataset.sfm.offset * 2)
    mpi_w = (800 + dataset.sfm.offset * 2)
  
  # PMPI_center: [mpi_h/p_h, mpi_w/p_w, 2]
  PMPI_center= get_PMPI((mpi_h// args.size_patch, mpi_w // args.size_patch), dataset.sfm)
  print("size of PMPI_center:{}".format(PMPI_center.size()))

  model = Network(PMPI_center, 
                 (args.layers,
                 4,
                 mpi_h,
                 mpi_w,
                 ), dataset.sfm)

  if args.lrc > 0:
    #mlp use lower lr, while mpi_c, light dir intensity get higher lr
    my_list = [name for name, params in model.named_parameters() if 'seq1' in name]
    mpi_c_params = [name for name, params in model.named_parameters() if 'mpi_c' in name]
    mlp_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in my_list, model.named_parameters()))))
    k0_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in mpi_c_params, model.named_parameters()))))
    optimizer = pt.optim.Adam([
      {'params': mlp_params, 'lr': 0},
      {'params': k0_params, 'lr': 0}])
  else:
    optimizer = pt.optim.Adam(model.parameters(), lr=0)

  start_epoch = 0
  runpath = "runs/"
  ckpt = runpath + args.model_dir + "/ckpt.pt"
  if os.path.exists(ckpt):
    start_epoch = loadFromCheckpoint(ckpt, model, optimizer)
  elif args.pretrained != "":
    start_epoch = loadFromCheckpoint(runpath + args.pretrained + "/ckpt.pt", model, optimizer)

  step = start_epoch * len(sampler_train)

  if args.epochs < 0 and args.steps < 0:
    raise Exception("Need to specify epochs or steps")

  if args.epochs < 0:
    args.epochs = int(np.ceil(args.steps / len(sampler_train)))

  if args.predict:
    generateAlpha(model, dataset, dataloader_val, None, runpath, dataloader_train = dataloader_train)
    if not args.no_video:
      if args.render_nearest:
        vid_path = 'video_nearest'
        render_type = 'nearest'
      elif args.render_viewing:
        vid_path = 'viewing_output'
        render_type = 'viewing'
      elif args.render_depth:
        vid_path = 'video_depth'
        render_type = 'depth'
      else:
        vid_path = 'video_output'
        render_type = 'default'
      pt.cuda.empty_cache()
      render_video(model, dataset, 2000 * args.ray, os.path.join(runpath, vid_path, args.model_dir),
                  render_type = render_type, dataloader = dataloader_train)
    if args.http:
      serve_files(args.model_dir, args.web_path)
    exit()


  backupConfigAndCode(runpath)
  ts = TrainingStatus(num_steps=args.epochs * len(sampler_train))
  writer = SummaryWriter(runpath + args.model_dir)
  writer.add_text('command',' '.join(sys.argv), 0)
  ts.tic()

   # shift by 1 epoch to save last epoch to tensorboard
  for epoch in range(start_epoch, args.epochs+1):
    
    #update dmin and dmax for every N epochs
    #N=500
    #update depth range at epoch 1500, 2000, 2500, 3000

    #model.updateDrange(dataset.sfm)
    #print("dmin.max():{}".format(model.float_drange[0, ...].max()))

    #for test
    
    #model.updateDrange(dataset.sfm, epoch)

    if epoch >= 1000 and epoch <= 2500 and epoch % 500 == 0:
      print('update drange')
      model.updateDrange(dataset.sfm)
    
    epoch_loss_total = 0
    epoch_mse = 0

    model.train()
      
    for i, feature in enumerate(dataloader_train):
      #print("step: {}".format(i))
      setLearningRate(optimizer, epoch)
      optimizer.zero_grad()

      output_shape = feature['image'].shape[-2:]
      #print("output_shape:{}".format(output_shape))

      #sample L-shaped rays
      sel = Lsel(output_shape, 2000 * args.ray)


      gt = feature['image']
      #print("gt.shape:{}".format(gt.shape))
      gt = gt.view(gt.shape[0], gt.shape[1], gt.shape[2] * gt.shape[3])
      gt = gt[:, :, sel, None].cuda()

      #T1 = time.time()
      #1, 3, sel, 1
      output = model(dataset.sfm, feature, output_shape, sel)
      #T2 = time.time()
      #print("runtime:{}ms".format((T2-T1)*1000))

      mse = pt.mean((output - gt) ** 2)

      loss_total = mse

      #tvc regularizer
      tvc = args.tvc * pt.mean(totalVariation(pt.sigmoid(model.mpi_c[:, :3])))
      loss_total = loss_total + tvc

      # grad loss
      ox = output[:, :, 1::3,  :] - output[:, :, 0::3, :]
      oy = output[:, :, 2::3,  :] - output[:, :, 0::3, :]
      gx = gt[:, :, 1::3,  :] - gt[:, :, 0::3, :]
      gy = gt[:, :, 2::3, :] - gt[:, :, 0::3, :]
      loss_total = loss_total + args.gradloss * (pt.mean(pt.abs(ox - gx)) + pt.mean(pt.abs(oy - gy)))

      epoch_loss_total += loss_total
      epoch_mse += mse

      loss_total.backward()
      optimizer.step()

      step += 1
      toc_msg = ts.toc(step, loss_total.item())
      if step % args.tb_toc == 0:  print(toc_msg)
      ts.tic()

    writer.add_scalar('loss/total', epoch_loss_total/len(sampler_train), epoch)
    writer.add_scalar('loss/mse', epoch_mse/len(sampler_train), epoch)

    var = pt.mean(pt.std(model.illumination, 2)** 2)
    mean = pt.mean(model.illumination)
    writer.add_scalar('loss/illumination_mean', mean, epoch)
    writer.add_scalar('loss/illumination_var', var, epoch)

    if (epoch+1) % args.checkpoint == 0 or epoch == args.epochs-1:
      if np.isnan(loss_total.item()):
        exit()
      checkpoint(ckpt, model, optimizer, epoch+1)


  print('Finished Training')

  generateAlpha(model, dataset, dataloader_val, None, runpath, dataloader_train = dataloader_train)

  depth_start = model.float_drange[0,:,:].view(-1).detach().numpy()
  depth_final = model.float_drange[1,:,:].view(-1).detach().numpy()
  depth_path = os.path.join('runs/layers/', args.model_dir)
  np.savetxt(os.path.join(depth_path, 'depth_origin.txt'), depth_start)
  np.savetxt(os.path.join(depth_path, 'depth_final.txt'), depth_final)

  # render the reference image
  print('rendering the reference image')
  feature = dataset.__getitem__(0)
  feature['r'] = dataset.sfm.ref_rT.t()
  feature['t'] = dataset.sfm.ref_t
  #feature['R'] = feature['r'][0].t()[None]
  #feature['center'] = (-feature['R'][0] @ feature['t'][0])[None]


  #feature['R'] = feature['r'][0].t()[None]
  #feature['center'] = (-feature['R'][0] @ feature['t'][0])[None]
  for key in feature.keys():
    if key != 'path':
      feature[key] = pt.from_numpy(np.expand_dims(feature[key], axis=0))

  reference_image = patch_render(model, dataset.sfm, feature, 2000 * args.ray)
  reference_image = reference_image.permute(0, 2, 3, 1).cpu().detach().numpy()[0]
  io.imsave(os.path.join(dpath, 'reference_image.png'), (reference_image * 255).astype(np.uint8))
  print('rendered the reference image')
  

  if not args.no_video:
    render_video(model, dataset, 2000 * args.ray, os.path.join(runpath, 'video_output', args.model_dir))
  if args.http:
    serve_files(args.model_dir, args.web_path)
  

def checkpoint(file, model, optimizer, epoch):
  print("Checkpointing Model @ Epoch %d ..." % epoch)
  pt.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    }, file)

def loadFromCheckpoint(file, model, optimizer):
  checkpoint = pt.load(file)
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  start_epoch = checkpoint['epoch']
  print("Loading %s Model @ Epoch %d" % (args.pretrained, start_epoch))
  return start_epoch

def backupConfigAndCode(runpath):
  if args.predict or args.clean:
    return
  model_path = os.path.join(runpath, args.model_dir)
  os.makedirs(model_path, exist_ok = True)
  now = datetime.now()
  t = now.strftime("_%Y_%m_%d_%H:%M:%S")
  with open(model_path + "/args.json", 'w') as out:
    json.dump(vars(args), out, indent=2, sort_keys=True)
  os.system("cp " + os.path.abspath(__file__) + " " + model_path + "/")
  os.system("cp " + os.path.abspath(__file__) + " " + model_path + "/" + __file__.replace(".py", t + ".py"))
  os.system("cp " + model_path + "/args.json " + model_path + "/args" + t + ".json")


def loadDataset(dpath):
  # if dataset directory has only image, create LLFF poses
  colmapGenPoses(dpath)


  # for debug
  #colmapGenPoses_modified(dpath)

  if args.scale == -1:
    args.scale = getDatasetScale(dpath, args.deepview_width, args.llff_width)

  if is_deepview(dpath) and args.ref_img == '':
    with open(dpath + "/ref_image.txt", "r") as fi:
      args.ref_img = str(fi.readline().strip())
  render_style = 'llff' if args.nice_llff else 'shiny' if args.nice_shiny else ''
  return OrbiterDataset(dpath, ref_img=args.ref_img, scale=args.scale,
                           dmin=args.dmin,
                           dmax=args.dmax,
                           invz=args.invz,
                           render_style=render_style,
                           offset=args.offset,
                           cv2resize=args.cv2resize)


if __name__ == "__main__":
  sys.excepthook = colored_hook(os.path.dirname(os.path.realpath(__file__)))
  train()
