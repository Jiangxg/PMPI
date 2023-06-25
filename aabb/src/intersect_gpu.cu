#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "cuda_utils.h"
#include "cutil_math.h"  // required for float3 vector math

__device__ float3 RayAABBIntersection(
  const float3 &ori,
  const float2 &dir,
  const float3 &center,
  float half_voxel) {

  float intersect_x, left_x, right_x, intersect_y, down_y, up_y;

  intersect_x = __fadd_rn(ori.x, __fmul_rn(__fsub_rn(center.z, ori.z), dir.x));
  left_x = __fsub_rn(center.x, half_voxel);
  if (intersect_x < left_x){
    return make_float3(-1.0f, -1.0f, -1.0f);
  }

  right_x = __fadd_rn(center.x, half_voxel);
  if (intersect_x >= right_x){
    return make_float3(-1.0f, -1.0f, -1.0f);
  }

  intersect_y = __fadd_rn(ori.y, __fmul_rn(__fsub_rn(center.z, ori.z), dir.y));
  down_y = __fsub_rn(center.y, half_voxel);
  if (intersect_y < down_y){
    return make_float3(-1.0f, -1.0f, -1.0f);
  }

  up_y = __fadd_rn(center.y, half_voxel);
  if (intersect_y >= up_y){
    return make_float3(-1.0f, -1.0f, -1.0f);
  }

  return make_float3(intersect_x, intersect_y, center.z);
}


__global__ void aabb_intersect_point_kernel(
            int b, int n, int m,
            int n_max,
            const float *__restrict__ ray_start,
            const float *__restrict__ ray_dir,
            const float *__restrict__ patches,
            float *__restrict__ points) {

  // __restrict__ 关键词对于加速很重要，减少了访问存储器的次数
  // points-->[2000, ray, n_max, 3]
  // patches-->[2000, num, 4]
  // ray_start-->[2000, 3, 1]
  // ray_dir-->[2000, ray, 2]

  // b=2000, n=num, m=ray (4)

  // 线程块的索引, 应该是 0,1,2,...
  int batch_index = blockIdx.x;

  // Grid是一维的
  // 将tensor指针移动到本线程块处理的数据内存地址上
  // n: patches.size(1)

  patches += batch_index * n * 4;

  // m: ray_start.size(1)
  ray_start += batch_index * 3;
  ray_dir += batch_index * m * 2;

  points += batch_index * m * n_max * 3;
    
  // 线程块block是一维的，本函数处理了一个线程块中的一列线程

  int j = threadIdx.x;

  //遍历所有voxel
  for (int k = 0, cnt = 0; k < n && cnt < n_max; ++k) {
    float3 intersects = RayAABBIntersection(
      make_float3(ray_start[0], ray_start[1], ray_start[2]),
      make_float2(ray_dir[j * 2 + 0], ray_dir[j * 2 + 1]),
      make_float3(patches[k * 4 + 0], patches[k * 4 + 1], patches[k * 4 + 2]),
      patches[k * 4 + 3]);

    if (intersects.z > 0.0f){
      points[(j * n_max + cnt) * 3 + 0] = intersects.x;
      points[(j * n_max + cnt) * 3 + 1] = intersects.y;
      points[(j * n_max + cnt) * 3 + 2] = intersects.z;
      ++cnt;
    }
  }
}




void aabb_intersect_point_kernel_wrapper(
  int b, int n, int m, int n_max,
  const float *ray_start, const float *ray_dir, const float *patches,
  float *points) {
  
  //cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // 这一句分b个线程块，每一个线程块分为了m个线程(thread)
  //aabb_intersect_point_kernel<<<b, opt_n_threads(m), 0, stream>>>(
      //b, n, m, n_max, ray_start, ray_dir, patches, points);
  
  aabb_intersect_point_kernel<<<b, opt_n_threads(m)>>>(
      b, n, m, n_max, ray_start, ray_dir, patches, points);


  CUDA_CHECK_ERRORS();
}

