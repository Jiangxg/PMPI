#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>


#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_IS_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be a float tensor")


void aabb_intersect_point_kernel_wrapper(
  int b, int n, int m, int n_max,
  const float *ray_start, const float *ray_dir, const float *patches,
  float *points);


// tuple ---> vector
at::Tensor aabb_intersect(at::Tensor ray_start, at::Tensor ray_dir, at::Tensor patches, 
               const int n_max){
  CHECK_CONTIGUOUS(ray_start);
  CHECK_CONTIGUOUS(ray_dir);
  CHECK_CONTIGUOUS(patches);
  CHECK_IS_FLOAT(ray_start);
  CHECK_IS_FLOAT(ray_dir);
  CHECK_IS_FLOAT(patches);
  CHECK_CUDA(ray_start);
  CHECK_CUDA(ray_dir);
  CHECK_CUDA(patches);

  // points-->[2000, ray, n_max, 3]
  // patches-->[2000, num, 4]
  // ray_start-->[2000, 3, 1]
  // ray_dir-->[2000, ray, 2]
  at::Tensor points =
      torch::ones({ray_start.size(0), ray_dir.size(1), n_max, 3},
                    at::device(ray_start.device()).dtype(at::ScalarType::Float));

  aabb_intersect_point_kernel_wrapper(patches.size(0), patches.size(1), ray_dir.size(1),
                                      n_max,
                                      ray_start.data_ptr <float>(), ray_dir.data_ptr <float>(), patches.data_ptr <float>(),
                                      points.data_ptr <float>());
  return points;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("aabb_intersect", &aabb_intersect);
}