#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>

#include "pytorch_cpp_helper.hpp"
#include "knn_cpu_kernel.hpp"


void knn_forward_cpu(Tensor xyz, Tensor new_xyz, Tensor idx, Tensor dist2, int b, 
                      int n, int m, int nsample) {
  // param new_xyz: (B, m, 3)
  // param xyz: (B, n, 3)
  // param idx: (B, m, nsample)
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      new_xyz.scalar_type(), "knn_forward_cpu", [&] {
        knn_forward_cpu_kernel<scalar_t>(
            b, n, m, nsample, xyz.data_ptr<scalar_t>(),
            new_xyz.data_ptr<scalar_t>(), idx.data_ptr<int>(),
            dist2.data_ptr<scalar_t>());
      });
}
