#include "pytorch_cpp_helper.hpp"
#include "bbox_overlaps_cpu_kernel.hpp"

void bbox_overlaps_cpu(const Tensor bboxes1, const Tensor bboxes2, Tensor ious,
                   const int mode, const bool aligned, const int offset) {
    bbox_overlaps_cpu_kernel(bboxes1, bboxes2, ious, mode, aligned, offset);

}