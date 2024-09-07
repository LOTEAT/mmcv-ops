#include <torch/extension.h>

#include "pytorch_cpp_helper.hpp"

void roi_align_forward_cuda(Tensor input, Tensor rois, Tensor output,
                       Tensor argmax_y, Tensor argmax_x, int aligned_height,
                       int aligned_width, float spatial_scale,
                       int sampling_ratio, int pool_mode, bool aligned);

void roi_align_backward_cuda(Tensor grad_output, Tensor rois, Tensor argmax_y,
                        Tensor argmax_x, Tensor grad_input, int aligned_height,
                        int aligned_width, float spatial_scale,
                        int sampling_ratio, int pool_mode, bool aligned);

void roi_align_forward_cpu(Tensor input, Tensor rois, Tensor output,
                                Tensor argmax_y, Tensor argmax_x,
                                int aligned_height, int aligned_width,
                                float spatial_scale, int sampling_ratio,
                                int pool_mode, bool aligned);

void roi_align_backward_cpu(Tensor grad_output, Tensor rois,
                                 Tensor argmax_y, Tensor argmax_x,
                                 Tensor grad_input, int aligned_height,
                                 int aligned_width, float spatial_scale,
                                 int sampling_ratio, int pool_mode,
                                 bool aligned);
void bbox_overlaps_cpu(const Tensor bboxes1, const Tensor bboxes2, Tensor ious,
                   const int mode, const bool aligned, const int offset);

void bbox_overlaps_cuda(const Tensor bboxes1, const Tensor bboxes2, Tensor ious,
                   const int mode, const bool aligned, const int offset);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("roi_align_forward_cuda", &roi_align_forward_cuda, "roi align forward cuda kernel",
        py::arg("input"), py::arg("rois"), py::arg("output"),
        py::arg("argmax_y"), py::arg("argmax_x"), py::arg("aligned_height"),
        py::arg("aligned_width"), py::arg("spatial_scale"),
        py::arg("sampling_ratio"), py::arg("pool_mode"), py::arg("aligned"));
  m.def("roi_align_backward_cuda", &roi_align_backward_cuda, "roi align backward cuda kernel",
        py::arg("grad_output"), py::arg("rois"), py::arg("argmax_y"),
        py::arg("argmax_x"), py::arg("grad_input"), py::arg("aligned_height"),
        py::arg("aligned_width"), py::arg("spatial_scale"),
        py::arg("sampling_ratio"), py::arg("pool_mode"), py::arg("aligned"));
  m.def("roi_align_forward_cpu", &roi_align_forward_cpu, "roi align forward cpu kernel",
        py::arg("input"), py::arg("rois"), py::arg("output"),
        py::arg("argmax_y"), py::arg("argmax_x"), py::arg("aligned_height"),
        py::arg("aligned_width"), py::arg("spatial_scale"),
        py::arg("sampling_ratio"), py::arg("pool_mode"), py::arg("aligned"));
  m.def("roi_align_backward_cpu", &roi_align_backward_cpu, "roi align backward cpu kernel",
        py::arg("grad_output"), py::arg("rois"), py::arg("argmax_y"),
        py::arg("argmax_x"), py::arg("grad_input"), py::arg("aligned_height"),
        py::arg("aligned_width"), py::arg("spatial_scale"),
        py::arg("sampling_ratio"), py::arg("pool_mode"), py::arg("aligned"));  
  m.def("bbox_overlaps_cpu", &bbox_overlaps_cpu, "bbox overlaps cpu kernel", py::arg("bboxes1"),
        py::arg("bboxes2"), py::arg("ious"), py::arg("mode"),
        py::arg("aligned"), py::arg("offset"));
  m.def("bbox_overlaps_cuda", &bbox_overlaps_cuda, "bbox overlaps cuda kernel", py::arg("bboxes1"),
        py::arg("bboxes2"), py::arg("ious"), py::arg("mode"),
        py::arg("aligned"), py::arg("offset"));
}
