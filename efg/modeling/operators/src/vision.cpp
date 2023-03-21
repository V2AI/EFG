#include <torch/types.h>
#include <torch/extension.h>
#include "box_attn/box_attn.h"
#include "voxelize/voxelization.h"
#include "iou_box3d/iou_box3d.h"
#include "box_iou_rotated/box_iou_rotated.h"
#include "box_iou_rotated_diff/sort_vert.h"
#include "window_process/swin_window_process.h"
#include "iou3d_nms/iou3d_cpu.h"
#include "iou3d_nms/iou3d_nms.h"
#include "deform_attn/ms_deform_attn.h"
#include "deform_conv/deform_conv.h"
#include "cocoeval/cocoeval.h"

namespace efg {

#ifdef WITH_CUDA
extern int get_cudart_version();
#endif

std::string get_cuda_version() {
#ifdef WITH_CUDA
  std::ostringstream oss;

  // copied from
  // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/cuda/detail/CUDAHooks.cpp#L231
  auto printCudaStyleVersion = [&](int v) {
    oss << (v / 1000) << "." << (v / 10 % 100);
    if (v % 10 != 0) {
      oss << "." << (v % 10);
    }
  };
  printCudaStyleVersion(get_cudart_version());
  return oss.str();
#else
  return std::string("not available");
#endif
}

// similar to
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Version.cpp
std::string get_compiler_version() {
  std::ostringstream ss;
#if defined(__GNUC__)
#ifndef __clang__

#if ((__GNUC__ <= 4) && (__GNUC_MINOR__ <= 8))
#error "GCC >= 4.9 is required!"
#endif

  { ss << "GCC " << __GNUC__ << "." << __GNUC_MINOR__; }
#endif
#endif

#if defined(__clang_major__)
  {
    ss << "clang " << __clang_major__ << "." << __clang_minor__ << "."
       << __clang_patchlevel__;
  }
#endif

#if defined(_MSC_VER)
  { ss << "MSVC " << _MSC_FULL_VER; }
#endif
  return ss.str();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    m.def("get_compiler_version", &get_compiler_version, "get_compiler_version");
    m.def("get_cuda_version", &get_cuda_version, "get_cuda_version");

    m.def("box_attn_forward", &box_attn_forward, "box_attn_forward");
    m.def("box_attn_backward", &box_attn_backward, "box_attn_backward");

    m.def("hard_voxelize", &hard_voxelize, "hard voxelize");
    m.def("dynamic_voxelize", &dynamic_voxelize, "dynamic voxelization");
    m.def("dynamic_point_to_voxel_forward", &dynamic_point_to_voxel_forward, "dynamic point to voxel forward");
    m.def("dynamic_point_to_voxel_backward", &dynamic_point_to_voxel_backward, "dynamic point to voxel backward");

    // 3D IoU
    m.def("iou_box3d", &IoUBox3D);

    // sort_vertices
    m.def("sort_vertices_forward", &sort_vertices, "sort vertices of a convex polygon. forward only");

    // Swin Window Process
    m.def("roll_and_window_partition_forward", &roll_and_window_partition_forward, "torch.roll and window_partition.");
    m.def("roll_and_window_partition_backward", &roll_and_window_partition_backward, "torch.roll and window_partition.");
    m.def("window_merge_and_roll_forward", &window_merge_and_roll_forward, "window merge and torch.roll.");
    m.def("window_merge_and_roll_backward", &window_merge_and_roll_backward, "window merge and torch.roll.");
    
    // IOU3d NMS
    m.def("boxes_overlap_bev_gpu", &boxes_overlap_bev_gpu, "oriented boxes overlap");
    m.def("boxes_iou_bev_gpu", &boxes_iou_bev_gpu, "oriented boxes iou");
    m.def("nms_gpu", &nms_gpu, "oriented nms gpu");
    m.def("nms_normal_gpu", &nms_normal_gpu, "nms gpu");
    m.def("boxes_iou_bev_cpu", &boxes_iou_bev_cpu, "oriented boxes iou");

    // Deformable Attention
    m.def("ms_deform_attn_forward", &ms_deform_attn_forward, "ms_deform_attn_forward");
    m.def("ms_deform_attn_backward", &ms_deform_attn_backward, "ms_deform_attn_backward");

    m.def("box_iou_rotated", &box_iou_rotated, "IoU for rotated boxes");
    m.def("deform_conv_forward", &deform_conv_forward, "deform_conv_forward");
    m.def("deform_conv_backward_input", &deform_conv_backward_input, "deform_conv_backward_input");
    m.def("deform_conv_backward_filter", &deform_conv_backward_filter, "deform_conv_backward_filter");
    m.def("modulated_deform_conv_forward", &modulated_deform_conv_forward, "modulated_deform_conv_forward");
    m.def("modulated_deform_conv_backward", &modulated_deform_conv_backward, "modulated_deform_conv_backward");

    m.def("COCOevalAccumulate", &COCOeval::Accumulate, "COCOeval::Accumulate");
    m.def("COCOevalEvaluateImages", &COCOeval::EvaluateImages, "COCOeval::EvaluateImages");
    pybind11::class_<COCOeval::InstanceAnnotation>(m, "InstanceAnnotation").def(pybind11::init<uint64_t, double, double, bool, bool>());
    pybind11::class_<COCOeval::ImageEvaluation>(m, "ImageEvaluation").def(pybind11::init<>());

}

}
