/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/torch.h>
#include <torch/extension.h>

#include "utils/efg_cutils.h"

namespace efg {


at::Tensor roll_and_window_partition_forward_cuda(
    at::Tensor & input, 
    //at::Tensor & output,
    const int B,
    const int H,
    const int W,
    const int C,
    const int shift_size,
    const int window_size);


at::Tensor roll_and_window_partition_backward_cuda(
    at::Tensor & grad_in, 
    //at::Tensor & grad_out,
    const int B,
    const int H,
    const int W,
    const int C,
    const int shift_size,
    const int window_size);


at::Tensor window_merge_and_roll_forward_cuda(
    at::Tensor & input, 
    //at::Tensor & output,
    const int B,
    const int H,
    const int W,
    const int C,
    const int shift_size,
    const int window_size);

at::Tensor window_merge_and_roll_backward_cuda(
    at::Tensor & grad_in, 
    //at::Tensor & grad_out,
    const int B,
    const int H,
    const int W,
    const int C,
    const int shift_size,
    const int window_size);


at::Tensor roll_and_window_partition_forward(
    at::Tensor & input, 
    //at::Tensor & output,
    const int B,
    const int H,
    const int W,
    const int C,
    const int shift_size,
    const int window_size){
    CHECK_INPUT(input);
    return roll_and_window_partition_forward_cuda(input, B, H, W, C, shift_size, window_size);
}


at::Tensor roll_and_window_partition_backward(
    at::Tensor & grad_in, 
    //at::Tensor & grad_out,
    const int B,
    const int H,
    const int W,
    const int C,
    const int shift_size,
    const int window_size){
    CHECK_INPUT(grad_in);
    return roll_and_window_partition_backward_cuda(grad_in, B, H, W, C, shift_size, window_size);
}


at::Tensor window_merge_and_roll_forward(
    at::Tensor & input, 
    //at::Tensor & output,
    const int B,
    const int H,
    const int W,
    const int C,
    const int shift_size,
    const int window_size){
    CHECK_INPUT(input);
    return window_merge_and_roll_forward_cuda(input, B, H, W, C, shift_size, window_size);
}


at::Tensor window_merge_and_roll_backward(
    at::Tensor & grad_in, 
    //at::Tensor & grad_out,
    const int B,
    const int H,
    const int W,
    const int C,
    const int shift_size,
    const int window_size){
    CHECK_INPUT(grad_in);
    return window_merge_and_roll_backward_cuda(grad_in, B, H, W, C, shift_size, window_size);
}

}
