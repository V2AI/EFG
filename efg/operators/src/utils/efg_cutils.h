/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor.")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous.")
#define CHECK_CONTIGUOUS_CUDA(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x) 

#define CHECK_IS_INT(x)                                    \
    do {                                                    \
        TORCH_CHECK(x.scalar_type()==at::ScalarType::Int,   \
                    #x " must be a int tensor");            \
    } while (0) 
#define CHECK_IS_FLOAT(x)                                       \
    do {                                                        \
        TORCH_CHECK(x.scalar_type()==at::ScalarType::Float,    \
                    #x " must be a float tensor");              \
    } while (0) 
#define CHECK_IS_LONG(x)                                       \
    do {                                                        \
        TORCH_CHECK(x.scalar_type()==at::ScalarType::Long,    \
                    #x " must be a long tensor");              \
    } while (0) 
#define CHECK_IS_BOOL(x)                                       \
    do {                                                        \
        TORCH_CHECK(x.scalar_type()==at::ScalarType::Bool,    \
                    #x " must be a bool tensor");              \
    } while (0)
