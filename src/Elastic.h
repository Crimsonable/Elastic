#pragma once
#ifdef ELASTIC_CUDA_ENABLE
#define ELASTIC_USE_CUDA 1
#endif  // ELASTIC_CUDA_ENABLE
#include "base.h"
#include "Tensor.h"
#include "tensor_imp_cpu.h"
#if ELASTIC_USE_CUDA
#include "tensor_imp_gpu.h"
#endif