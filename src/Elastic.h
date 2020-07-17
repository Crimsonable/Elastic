#pragma once
#include "Tensor.h"
#include "tensor_imp_cpu.h"
#if ELASTIC_USE_CUDA
#include "tensor_imp_gpu.h"
#endif