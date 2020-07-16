#pragma once
#include <assert.h>
#include <immintrin.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>

namespace Elastic {
namespace type {
enum device { cpu, gpu };
const int container = 0;
const int keepDim = 1;
const int complex = 3;
}  // namespace type
using index = std::size_t;
}  // namespace Elastic

//#define DEBUG_INFO
//#define _DEBUG
#define VECTORIZATION_ALIGN_BYTES 32
#define VEC_CALL

#ifdef _MSC_VER
#define FORCE_INLINE __forceinline
#else
#define FORCE_INLINE inline __attribute__((always_inline))
#endif
#ifdef __CUDACC__
#define ELASTIC_CALL __device__ __host__
#else
#define ELASTIC_CALL
#endif

#define CHECK_CON(condition, message)  \
  if (condition) {                     \
    std::cout << message << std::endl; \
    abort();                           \
  }