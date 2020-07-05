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

#define VECTORIZATION_ALIGN_BYTES 16
#define VEC_CALL  
#define FORCE_INLINE inline
using index = std::size_t;

}  // namespace Elastic

//#define DEBUG_INFO
//#define _DEBUG

#define CHECK_CON(condition, message)  \
  if (condition) {                     \
    std::cout << message << std::endl; \
    abort();                           \
  }