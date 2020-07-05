#pragma once
#include "base.h"
#include "memory.h"

namespace Elastic {
namespace Memory {
template <typename T, int size>
struct alignas(VECTORIZATION_ALIGN_BYTES) plain_array {
  T array[size + 64];
};

template <typename T>
struct alignas(VECTORIZATION_ALIGN_BYTES) plain_array<T, -1> {
  T* array = nullptr;
  plain_array(index size) { alloc(size); }

  inline void alloc(index size) {
    size += 64;
    array = mynew_fill0<T>(size, VECTORIZATION_ALIGN_BYTES);
  }

  inline void free() {
    if (array) aligned_free(array);
  }

  ~plain_array() { aligned_free(array); }
};
}  // namespace Memory
}  // namespace Elastic