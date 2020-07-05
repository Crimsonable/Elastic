#pragma once
#include <immintrin.h>
#include "base.h"
using namespace Elastic;

namespace Packet {
template <typename T>
struct PacketHandle;

template <>
struct PacketHandle<float> {
  using type = __m256;
  FORCE_INLINE static __m256 load(const float* src) {
    return _mm256_load_ps(src);
  }
  FORCE_INLINE static __m256 fill(const float* src) {
    return _mm256_broadcast_ss(src);
  }
  FORCE_INLINE static void store(float* dst, const __m256& src) {
    _mm256_store_ps(dst, src);
  }
  FORCE_INLINE static std::size_t alignment_check(const void* ptr) {
    return reinterpret_cast<std::size_t>(ptr) % 16;
  }
  FORCE_INLINE static std::size_t size() { return 8; }
};

template <>
struct PacketHandle<double> {
  using type = __m256d;
  FORCE_INLINE static __m256d load(const double* src) {
    return _mm256_load_pd(src);
  }
  FORCE_INLINE static __m256d fill(const double* src) {
    return _mm256_broadcast_sd(src);
  }
  FORCE_INLINE static void store(double* dst, const __m256d& src) {
    _mm256_store_pd(dst, src);
  }
  FORCE_INLINE static std::size_t alignment_check(const void* ptr) {
    return reinterpret_cast<std::size_t>(ptr) % 16;
  }
  FORCE_INLINE static std::size_t size() { return 4; }
};

#ifdef _MSC_VER

FORCE_INLINE __m256 operator+(const __m256& a, const __m256& b) {
  return _mm256_add_ps(a, b);
}

FORCE_INLINE __m256 operator-(const __m256& a, const __m256& b) {
  return _mm256_sub_ps(a, b);
}

FORCE_INLINE __m256d operator+(const __m256d& a, const __m256d& b) {
  return _mm256_add_pd(a, b);
}

FORCE_INLINE __m256d operator-(const __m256d& a, const __m256d& b) {
  return _mm256_sub_pd(a, b);
}

FORCE_INLINE __m256 operator*(const __m256& a, const __m256& b) {
  return _mm256_mul_ps(a, b);
}

FORCE_INLINE __m256d operator*(const __m256d& a, const __m256d& b) {
  return _mm256_mul_pd(a, b);
}

FORCE_INLINE __m256d fmadd(const __m256d& a, const __m256d& b,
                           const __m256d& c) {
  return _mm256_fmadd_pd(a, b, c);
}

FORCE_INLINE __m256 fmadd(const __m256& a, const __m256& b, const __m256& c) {
  return _mm256_fmadd_ps(a, b, c);
}
#endif
}  // namespace Packet