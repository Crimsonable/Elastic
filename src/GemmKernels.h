#pragma once
#include "Packet.h"
#include "base.h"
#include "memory.h"

using namespace Packet;

namespace CSM {
FORCE_INLINE void VEC_CALL Gemm_kernel_avx256(float *a, const int &lda,
                                              float *b, const int &ldb,
                                              float *c, const int &ldc,
                                              const int &k, int leftRows,
                                              int leftCols) {
  __m256 a0, a1, b0, b1, b2, b3;
  __m256 c0, c1, c2, c3, c4, c5, c6, c7;
  int m1 = 1 % leftCols, m2 = 2 % leftCols, m3 = 3 % leftCols;

  float *b_pntr, *a_top_pntr, *a_down_pntr;
  b_pntr = b;

  a_top_pntr = a;
  a_down_pntr = a + 8;

  c0 = _mm256_load_ps(c);
  c1 = _mm256_load_ps(c + 8);
  c2 = _mm256_load_ps(c + m1 * ldc);
  c3 = _mm256_load_ps(c + m1 * ldc + 8);
  c4 = _mm256_load_ps(c + m2 * ldc);
  c5 = _mm256_load_ps(c + m2 * ldc + 8);
  c6 = _mm256_load_ps(c + m3 * ldc);
  c7 = _mm256_load_ps(c + m3 * ldc + 8);

  for (int p = 0; p < k; ++p) {
    a0 = _mm256_load_ps(a_top_pntr);
    a1 = _mm256_load_ps(a_down_pntr);
    b0 = _mm256_broadcast_ss(b_pntr);
    b_pntr += ldb;
    b1 = _mm256_broadcast_ss(b_pntr);
    b_pntr += ldb;
    b2 = _mm256_broadcast_ss(b_pntr);
    b_pntr += ldb;
    b3 = _mm256_broadcast_ss(b_pntr);
    b_pntr += ldb;  // load and duplicate

    c0 = _mm256_fmadd_ps(a0, b0, c0);
    c2 = _mm256_fmadd_ps(a0, b1, c2);
    c4 = _mm256_fmadd_ps(a0, b2, c4);
    c6 = _mm256_fmadd_ps(a0, b3, c6);
    c1 = _mm256_fmadd_ps(a1, b0, c1);
    c3 = _mm256_fmadd_ps(a1, b1, c3);
    c5 = _mm256_fmadd_ps(a1, b2, c5);
    c7 = _mm256_fmadd_ps(a1, b3, c7);

    a_top_pntr += lda;
    a_down_pntr += lda;
  }

  if (leftRows != 16) {
#ifdef _MSC_VER
    _declspec(align(VECTORIZATION_ALIGN_BYTES)) int temp[8] = {0};
#else
    __attribute__((aligned(VECTORIZATION_ALIGN_BYTES))) int temp[8] = {0};
#endif
    __m256i *mask = reinterpret_cast<__m256i *>(temp);
    if (leftRows > 8) {
      for (int i = 0; i < leftRows - 8; ++i) temp[i] = -1;
      _mm256_store_ps(c + m3 * ldc, c6);
      _mm256_maskstore_ps(c + m3 * ldc + 8, *mask, c7);
      _mm256_store_ps(c + m2 * ldc, c4);
      _mm256_maskstore_ps(c + m2 * ldc + 8, *mask, c5);
      _mm256_store_ps(c + m1 * ldc, c2);
      _mm256_maskstore_ps(c + m1 * ldc + 8, *mask, c3);
      _mm256_store_ps(c, c0);
      _mm256_maskstore_ps(c + 8, *mask, c1);
    } else {
      for (int i = 0; i < leftRows; ++i) *((int *)(&mask) + i) = -1;
      ;
      _mm256_maskstore_ps(c + m3 * ldc, *mask, c6);
      _mm256_maskstore_ps(c + m2 * ldc, *mask, c4);
      _mm256_maskstore_ps(c + m1 * ldc, *mask, c2);
      _mm256_maskstore_ps(c, *mask, c0);
    }
  } else {
    _mm256_store_ps(c + m3 * ldc + 8, c7);
    _mm256_store_ps(c + m3 * ldc, c6);
    _mm256_store_ps(c + m2 * ldc + 8, c5);
    _mm256_store_ps(c + m2 * ldc, c4);
    _mm256_store_ps(c + m1 * ldc + 8, c3);
    _mm256_store_ps(c + m1 * ldc, c2);
    _mm256_store_ps(c + 8, c1);
    _mm256_store_ps(c, c0);
  }
}
}  // namespace CSM
