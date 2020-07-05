#pragma once
#include "GemmInnerLoop.h"
#include "Pack.h"
#include "base.h"

#ifdef HAS_CUDA
#include "CublasProduct.h"
#endif  // HAS_CUDA

#ifdef _DEBUG
#include "testTools.h"
#endif  // _DEBUG

#define m_kernel 128
#define k_kernel 416
#define inner_rows 16
#define inner_cols 4
#define basic_step 4
#define _min_(i, j) ((i) < (j) ? (i) : (j))
#define ELEMENT(i, j, k, p) c_##i##j##_c_##k##j##_vreg.d[p]
#define VELEMENT(i, j, k) c_##i##j##_c_##k##j##_vreg.v

/*
A:	num*[m_kernel*k_kernel]		B:	  Bnum*[k_kernel*n]
         ________  ________				 ____________________
        |________||________|			|____________________|
        |________||________|			|____________________|
        |________||________|			|____________________|
        |________||________|			|____________________|
        |________||________|			|____________________|
        |________||________|			|____________________|
        |________||________|			|____________________|
        |________||________|			|____________________|

*/

namespace CSM {
template <typename T>
struct GemmImp {
  T *packA = nullptr, *packB = nullptr;
  static constexpr int KernelRowSize = 16 / sizeof(T) * sizeof(float);
  static constexpr int KernelColSize = 4;

  ~GemmImp() {
    if (packA) aligned_free(packA);
    if (packB) aligned_free(packB);
  }

  FORCE_INLINE void GEMM_Kernel(T* a, const int& lda, T* b, const int& ldb,
                                T* c, const int ldc, int m, int n, int k,
                                bool start) {
    // alloc memory for packing
    const std::size_t packedARows =
        m % KernelRowSize == 0 ? m : (m + KernelRowSize - m % KernelRowSize);
    const std::size_t packedBCols =
        n % KernelColSize == 0 ? n : (n + KernelColSize - n % KernelColSize);

    if (!packA && !packB) {
      packA = mynew<T>(packedARows * k, VECTORIZATION_ALIGN_BYTES);
      packB = mynew<T>(packedBCols * k, VECTORIZATION_ALIGN_BYTES);
    }
    // pack InnerKernel A(rm*rk) into packedA
    PackLhs(a, lda, m, k, packA, KernelRowSize, KernelRowSize);
    if (start) PackRhs(b, ldb, k, n, packB, KernelColSize, KernelColSize);

#ifdef DEBUG_INFO
    DEBUG_TOOLS::printRawMatrix(packA, KernelRowSize,
                                packedARows / KernelRowSize * k, "packA: ");
    DEBUG_TOOLS::printRawMatrix(packB, KernelColSize,
                                packedBCols / KernelColSize * k, "packB: ");
    DEBUG_TOOLS::printRawMatrix(a, m, k, "A:");
    DEBUG_TOOLS::printRawMatrix(b, k, n, "B:");
#endif  // _DEBUG

    GemmInnerLoop(packA, KernelRowSize, packB, KernelColSize, c, ldc, m, n, k,
                  KernelRowSize, KernelColSize);
  }
};

template <typename T>
void Gemm(T* A, const int& lda, T* B, int const& ldb, T* C, int const ldc,
          int m, int n, int k) {
  int colStep = k_kernel, rowStep = m_kernel;
  auto handle = GemmImp<float>();
  for (int colIndex = 0; colIndex < k; colIndex += colStep) {
    colStep = _min_(k - colIndex, k_kernel);
    for (int rowIndex = 0; rowIndex < m; rowIndex += rowStep) {
      rowStep = _min_(m - rowIndex, m_kernel);
      handle.GEMM_Kernel(A + rowIndex + colIndex * lda, lda, B + colIndex, ldb,
                         C + rowIndex, ldc, rowStep, n, colStep, rowIndex == 0);
    }
  }
}

}  // namespace CSM
