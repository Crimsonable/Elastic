#pragma once
#include "base.h"
#include "GemmKernels.h"

namespace CSM {

	template<typename T>
	inline void GemmInnerLoop(T* a, const int& lda, T* b, const int& ldb, T* c, const int& ldc, int m, int n, int k, const int& kernelRows,const int& kernelCols) {
#pragma omp parallel
		{
#pragma omp for schedule(dynamic) nowait
			for (int colIndex = 0; colIndex < n; colIndex += kernelCols) {
				int offset_b = colIndex * k;
				int offset_c = colIndex * ldc;
				int leftCols = n - colIndex > 4 ? 4 : n - colIndex;
				for (int rowIndex = 0; rowIndex < m; rowIndex += kernelRows) {
					Gemm_kernel_avx256(a + rowIndex * k, lda, b + offset_b, 1, c + rowIndex + offset_c, ldc, k, m - rowIndex > 16 ? 16 : m - rowIndex, leftCols);
				}
			}
		}
	}
}
