#pragma once

namespace CSM {
	template<typename T>
	inline void PackLhs(T* src, const int& ld1, const int& rows, const int& cols, T* dst, const int& ld2, const int& packRows) {
		const int EndVec = rows % packRows ? rows - rows % packRows : rows;
#pragma omp parallel
		{
#pragma omp for schedule(dynamic) nowait
			for (int rowIndex = 0; rowIndex < EndVec; rowIndex += packRows) {
				int offset = rowIndex * cols;
				for (int colIndex = 0; colIndex < cols; ++colIndex) {
					memcpy(dst + offset + colIndex * ld2, src + rowIndex + colIndex * ld1, sizeof(T)*packRows);
				}
			}
			for (int rowIndex = EndVec; rowIndex < rows; rowIndex += packRows) {
#pragma omp for schedule(dynamic) nowait
				for (int colIndex = 0; colIndex < cols; ++colIndex) {
					int offset = rowIndex * cols + colIndex * ld2;
					memcpy(dst + offset, src + rowIndex + colIndex * ld1, sizeof(T)*(rows - rowIndex));
					std::fill_n(dst + offset + (rows - rowIndex), packRows - rows + rowIndex, T(0));
				}
			}
		}
	}

	template<typename T>
	void PackRhs(T* src, const int& ld1, const int& rows, const int& cols, T* dst, const int& ld2, const int& packCols) {
		const int EndVec = cols % packCols ? cols - cols % packCols : cols;
#pragma omp parallel
		{
#pragma omp for schedule(dynamic) nowait
			for (int colIndex = 0; colIndex < EndVec; colIndex += packCols) {
				int offset = colIndex * rows;
				for (int k = 0; k < packCols; ++k) {
					for (int rowIndex = 0; rowIndex < rows; ++rowIndex) {
						*(dst + offset + k + rowIndex * ld2) = *(src + colIndex * ld1 + rowIndex + k * ld1);
					}
				}
			}
			for (int colIndex=EndVec; colIndex < cols; colIndex += packCols) {
#pragma omp for schedule(dynamic) nowait
				for (int rowIndex = 0; rowIndex < rows; ++rowIndex) {
					int offset = rowIndex * ld2 + colIndex * rows;
					int k = 0;
					for (; k < cols - colIndex; ++k) {
						*(dst + k + offset) = *(src + (colIndex + k)*ld1 + rowIndex);
					}
					std::fill_n(dst + offset + k, packCols - cols + colIndex, T(0));
				}
			}
		}
	}
}