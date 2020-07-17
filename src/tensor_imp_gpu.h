#pragma once
#include "Shape.h"
#include "base.h"
#include "stream.h"

namespace Elastic {
template <typename Derived, typename Dtype, int exp_type>
class ExpBase;

template <typename T, int Dim, typename type::device Device>
class Tensor;

template <typename ExpType>
struct ImpExp;

template <typename type::device Device, ENABLE_IF(Device == type::device::gpu)>
inline Stream<Device>* NewStream(bool create_blas_handle) {
  struct StreamDealloctor {
    void operator()(Stream<type::device::gpu>* ptr) const {
      ptr->~Stream();
      ptr = nullptr;
    }
  };
  std::unique_ptr<Stream<type::device::gpu>, StreamDealloctor> stream_ptr(
      new Stream<type::device::gpu>());
  ELASTIC_CUDA_CALL(cudaStreamCreate(&stream_ptr->stream))
  if (create_blas_handle) stream_ptr->createBlasHandle();
  return stream_ptr.release();
}

template <typename T, typename type::device Device>
struct BlasEnigen;

template <typename T>
struct BlasEnigen<T, type::device::gpu> {
  constexpr static type::device device = type::device::gpu;

  inline static void setStream(Stream<device>* stream) {
    CHECK_CON(cublasSetStream(Stream<device>::getBlasHandle(stream),
                              Stream<device>::getStream(stream)),
              "Fail to set stream for cublas")
  }

  inline static void gemm(Stream<device>* stream, bool l_trans, T* A,
                          const int& lda, bool r_trans, T* B, const int& ldb,
                          T* C, const int& ldc, int m, int n, int k) {
    T alpha = 1.0, beta = 0.0;
    if constexpr (std::is_same_v<T, float>)
      cublasSgemm(Stream<device>::getBlasHandle(stream), l_trans, r_trans, m, n,
                  k, &alpha, A, lda, B, ldb, &beta, C, ldc);
    else if constexpr (std::is_same_v<T, double>)
      cublasDgemm(Stream<device>::getBlasHandle(stream), l_trans, r_trans, m, n,
                  k, &alpha, A, lda, B, ldb, &beta, C, ldc);
  }
};

template <typename T, int Dim>
__host__ FORCE_INLINE void AllocSpace(Tensor<T, Dim, type::device::gpu>* dst,
                                      bool pad = false) {
  if (pad)
    cudaMallocPitch(&dst->m_storage, &dst->ld, sizeof(T) * dst->shape.dimx(),
                    dst->shape.size() / dst->shape.dimx());
  else
    cudaMalloc(&dst->m_storage, sizeof(T) * dst->_size);
  dst->hasAlloc = true;
}

template <typename T, int Dim>
__host__ FORCE_INLINE void destory(Tensor<T, Dim, type::device::gpu>* dst) {
  if (dst->hasAlloc) {
    cudaFree(dst->m_storage);
    dst->m_storage = nullptr;
  }
}
}  // namespace Elastic