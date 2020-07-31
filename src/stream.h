#pragma once
#include "base.h"

namespace Elastic {
template <typename type::device Device>
struct Stream;

template <>
struct Stream<type::device::cpu> {};

#if ELASTIC_USE_CUDA
template <>
struct Stream<type::device::gpu> {
  cudaStream_t stream = NULL;
  cublasHandle_t cublas_handle = NULL;

  ~Stream() {
    ELASTIC_CUDA_CALL(cudaStreamDestroy(stream))
    destoryBlasHandle();
  }

  inline static cudaStream_t getStream(Stream<type::device::gpu>* _stream) {
    if (_stream)
      return _stream->stream;
    else
      return 0;
  }

  inline static cublasHandle_t getBlasHandle(
      Stream<type::device::gpu>* _stream) {
    if (_stream)
      return _stream->cublas_handle;
    else
      return 0;
  }

  inline void destoryBlasHandle() {
    if (cublas_handle != NULL) {
      CHECK_CON(cublasDestroy(cublas_handle) == CUBLAS_STATUS_SUCCESS,
                "Fail to destory cublas handle")
    }
  }

  inline void createBlasHandle() {
    destoryBlasHandle();
    CHECK_CON(cublasCreate(&cublas_handle) == CUBLAS_STATUS_SUCCESS,
              "Fail to create cublas handle")
    CHECK_CON(cublasSetStream(cublas_handle, stream) == CUBLAS_STATUS_SUCCESS,
              "Fail to set cublas stream")
  }
};
#endif
}  // namespace Elastic