#include "Shape.h"
#include "base.h"

namespace Elastic {
template <typename T, int Dim, typename type::device Device>
class Tensor;

template <typename type::device Device>
struct Stream;

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

};

template <typename T, int Dim>
__host__ FORCE_INLINE void AllocSpace(Tensor<T, Dim, type::device::gpu>* dst,
                                      bool pad = false) {
  if (pad)
    cuMemAllocPitch(&dst->m_storage, &dst->ld, sizeof(T) * dst->shape.dimx(),
                    dst->shape.size() / shape.dimx());
  else
    cuMemAlloc(&dst->m_storage, sizeof(T) * dst->_size);
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