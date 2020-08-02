#pragma once
#include "Shape.h"
#include "base.h"
#include "stream.h"
#include "tensor_imp_gpu.cuh"

namespace Elastic {
template <typename Derived, typename Dtype, int exp_type>
class ExpBase;

template <typename T, int Dim, typename type::device Device>
class Tensor;

template <typename ExpType>
struct ImpExp;

template <typename T>
struct ExpTraits;

template <typename Dtype, typename Op>
struct ScalarSaver;

template <typename type::device Device_1, typename type::device Device_2,
          typename T, int Dim>
inline void Copy(const Tensor<T, Dim, Device_1>& src,
                 Tensor<T, Dim, Device_2>& dst);

template <typename T, int Dim>
inline void Copy(const Tensor<T, Dim, type::device::gpu>& src,
                 Tensor<T, Dim, type::device::cpu>& dst) {
  auto _shape = src.shape.Flat2d();
  ELASTIC_CUDA_CALL(cudaMemcpy2D(dst.m_storage, dst.ld * sizeof(T), src.m_storage,
               src.ld * sizeof(T), _shape.dimx() * sizeof(T),
               _shape.size() / _shape.dimx(), cudaMemcpyDeviceToHost));
}

template <typename T, int Dim>
inline void Copy(const Tensor<T, Dim, type::device::cpu>& src,
                 Tensor<T, Dim, type::device::gpu>& dst) {
  auto _shape = src.shape.Flat2d();
  ELASTIC_CUDA_CALL(cudaMemcpy2D(dst.m_storage, dst.ld * sizeof(T), src.m_storage,
               src.ld * sizeof(T), _shape.dimx() * sizeof(T),
               _shape.size() / _shape.dimx(), cudaMemcpyHostToDevice));
}

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

template <typename Op, typename T, int Dim, typename ExpType, typename Dtype,
          int exp_type>
FORCE_INLINE void ExpEngineExcutor(
    Tensor<T, Dim, type::device::gpu>* _dst,
    const ExpBase<ExpType, Dtype, exp_type>& _exp) {
  static_assert(ExpTraits<ExpType>::dev == type::device::None ||
                    type::device::gpu == ExpTraits<ExpType>::dev,
                "Target's device(gpu) does't match the device of Op(cpu)");

  auto exp = ImpExp<ExpType>(_exp.derived_to());
  auto dst = ImpExp<Tensor<T, Dim, type::device::gpu>>(_dst->derived_to());
  auto _stream = Stream<type::device::gpu>::getStream(_dst->stream);

  cuda::KernelEntry<ScalarSaver<T, Op>, Tensor<T, Dim, type::gpu>, ExpType,
                    Dtype>(dst, exp, _stream, _dst->shape.Flat2d(), _dst->ld);
}

}  // namespace Elastic