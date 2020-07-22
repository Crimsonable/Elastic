#pragma once
#include "Gemm.h"
#include "Packet.h"
#include "base.h"
#include "stream.h"

namespace Elastic {
template <typename Derived, typename Dtype, int exp_type>
class ExpBase;

template <typename T, int Dim, typename type::device Device>
class Tensor;

template <typename ExpType>
struct ImpExp;

template <typename Dtype, typename Op>
struct ScalarSaver;

template <typename Dtype, typename Op>
struct PacketSaver;

template <typename T, typename type::device Device>
struct BlasEnigen;

template <typename T>
struct BlasEnigen<T, type::device::cpu> {
  constexpr static type::device device = type::device::cpu;

  inline static void setStream(Stream<device>* stream) {}

  inline static void gemm(Stream<device>* stream, bool l_trans, T* A,
                          const int& lda, bool r_trans, T* B, const int& ldb,
                          T* C, const int& ldc, int m, int n, int k) {
    CSM::Gemm(A, lda, B, ldb, C, ldc, m, n, k);
  }
};

template <typename T, int Dim>
FORCE_INLINE void AllocSpace(Tensor<T, Dim, type::device::cpu>* dst,
                             bool pad = false) {
  index real_size = dst->_size;
  index ld = dst->ld;
  if (pad) {
    index dimx = dst->shape.dimx();
    Shape<Dim> _temp = dst->shape;
    dimx = dimx / VECTORIZATION_ALIGN_BYTES * VECTORIZATION_ALIGN_BYTES +
           dimx % VECTORIZATION_ALIGN_BYTES;
    _temp[Dim - 2] = dimx;
    ld = dimx;
    real_size = _temp.size();
  }
  dst->m_storage = mynew_fill0<T>(real_size, VECTORIZATION_ALIGN_BYTES);
  dst->hasAlloc = true;
}

template <typename T, int Dim>
FORCE_INLINE void destory(Tensor<T, Dim, type::device::cpu>* dst) {
  if (dst->hasAlloc) {
    aligned_free(dst->m_storage);
    dst->m_storage = nullptr;
  }
}

template <typename Op, typename T, int Dim, typename ExpType, typename Dtype,
          int exp_type>
FORCE_INLINE void ExpEngineExcutor(
    Tensor<T, Dim, type::device::cpu>* _dst,
    const ExpBase<ExpType, Dtype, exp_type>& _exp) {
  static_assert(type::device::cpu == ExpTraits<ExpType>::dev,
                "Target's device(cpu) does't match the device of Op(gpu)");

  auto exp = ImpExp<ExpType>(_exp.derived_to());
  auto dst = ImpExp<Tensor<T, Dim, type::device::cpu>>(_dst->derived_to());
  using Container = Tensor<T, Dim, type::device::cpu>;

  auto shape = _dst->shape;
  index ld = _dst->ld;
  index last = shape.last();
  index size = shape.size();

  bool can_vec = dst.alignment_check();
  index packed_size = Packet::PacketHandle<T>::size();
  index end_vec = can_vec ? size - size % packed_size : 0;

#pragma omp parallel
  {
#pragma omp for nowait
    for (index idx = 0; idx < end_vec; idx += packed_size) {
      PacketSaver<T, Op>::template save<typename PacketHandle<T>::type>(
          dst.template Eval<T>(idx % ld, idx / ld),
          exp.template Eval<typename PacketHandle<T>::type, Container>(
              idx % ld, idx / ld, _dst));
    }
    for (index idx = end_vec; idx < size; ++idx) {
      ScalarSaver<T, Op>::save(
          dst.template Eval<T>(idx % ld, idx / ld),
          exp.template Eval<T, Container>(idx % ld, idx / ld, _dst));
    }
  }
}

}  // namespace Elastic