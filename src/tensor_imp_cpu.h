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

template <typename Op, typename T, int Dim, typename ExpType, typename Dtype,
          int exp_type>
FORCE_INLINE void ExpEngineExcutor(
    Tensor<T, Dim, type::device::cpu>* _dst,
    const ExpBase<ExpType, Dtype, exp_type>& _exp) {
  static_assert(ExpTraits<ExpType>::dev == type::device::None ||
                    type::device::cpu == ExpTraits<ExpType>::dev,
                "Target's device(cpu) does't match the device of Op(gpu)");

  auto exp = ImpExp<ExpType>(_exp.derived_to());
  auto dst = ImpExp<Tensor<T, Dim, type::device::cpu>>(_dst->derived_to());
  using Container = Tensor<T, Dim, type::device::cpu>;

  index ld = _dst->ld;
  index last = _dst->shape.last();
  index size = ld * last;

  bool can_vec = dst.alignment_check();
  index packed_size = Packet::PacketHandle<T>::size();
  index end_vec = can_vec ? size - size % packed_size : 0;

#pragma omp parallel
  {
#pragma omp for nowait
    for (index idx = 0; idx < end_vec; idx += packed_size) {
      PacketSaver<T, Op>::template save<PacketHandle<T>::type>(
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