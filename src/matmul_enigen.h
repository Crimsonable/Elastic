#pragma once
#include "ExpBase.h"
#include "Shape.h"
#include "base.h"
#include "memory.h"
#include "tensor_imp_cpu.h"

namespace Elastic {
template <typename ExpType>
class ImpExp;

template <typename T, int Dim, typename type::device Device>
class Tensor;

template <typename Lhs, typename Rhs, typename Dtype>
class ImpExp<MatMul<Lhs, Rhs, Dtype>> {
  const MatMul<Lhs, Rhs, Dtype>& _exp;
  Tensor<Dtype, ExpTraits<Lhs>::dim, ExpTraits<Lhs>::dev> l_buffer;
  Tensor<Dtype, ExpTraits<Rhs>::dim, ExpTraits<Rhs>::dev> r_buffer;
  Tensor<Dtype, ExpTraits<Lhs>::dim, ExpTraits<Lhs>::dev>* d_buffer;
  bool evaled = false;

 public:
  ImpExp(const MatMul<Lhs, Rhs, Dtype>& exp) : _exp(exp) {}

  inline void assign_memory(
      Tensor<Dtype, ExpTraits<Lhs>::dim, ExpTraits<Lhs>::dev>* dst) {
    if constexpr (ExpTraits<Lhs>::exp) {
      l_buffer = Tensor<Dtype, ExpTraits<Lhs>::dim, ExpTraits<Lhs>::dev>(
          _exp._lhs.shape);
      l_buffer.alloc();
    }

    if constexpr (ExpTraits<Rhs>::exp) {
      r_buffer = Tensor<Dtype, ExpTraits<Rhs>::dim, ExpTraits<Rhs>::dev>(
          _exp._rhs.shape);
      r_buffer.alloc();
    }
    d_buffer = dst;
    l_buffer = *const_cast<Lhs*>(&_exp._lhs);
    r_buffer = *const_cast<Rhs*>(&_exp._rhs);
  }

  template <
      typename DataType, typename Container,
      typename std::enable_if<std::is_arithmetic_v<DataType>>::type* = nullptr>
  FORCE_INLINE DataType& Eval(index x, index y, Container* dst) {
    if (!evaled) {
      assign_memory(dst);
      BlasEnigen<Dtype, ExpTraits<Lhs>::dev>::gemm(
          dst->stream, false, l_buffer.data(), l_buffer.ld, false,
          r_buffer.data(), r_buffer.ld, d_buffer->data(), d_buffer->ld,
          _exp.shape[0], _exp._lhs.shape[1], _exp.shape[1]);
      evaled = true;
      destory(&l_buffer);
      destory(&r_buffer);
    }
    return d_buffer->coeffRef(x, y);
  }

  template <
      typename DataType, typename Container,
      typename std::enable_if<!std::is_arithmetic_v<DataType>>::type* = nullptr>
  FORCE_INLINE DataType Eval(index x, index y, Container* dst) {
    if (!evaled) {
      assign_memory(dst);
      BlasEnigen<Dtype, ExpTraits<Lhs>::dev>::gemm(
          dst->stream, false, l_buffer.data(), l_buffer.ld, false,
          r_buffer.data(), r_buffer.ld, d_buffer->data(), d_buffer->ld,
          _exp.shape[0], _exp._lhs.shape[1], _exp.shape[1]);
      evaled = true;
      destory(&l_buffer);
      destory(&r_buffer);
    }
    return Packet::PacketHandle<Dtype>::load(&d_buffer->coeffRef(x, y));
  }
};

template <typename T, int Dim, typename Lhs, typename Rhs,
          typename type::device Device, typename Dtype>
FORCE_INLINE void ComplexEngineExcutor(Tensor<T, Dim, Device>* _dst,
                                       const MatMul<Lhs, Rhs, Dtype>& _exp) {
  auto exp = ImpExp<MatMul<Lhs, Rhs, Dtype>>(_exp);
  exp.template Eval<T>(0, 0, _dst);
}

}  // namespace Elastic