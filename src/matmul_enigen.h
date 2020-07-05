#pragma once
#include "ExpBase.h"
#include "Gemm.h"
#include "Shape.h"
#include "base.h"
#include "memory.h"

namespace Elastic {
template <typename ExpType>
class ImpExp;

template <typename T, int Dim, typename type::device Device>
class Tensor;

struct PlayGround {
  index size = 0;
  std::vector<index> offset;

  void* buffer = nullptr;
  PlayGround(index _size) : size(_size) { alloc(_size); }
  PlayGround() {}
  ~PlayGround() {
    if (buffer) aligned_free(buffer);
  }

  inline void alloc(index _size) {
    buffer = aligned_alloc(_size, VECTORIZATION_ALIGN_BYTES);
  }

  template <typename ExpType, typename Dtype>
  inline index TotalBufferSize(
      const ExpBase<ExpType, Dtype, type::complex>& exp) {
    BufferSize_help_func(exp.derived_to());
    alloc(offset.back());
  }

  template <typename Op, typename SelfType, typename Dtype, int exp_type>
  inline void BufferSize_help_func(
      const UnaryExp<Op, SelfType, Dtype, exp_type>& exp) {
    BufferSize_help_func(exp._self);
  }

  template <typename Op, typename Lhs, typename Rhs, typename Dtype,
            int exp_type>
  inline void BufferSize_help_func(
      const BinaryExp<Op, Lhs, Rhs, Dtype, exp_type>& exp) {
    BufferSize_help_func(exp._lhs);
    BufferSize_help_func(exp._rhs);
  }

  template <typename Lhs, typename Rhs, typename Dtype>
  inline void BufferSize_help_func(const MatMul<Lhs, Rhs, Dtype>& exp) {
    if constexpr (ExpTraits<Lhs>::exp) {
      index off_l = offset.back() - offset.back() % VECTORIZATION_ALIGN_BYTES +
                    VECTORIZATION_ALIGN_BYTES;
      offset.push_back(off_l);
    }
    if constexpr (ExpTraits<Rhs>::exp) {
      index off_r = offset.back() - offset.back() % VECTORIZATION_ALIGN_BYTES +
                    VECTORIZATION_ALIGN_BYTES;
      offset.push_back(off_r);
    }
  }
};

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
  }

  template <
      typename DataType, typename Container,
      typename std::enable_if<std::is_arithmetic_v<DataType>>::type* = nullptr>
  FORCE_INLINE DataType& Eval(index x, index y, Container* dst) {
    if (!evaled) {
      assign_memory(dst);
      l_buffer = *const_cast<Lhs*>(&_exp._lhs);
      r_buffer = *const_cast<Rhs*>(&_exp._rhs);
      CSM::Gemm(l_buffer.data(), l_buffer.ld, r_buffer.data(), r_buffer.ld,
                d_buffer->data(), d_buffer->ld, _exp.shape[0],
                _exp._lhs.shape[1], _exp.shape[1]);
      evaled = true;
    }
    return d_buffer->coeffRef(x, y);
  }

  template <
      typename DataType, typename Container,
      typename std::enable_if<!std::is_arithmetic_v<DataType>>::type* = nullptr>
  FORCE_INLINE DataType Eval(index x, index y, Container* dst) {
    if (!evaled) {
      assign_memory(dst);
      l_buffer = *const_cast<Lhs*>(&_exp._lhs);
      r_buffer = *const_cast<Rhs*>(&_exp._rhs);
      CSM::Gemm(l_buffer.data(), l_buffer.ld, r_buffer.data(), r_buffer.ld,
                d_buffer->data(), d_buffer->ld, _exp.shape[0],
                _exp._lhs.shape[1], _exp.shape[1]);
      evaled = true;
    }
    return Packet::PacketHandle<Dtype>::load(&d_buffer->coeffRef(x, y));
  }
};

template <typename T, int Dim, typename Lhs, typename Rhs, typename Dtype>
FORCE_INLINE void ComplexEngineExcutor(Tensor<T, Dim, type::device::cpu>* _dst,
                                       const MatMul<Lhs, Rhs, Dtype>& _exp) {
  auto exp = ImpExp<MatMul<Lhs, Rhs, Dtype>>(_exp);
  exp.template Eval<T>(0, 0, _dst);
}

}  // namespace Elastic