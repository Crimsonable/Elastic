#pragma once
#include "ExpBase.h"
#include "Shape.h"
#include "base.h"
#include "tensor_alloc.h"
#include "tensor_imp_cpu.h"
#ifdef ELASTIC_USE_CUDA
#include "tensor_imp_gpu.h"
#endif

namespace Elastic {
template <typename ExpType>
class ImpExp;

template <typename T, int Dim, typename type::device Device>
class Tensor;

template <typename T, typename type::device Device>
struct BlasEnigen;

#ifdef ELASTIC_USE_CUDA
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
        cublasSgemm(Stream<device>::getBlasHandle(stream), l_trans, r_trans, m,
                    n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
      else if constexpr (std::is_same_v<T, double>)
        cublasDgemm(Stream<device>::getBlasHandle(stream), l_trans, r_trans, m,
                    n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
    }
  };
#endif
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

template <typename Lhs, typename Rhs, typename Dtype>
class ImpExp<MatMul<Lhs, Rhs, Dtype>> {
  const MatMul<Lhs, Rhs, Dtype>& _exp;
  Tensor<Dtype, ExpTraits<Lhs>::dim, ExpTraits<Lhs>::dev>* l_buffer;
  Tensor<Dtype, ExpTraits<Rhs>::dim, ExpTraits<Rhs>::dev>* r_buffer;
  Tensor<Dtype, ExpTraits<Lhs>::dim, ExpTraits<Lhs>::dev>* d_buffer;
  bool evaled = false;

 public:
  ImpExp(const MatMul<Lhs, Rhs, Dtype>& exp) : _exp(exp) {}

  inline void assign_memory(
      Tensor<Dtype, ExpTraits<Lhs>::dim, ExpTraits<Lhs>::dev>* dst) {
    if constexpr (ExpTraits<Lhs>::exp) {
      l_buffer = new Tensor<Dtype, ExpTraits<Lhs>::dim, ExpTraits<Lhs>::dev>(
          _exp._lhs.shape);
      AllocSpace(l_buffer);
    }

    if constexpr (ExpTraits<Rhs>::exp) {
      r_buffer = new Tensor<Dtype, ExpTraits<Rhs>::dim, ExpTraits<Rhs>::dev>(
          _exp._rhs.shape);
      AllocSpace(r_buffer);
    }
    d_buffer = dst;
    l_buffer = const_cast<Lhs*>(&_exp._lhs);
    r_buffer = const_cast<Rhs*>(&_exp._rhs);
  }

  template <
      typename DataType, typename Container,
      typename std::enable_if<std::is_arithmetic_v<DataType>>::type* = nullptr>
  FORCE_INLINE DataType& Eval(index x, index y, Container* dst) {
    if (!evaled) {
      assign_memory(dst);
      BlasEnigen<Dtype, ExpTraits<Lhs>::dev>::gemm(
          dst->stream, false, l_buffer->data(), l_buffer->ld, false,
          r_buffer->data(), r_buffer->ld, d_buffer->data(), d_buffer->ld,
          _exp.shape[0], _exp._lhs.shape[1], _exp.shape[1]);
      evaled = true;
      destory(l_buffer);
      destory(r_buffer);
      delete (l_buffer);
      delete (r_buffer);
    }
    return d_buffer->coeffRef(x, y);
  }

  template <
      typename DataType, typename Container,
      typename std::enable_if<!std::is_arithmetic_v<DataType>>::type* = nullptr>
  FORCE_INLINE DataType Eval(index x, index y, Container* dst) {
    if (!evaled) {
      assign_memory(dst);
      BlasEnigen<Dtype, ExpTraits<Container>::dev>::setStream(dst->stream);
      BlasEnigen<Dtype, ExpTraits<Lhs>::dev>::gemm(
          dst->stream, false, l_buffer->data(), l_buffer->ld, false,
          r_buffer->data(), r_buffer->ld, d_buffer->data(), d_buffer->ld,
          _exp.shape[0], _exp._lhs.shape[1], _exp.shape[1]);
      evaled = true;
      destory(l_buffer);
      destory(r_buffer);
      delete (l_buffer);
      delete (r_buffer);
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