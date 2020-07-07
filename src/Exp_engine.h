#pragma once
#include "ExpBase.h"
#include "OpRegister.h"
#include "Tensor.h"
#include "base.h"
#include "matmul_enigen.h"
#include "rand.h"

using Packet::PacketHandle;

namespace Elastic {
template <typename T, int Dim, typename type::device Device>
class Tensor;

template <typename Container, typename Dtype>
class ComplexEngine;

template <typename ExpType>
struct ImpExp;

template <typename T>
struct ImpExp<Scalar<T>> {
  const T _data;

  ImpExp(const Scalar<T>& data) : _data(data.data) {}

  FORCE_INLINE bool alignment_check() { return true; }

  template <
      typename DataType, typename Container,
      typename std::enable_if<std::is_arithmetic_v<DataType>>::type* = nullptr>
  FORCE_INLINE DataType Eval(index x, index y, Container* dst = nullptr) const {
    return _data;
  }

  template <
      typename DataType, typename Container,
      typename std::enable_if<!std::is_arithmetic_v<DataType>>::type* = nullptr>
  FORCE_INLINE DataType Eval(index x, index y, Container* dst = nullptr) const {
    return Packet::PacketHandle<T>::fill(const_cast<T*>(&_data));
  }
};

template <typename T, int Dim, typename type::device Device>
struct ImpExp<Tensor<T, Dim, Device>> {
  const Tensor<T, 2, Device>& _data;

  ImpExp(const Tensor<T, Dim, Device>& data) : _data(data) {}

  FORCE_INLINE bool alignment_check() {
    return Packet::PacketHandle<T>::alignment_check(_data.m_storage);
  }

  template <
      typename DataType, typename Container = void,
      typename std::enable_if<std::is_arithmetic_v<DataType>>::type* = nullptr>
  FORCE_INLINE DataType& Eval(index x, index y, Container* dst = nullptr) {
    return const_cast<Tensor<T, Dim, Device>*>(&_data)->coeffRef(x, y);
  }

  template <
      typename DataType, typename Container = void,
      typename std::enable_if<!std::is_arithmetic_v<DataType>>::type* = nullptr>
  FORCE_INLINE DataType Eval(index x, index y, Container* dst = nullptr) {
    return Packet::PacketHandle<T>::load(
        &const_cast<Tensor<T, Dim, Device>*>(&_data)->coeffRef(x, y));
  }
};

template <int Method, typename Dtype>
struct ImpExp<InitExp<Method, Dtype>> {
  const InitExp<Method, Dtype>& _exp;

  ImpExp(const InitExp<Method, Dtype>& exp) : _exp(exp) {}

  FORCE_INLINE bool alignment_check() { return true; }

  template <typename T, typename Container = void>
  FORCE_INLINE T Eval(index x, index y, Container* dst = nullptr) const {
    return const_cast<InitExp<Method, Dtype>*>(&_exp)->template Eval<T>(x, y,
                                                                        dst);
  }
};

template <typename Op, typename Lhs, typename Rhs, typename Dtype, int exp_type>
struct ImpExp<BinaryExp<Op, Lhs, Rhs, Dtype, exp_type>> {
  ImpExp<Lhs> _lhs;
  ImpExp<Rhs> _rhs;

  ImpExp(const BinaryExp<Op, Lhs, Rhs, Dtype, exp_type>& exp)
      : _lhs(exp._lhs), _rhs(exp._rhs) {}

  FORCE_INLINE bool alignment_check() {
    return _lhs.alignment_check() && _rhs.alignment_check();
  }

  template <typename DataType, typename Container = void>
  FORCE_INLINE DataType Eval(index x, index y, Container* dst = nullptr) const {
    return Op::apply(
        const_cast<ImpExp<Lhs>*>(&_lhs)->template Eval<DataType, Container>(
            x, y, dst),
        const_cast<ImpExp<Rhs>*>(&_rhs)->template Eval<DataType, Container>(
            x, y, dst));
  }
};

template <typename Op, typename SelfType, typename Dtype, int exp_type>
struct ImpExp<UnaryExp<Op, SelfType, Dtype, exp_type>> {
  ImpExp<SelfType> _self;

  ImpExp(const UnaryExp<Op, SelfType, Dtype, exp_type>& exp)
      : _self(exp._self) {}

  FORCE_INLINE bool alignment_check() { return _self.alignment_check(); }

  template <typename DataType, typename Container = void>
  FORCE_INLINE DataType Eval(index x, index y, Container* dst = nullptr) const {
    return Op::apply(_self.template Eval<DataType>(x, y, dst));
  }
};

template <typename Dtype, typename Op>
struct ScalarSaver;

template <typename Dtype, typename Op>
struct ScalarSaver {
  FORCE_INLINE static void save(Dtype& dst, const Dtype& src) {
    Op::apply(dst, src);
  }
};

template <typename Dtype, typename Op>
struct PacketSaver {
  template <typename T>
  FORCE_INLINE static void save(Dtype& dst, const T& src) {
    Op::apply(dst, src);
  }
};

template <typename Op, typename T, int Dim, typename ExpType, typename Dtype>
FORCE_INLINE void ExpEngineExcutor(
    Tensor<T, Dim, type::device::cpu>* _dst,
    const ExpBase<ExpType, Dtype, type::keepDim>& _exp) {
  auto exp = ImpExp<ExpType>(_exp.derived_to());
  auto dst = ImpExp<Tensor<T, Dim, type::device::cpu>>(_dst->derived_to());

  auto shape = _dst->shape;
  index ld = _dst->ld;
  index packed_size = Packet::PacketHandle<T>::size();
  index last = shape.last();
  index size = shape.size();

  bool can_vec = dst.alignment_check();
  index end_vec = can_vec ? size - size % packed_size : 0;

#pragma omp parallel
  {
#pragma omp for nowait
    for (index idx = 0; idx < end_vec; idx += packed_size) {
      PacketSaver<T, Op>::template save<typename PacketHandle<T>::type>(
          dst.template Eval<T>(idx % ld, idx / ld),
          exp.template Eval<typename PacketHandle<T>::type>(idx % ld,
                                                            idx / ld));
    }
    for (index idx = end_vec; idx < size; ++idx) {
      ScalarSaver<T, Op>::save(dst.template Eval<T>(idx % ld, idx / ld),
                               exp.template Eval<T>(idx % ld, idx / ld));
    }
  }
}

template <typename Op, typename T, int Dim, typename ExpType, typename Dtype>
FORCE_INLINE void ExpEngineExcutor(
    Tensor<T, Dim, type::device::cpu>* _dst,
    const ExpBase<ExpType, Dtype, type::complex>& _exp) {
  auto exp = ImpExp<ExpType>(_exp.derived_to());
  auto dst = ImpExp<Tensor<T, Dim, type::device::cpu>>(_dst->derived_to());
  using Container = Tensor<T, Dim, type::device::cpu>;

  auto shape = _dst->shape;
  index ld = _dst->ld;
  index packed_size = Packet::PacketHandle<T>::size();
  index last = shape.last();
  index size = shape.size();

  bool can_vec = dst.alignment_check();
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

template <typename Container, typename Dtype>
class ComplexEngine {
 public:
  template <typename Lhs, typename Rhs>
  FORCE_INLINE static void Eval(Container* dst,
                                const MatMul<Lhs, Rhs, Dtype>& exp) {
    ComplexEngineExcutor(dst, exp);
  }
};

template <typename Op, typename Container, typename Dtype>
class ExpEngine {
 public:
  template <typename ExpType>
  FORCE_INLINE static void Eval(
      Container* dst, const ExpBase<ExpType, Dtype, type::container>& exp) {
    dst->set_ptr(const_cast<ExpType*>(&exp.derived_to())->m_storage,
                 const_cast<ExpType*>(&exp.derived_to())->shape);
  }

  template <typename ExpType>
  FORCE_INLINE static void Eval(
      Container* dst, const ExpBase<ExpType, Dtype, type::keepDim>& exp) {
    ExpEngineExcutor<Op>(dst, exp);
  }

  template <typename ExpType>
  FORCE_INLINE static void Eval(
      Container* dst, const ExpBase<ExpType, Dtype, type::complex>& exp) {
    ExpEngineExcutor<Op>(dst, exp);
  }

  template <typename Lhs, typename Rhs>
  FORCE_INLINE static void Eval(Container* dst,
                                const MatMul<Lhs, Rhs, Dtype>& exp) {
    ComplexEngine<Container, Dtype>::template Eval<Lhs, Rhs>(dst, exp);
  }
};
}  // namespace Elastic