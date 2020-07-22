#pragma once
#include "OpRegister.h"
#include "Shape.h"
#include "base.h"

namespace Elastic {

template <typename T>
struct ExpTraits;

template <typename Op, typename Container, typename Dtype>
class ExpEngine;

template <typename Derived, typename Dtype, int exp_type>
class ExpBase {
 public:
  Derived* derived() { return static_cast<Derived*>(this); }
  inline const Derived& derived_to() const {
    return *static_cast<const Derived*>(this);
  }
};

template <typename Dtype>
class Scalar : public ExpBase<Scalar<Dtype>, Dtype, type::keepDim> {
 public:
  Dtype data;
  Shape<0> shape;

  Scalar(Dtype src) : data(src) {}
};

template <typename Container, typename Dtype>
class ContainerWarpper : public ExpBase<Container, Dtype, type::container> {
 public:
  FORCE_INLINE Container& operator+=(Dtype scalar) {
    ExpEngine<OP::plusto, Container, Dtype>::Eval(this->derived(),
                                                  Scalar<Dtype>(scalar));
    return *(this->derived());
  }

  FORCE_INLINE Container& operator-=(Dtype scalar) {
    ExpEngine<OP::minusto, Container, Dtype>::Eval(this->derived(),
                                                   Scalar<Dtype>(scalar));
    return *(this->derived());
  }

  FORCE_INLINE Container& operator*=(Dtype scalar) {
    ExpEngine<OP::multo, Container, Dtype>::Eval(this->derived(),
                                                 Scalar<Dtype>(scalar));
    return *(this->derived());
  }

  template <typename Rhs, int exp_type>
  FORCE_INLINE Container& assign(const ExpBase<Rhs, Dtype, exp_type>& exp) {
    ExpEngine<OP::assign, Container, Dtype>::Eval(this->derived(),
                                                  exp.derived_to());
    return *(this->derived());
  }

  template <typename Rhs, int exp_type>
  FORCE_INLINE Container& operator+=(const ExpBase<Rhs, Dtype, exp_type>& exp) {
    ExpEngine<OP::plusto, Container, Dtype>::Eval(this->derived(),
                                                  exp.derived_to());
    return *(this->derived());
  }

  template <typename Rhs, int exp_type>
  FORCE_INLINE Container& operator*=(const ExpBase<Rhs, Dtype, exp_type>& exp) {
    ExpEngine<OP::multo, Container, Dtype>::Eval(this->derived(),
                                                 exp.derived_to());
    return *(this->derived());
  }

  template <typename Rhs, int exp_type>
  FORCE_INLINE Container& operator-=(const ExpBase<Rhs, Dtype, exp_type>& exp) {
    ExpEngine<OP::minusto, Container, Dtype>::Eval(this->derived(),
                                                   exp.derived_to());
    return *(this->derived());
  }
};

template <typename Op, typename SelfType, typename Dtype, int exp_type>
class UnaryExp
    : public ExpBase<UnaryExp<Op, SelfType, Dtype, exp_type>, Dtype, exp_type> {
 public:
  const SelfType& _self;
  Shape<getShapeDim<decltype(_self.shape)>::dim> shape;

  UnaryExp(const SelfType& self) : _self(self) { shape = _self.getShape(); }
  FORCE_INLINE auto getShape() { return _self.getShape(); }
};

template <typename Op, typename SelfType, typename Dtype, int exp_type>
inline UnaryExp<Op, SelfType, Dtype, exp_type | type::keepDim> MakeExp(
    const ExpBase<SelfType, Dtype, exp_type>& _self) {
  return UnaryExp<Op, SelfType, Dtype, exp_type | type::keepDim>(
      _self.derived_to());
}

template <typename Op, typename Lhs, typename Rhs, typename Dtype, int exp_type>
class BinaryExp : public ExpBase<BinaryExp<Op, Lhs, Rhs, Dtype, exp_type>,
                                 Dtype, exp_type> {
 public:
  const Lhs& _lhs;
  const Rhs& _rhs;
  Shape<getShapeDim<decltype(_lhs.shape)>::dim
            ? getShapeDim<decltype(_lhs.shape)>::dim
            : getShapeDim<decltype(_rhs.shape)>::dim>
      shape;

  BinaryExp(const Lhs& _l, const Rhs& _r) : _lhs(_l), _rhs(_r) {
    static_assert(getShapeDim<decltype(_lhs.shape)>::dim ==
                      getShapeDim<decltype(_rhs.shape)>::dim,
                  "Dims don't match for BinaryOp");
    static_assert(ExpTraits<Lhs>::dev == ExpTraits<Rhs>::dev, "Op's device does't match");
    shape = getShape();
  }

  FORCE_INLINE auto getShape() {
    auto l_shape = _lhs.shape;
    auto r_shape = _rhs.shape;
    if (l_shape.size() != 0 && r_shape.size() != 0) {
      CHECK_CON(l_shape == r_shape, "Shapes don't match for BinaryOp")
      return l_shape;
    }
    return l_shape.size() == 0 ? r_shape : l_shape;
  }
};

template <typename Op, typename Lhs, typename Rhs, typename Dtype, int tl,
          int tr>
inline BinaryExp<Op, Lhs, Rhs, Dtype, tl | tr | type::keepDim> MakeExp(
    const ExpBase<Lhs, Dtype, tl>& l, const ExpBase<Rhs, Dtype, tr>& r) {
  return BinaryExp<Op, Lhs, Rhs, Dtype, tl | tr | type::keepDim>(
      l.derived_to(), r.derived_to());
}

template <typename Lhs, typename Rhs, typename Dtype, int tl, int tr>
inline BinaryExp<OP::plus, Lhs, Rhs, Dtype, tl | tr | type::keepDim> operator+(
    const ExpBase<Lhs, Dtype, tl>& l, const ExpBase<Rhs, Dtype, tr>& r) {
  return MakeExp<OP::plus>(l.derived_to(), r.derived_to());
}

template <typename Lhs, typename Dtype, int tl>
inline BinaryExp<OP::plus, Lhs, Scalar<Dtype>, Dtype, tl | type::keepDim>
operator+(const ExpBase<Lhs, Dtype, tl>& l, Dtype r) {
  return MakeExp<OP::plus>(l.derived_to(), Scalar<Dtype>(r));
}

template <typename Lhs, typename Rhs, typename Dtype, int tl, int tr>
inline BinaryExp<OP::minus, Lhs, Rhs, Dtype, tl | tr | type::keepDim> operator-(
    const ExpBase<Lhs, Dtype, tl>& l, const ExpBase<Rhs, Dtype, tr>& r) {
  return MakeExp<OP::minus>(l.derived_to(), r.derived_to());
}

template <typename Lhs, typename Dtype, int tl>
inline BinaryExp<OP::minus, Lhs, Scalar<Dtype>, Dtype, tl | type::keepDim>
operator-(const ExpBase<Lhs, Dtype, tl>& l, Dtype r) {
  return MakeExp<OP::minus>(l.derived_to(), Scalar<Dtype>(r));
}

template <typename Lhs, typename Rhs, typename Dtype, int tl, int tr>
inline BinaryExp<OP::mul, Lhs, Rhs, Dtype, tl | tr | type::keepDim> operator*(
    const ExpBase<Lhs, Dtype, tl>& l, const ExpBase<Rhs, Dtype, tr>& r) {
  return MakeExp<OP::mul>(l.derived_to(), r.derived_to());
}

template <typename Lhs, typename Dtype, int tl>
inline BinaryExp<OP::mul, Lhs, Scalar<Dtype>, Dtype, tl | type::keepDim>
operator*(const ExpBase<Lhs, Dtype, tl>& l, Dtype r) {
  return MakeExp<OP::mul>(l.derived_to(), Scalar<Dtype>(r));
}

template <typename Lhs, typename Rhs, typename Dtype>
class MatMul : public ExpBase<MatMul<Lhs, Rhs, Dtype>, Dtype, type::complex> {
 public:
  const Lhs& _lhs;
  const Rhs& _rhs;
  Shape<getShapeDim<decltype(_lhs.shape)>::dim> shape;

  MatMul(const Lhs& l, const Rhs& r) : _lhs(l), _rhs(r) {
    static_assert(getShapeDim<decltype(_lhs.shape)>::dim ==
                      getShapeDim<decltype(_rhs.shape)>::dim,
                  "Dims don't match for MatmulOp");
    static_assert(ExpTraits<Lhs>::dev == ExpTraits<Rhs>::dev, "Op's device does't match");
    shape = getShape();
  }

  FORCE_INLINE auto getShape() {
    auto l_shape = _lhs.shape;
    auto r_shape = _rhs.shape;
    const int dim_l = getShapeDim<decltype(l_shape)>::dim;
    for (int i = 0; i < dim_l - 2; ++i)
      CHECK_CON(l_shape[i] == r_shape[i], "Shapes don't match for MatmulOp")
    CHECK_CON(l_shape.last() == r_shape[dim_l - 2],
              "Shapes don't match for MatmulOp")
    auto dst_shape = l_shape;
    dst_shape[dim_l - 1] = r_shape.last();
    return dst_shape;
  }
};

template <typename Lhs, typename Rhs, typename Dtype, int tl, int tr>
inline MatMul<Lhs, Rhs, Dtype> dot(const ExpBase<Lhs, Dtype, tl>& l,
                                   const ExpBase<Rhs, Dtype, tr>& r) {
  return MatMul<Lhs, Rhs, Dtype>(l.derived_to(), r.derived_to());
}

template <typename Op, typename SelfType, typename Dtype, int exp_type>
struct ExpTraits<UnaryExp<Op, SelfType, Dtype, exp_type>> {
  static constexpr int dim = ExpTraits<SelfType>::dim;
  static constexpr type::device dev = ExpTraits<SelfType>::dev;
  static constexpr bool exp = true;
};

template <typename Lhs, typename Rhs, typename Dtype>
struct ExpTraits<MatMul<Lhs, Rhs, Dtype>> {
  static constexpr int dim = ExpTraits<Rhs>::dim;
  static constexpr type::device dev = ExpTraits<Rhs>::dev;
  static constexpr bool exp = true;
};

template <typename Op, typename Lhs, typename Rhs, typename Dtype, int exp_type>
struct ExpTraits<BinaryExp<Op, Lhs, Rhs, Dtype, exp_type>> {
  static constexpr int dim = ExpTraits<Rhs>::dim;
  static constexpr type::device dev = ExpTraits<Rhs>::dev;
  static constexpr bool exp = true;
};
}  // namespace Elastic