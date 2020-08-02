#pragma once
#include "base.h"

namespace Elastic {
template <int Dim>
struct Shape {
  index shape[Dim] = {0};
  ELASTIC_CALL inline index& operator[](size_t i) { return shape[i]; }
  ELASTIC_CALL inline index operator[](size_t i) const { return shape[i]; }
  ELASTIC_CALL inline index last() const { return shape[Dim - 1]; }
  ELASTIC_CALL inline index first() const { return shape[0]; }
  ELASTIC_CALL inline index dimx() const { return shape[Dim - 2]; }
  ELASTIC_CALL inline index inner_size() const {
    return shape[Dim - 1] * shape[Dim - 2];
  }

  ELASTIC_CALL FORCE_INLINE Shape<2> Flat2d() const {
    Shape<2> temp;
    temp[0] = this->dimx();
    temp[1] = this->size() / temp[0];
    return temp;
  }

  ELASTIC_CALL inline index size() const {
    index size = 1;
    for (int i = 0; i < Dim; ++i) size *= shape[i];
    return size;
  }

  template <int dim>
  inline bool operator==(Shape<dim>& _shape) const {
    if (dim != Dim) return false;
    //if constexpr (dim != Dim) return false;
    for (index i = 0; i < Dim; ++i)
      if (shape[i] != _shape[i]) return false;
    return true;
  }

  inline bool operator>=(Shape<Dim>& _shape) const {
    return this->size() >= _shape.size();
  }
};

template <>
struct Shape<0> {};

template <typename shape>
struct getShapeDim;

template <int Dim>
struct getShapeDim<Shape<Dim>> {
  static constexpr int dim = Dim;
};

}  // namespace Elastic