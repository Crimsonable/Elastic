#pragma once
#include "Exp_engine.h"
#include "Shape.h"
#include "memory.h"

namespace Elastic {

template <typename T, int Dim, typename type::device Device>
class Tensor : public ContainerWarpper<Tensor<T, Dim, Device>, T> {
 public:
  T* m_storage = nullptr;
  type::device dev = Device;
  index ld = 1, _size;
  Shape<Dim> shape;
  bool hasAlloc = false;

  Tensor() {}

  Tensor(Shape<Dim> _shape) : shape(_shape), _size(_shape.size()) {
    ld = _size / shape.last();
  }

  Tensor(Shape<Dim> _shape, T* data) : shape(_shape), m_storage(data) {
    _size = shape.size();
    ld = _size / shape.last();
  }

  ~Tensor() {
    if (hasAlloc) aligned_free(m_storage);
  }

  inline void set_ptr(T* ptr, Shape<Dim> _shape) {
    if (hasAlloc) {
      hasAlloc = false;
      aligned_free(m_storage);
    }
    m_storage = ptr;
    shape = _shape;
    _size = shape.size();
    ld = _size / shape.last();
  }

  inline Tensor<T, 2, Device> flat2D() {
    Shape<2> _temp;
    _temp[0] = ld;
    _temp[1] = shape.last();
    return Tensor<T, 2, Device>(_temp, m_storage);
  }

  inline Shape<Dim> getShape() { return this->shape; }

  inline Tensor<T, Dim, Device>& operator=(const Tensor<T, Dim, Device>& exp) {
    return this->assign(exp);
  }

  template <typename Exp, int exp_type>
  inline Tensor<T, Dim, Device>& operator=(
      const ExpBase<Exp, T, exp_type>& exp) {
    return this->assign(exp);
  }

  template <int dim>
  Tensor<T, dim, Device> resize(Shape<dim> _shape) {
    index newsize = _shape.size();
    if (newsize != _size) {
      std::cout << "Size of the new shape doesn't fit the original size"
                << std::endl;
      abort();
    }
    return Tensor<T, dim, Device>(_shape, m_storage);
  }

  Tensor<T, 2, Device> Flat2D(Shape<2> shape) { return resize(shape); }

  inline void alloc() {
    m_storage = mynew_fill0<T>(_size + 16, VECTORIZATION_ALIGN_BYTES);
    hasAlloc = true;
  }

  inline T* data() { return m_storage; }

  inline T coeff(index idx) const { return *(m_storage + idx); }

  inline T coeff(index r, index c) const { return coeff(r + c * this->ld); }

  inline T& coeffRef(index idx) { return *(m_storage + idx); }

  inline T& coeffRef(index r, index c) { return coeffRef(r + c * this->ld); }

  void printMatrix() {
    for (index i = 0; i < ld; ++i) {
      for (index j = 0; j < shape.last(); ++j)
        std::cout << m_storage[j * ld + i] << " ";
      std::cout << std::endl;
    }
  }
};

template <typename T, int Dim, typename type::device Device>
struct ExpTraits<Tensor<T, Dim, Device>> {
  static constexpr int dim = Dim;
  static constexpr type::device dev = Device;
  static constexpr bool exp = false;
};

}  // namespace Elastic