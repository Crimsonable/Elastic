#pragma once
#include "Exp_engine.h"
#include "Shape.h"
#include "memory.h"
#include "stream.h"

namespace Elastic {

template <typename T, int Dim, typename type::device Device>
class Tensor : public ContainerWarpper<Tensor<T, Dim, Device>, T> {
 public:
  T* m_storage = nullptr;
  static constexpr type::device device = Device;
  index ld = 1, _size;
  Shape<Dim> shape;
  bool hasAlloc = false;
  Stream<Device>* stream = nullptr;

  Tensor() {}

  Tensor(Shape<Dim> _shape) : shape(_shape), _size(_shape.size()) {
    ld = shape.dimx();
  }

  Tensor(Shape<Dim> _shape, T* data) : shape(_shape), m_storage(data) {
    _size = shape.size();
    ld = shape.dimx();
  }

  inline void set_ptr(T* ptr, Shape<Dim> _shape) {
    m_storage = ptr;
    shape = _shape;
    _size = shape.size();
    ld = shape.dimx();
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

  inline ELASTIC_CALL T* data() { return m_storage; }

  inline ELASTIC_CALL T coeff(index idx) const { return *(m_storage + idx); }

  inline ELASTIC_CALL T coeff(index r, index c) const {
    return coeff(r + c * this->ld);
  }

  inline ELASTIC_CALL T& coeffRef(index idx) { return *(m_storage + idx); }

  inline ELASTIC_CALL T& coeffRef(index r, index c) {
    return coeffRef(r + c * this->ld);
  }

  void printMatrix() {
    for (index i = 0; i < shape.dimx(); ++i) {
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