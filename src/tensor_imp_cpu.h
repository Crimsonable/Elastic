#include "Gemm.h"
#include "base.h"

namespace Elastic {
template <typename T, int Dim, typename type::device Device>
class Tensor;

template <typename type::device Device>
struct Stream;

template <>
struct Stream<type::device::cpu> {};

template <typename T, typename type::device Device>
struct BlasEnigen;

template <typename T>
struct BlasEnigen<T, type::device::cpu> {
  inline static setStream(Stream<type::device::cpu>* stream) {}

  inline static void gemm(T* a, const int& lda, T* b, const int& ldb, T* c,
                     const int& ldc, const int& m, const int& n, const int& k) {
    CSM::Gemm(a, lda, b, ldb, c, ldc, m, n, k);
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

}  // namespace Elastic