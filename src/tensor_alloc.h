#include "base.h"
#include "memory.h"

namespace Elastic {
template <typename T, int Dim, typename type::device Device>
class Tensor;

template <typename T, int Dim>
FORCE_INLINE void destory(Tensor<T, Dim, type::device::cpu>* dst) {
  if (dst->hasAlloc) aligned_free(dst->m_storage);
  dst->m_storage = nullptr;
}

template <typename T, int Dim>
FORCE_INLINE void AllocSpace(Tensor<T, Dim, type::device::cpu>* dst,
                             bool pad = false) {
  destory(dst);
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

#ifdef ELASTIC_USE_CUDA
template <typename T, int Dim>
FORCE_INLINE void destory(Tensor<T, Dim, type::device::gpu>* dst) {
  if (dst->hasAlloc) cudaFree(dst->m_storage);
  dst->m_storage = nullptr;
}

template <typename T, int Dim>
FORCE_INLINE void AllocSpace(Tensor<T, Dim, type::device::gpu>* dst,
                             bool pad = false) {
  destory(dst);
  if (pad)
    cudaMallocPitch(&dst->m_storage, &dst->ld, sizeof(T) * dst->shape.dimx(),
                    dst->shape.size() / dst->shape.dimx());
  else
    cudaMalloc(&dst->m_storage, sizeof(T) * dst->_size);
  dst->hasAlloc = true;
}
#endif
}  // namespace Elastic