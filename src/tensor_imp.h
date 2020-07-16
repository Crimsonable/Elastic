#include "Tensor.h"
#include "base.h"

namespace Elastic {
template <typename T, int Dim>
FORCE_INLINE void AllocSpace(Tensor<T, Dim, type::device::cpu>* dst,
                             bool pad = false) {
  index real_size = dst->_size;
  index ld = dst->ld;
  if (pad) {
    index dimx = dst->shape.dimx();
    Shape<Dim> _temp = dst->shape;
    dimx += dimx % VECTORIZATION_ALIGN_BYTES;
    _temp[Dim - 2] = dimx;
    ld = dimx;
    real_size = _temp.size();
  }
  dst->m_storage = mynew_fill0<T>(real_size, VECTORIZATION_ALIGN_BYTES);
  dst->hasAlloc = true;
}
}  // namespace Elastic