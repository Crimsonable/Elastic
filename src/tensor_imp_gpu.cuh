#pragma once
#include "Shape.h"
#include "base.h"

namespace Elastic {
template <typename ExpType>
struct ImpExp;

template <typename type::device Device>
struct Stream;

namespace cuda {
const int BaseThreadNum = 256;
const int BaseThreadNum_bit = 8;
const int MaxGridNum = 65535;

template <typename Saver, int block_dim_bits, typename DstImp, typename ExpImp,
          typename DataType>
__device__ void BasicKernelEval(DstImp dst, const index ld, Shape<2> shape,
                                const ExpImp& exp, int block_idx) {
  const index tid = (block_idx << block_dim_bits) + threadIdx.x;
  const index y = tid / ld;
  const index x = tid % ld;
  if (y < shape[0] && x < shape[1]) {
    Saver::save(dst.template Eval<DataType>(y, x),
                exp.template Eval<DataType>(y, x));
  }
}

template <typename Saver, int block_dim_bits, typename DstImp, typename ExpImp,
          typename DataType>
__global__ void KernelEval(DstImp dst, const index ld, Shape<2> shape,
                           const ExpImp exp) {
  BasicKernelEval<Saver, block_dim_bits, DstImp, ExpImp, DataType>(
      dst, ld, shape, exp, blockIdx.x);
}

template <typename Saver, typename DstImp, typename ExpImp, typename DataType>
inline void KernelEntry(ImpExp<DstImp> dst, const ImpExp<ExpImp> exp,
                        cudaStream_t stream, Shape<2> shape, const index ld) {
  const int num_blocks = (ld * shape[1] + BaseThreadNum - 1) / BaseThreadNum;
  dim3 dimBlock(BaseThreadNum, 1, 1);


  if (num_blocks < MaxGridNum) {
    dim3 dimGrid(num_blocks, 1, 1);
    KernelEval<Saver, BaseThreadNum_bit, decltype(dst), decltype(exp), DataType>
        <<<dimGrid, dimBlock, 0, stream>>>(dst, ld, shape, exp);
  }
}

}  // namespace cuda
}  // namespace Elastic