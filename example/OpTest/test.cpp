#include <omp.h>

#include <chrono>
#include <iostream>

#include "../../Elastic.h"

#define M 2048
#define K 2048
#define N 2048
#define numThread 4
//#define _DEBUG

using namespace std;
using namespace Elastic;

void bench(int times) {
#ifndef _DEBUG
  omp_set_num_threads(numThread);
#endif
  Shape<2> a_shape, b_shape, c_shape;
  a_shape[0] = M;
  a_shape[1] = K;
  b_shape[0] = K;
  b_shape[1] = N;
  c_shape[0] = M;
  c_shape[1] = N;
  Tensor<float, 2, type::device::cpu> a(a_shape), b(b_shape), c(c_shape);
  a.alloc();
  b.alloc();
  c.alloc();
  auto t0 = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < times; ++i) {
    a = dot(b, c);
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0);
  std::cout << "M: " << M << " "
            << "K: " << K << " "
            << "N: " << N << std::endl;
  std::cout << "Time cost: " << duration.count() << std::endl;
  std::cout << "Gflops: " << 1e-9 * M * N * K * 2 / duration.count() * 50
            << std::endl;
}

int main() {
  bench(50);
  return 1;
}
