#include <iostream>
#include "Elastic.h"
using namespace std;
using namespace Elastic;

int main() {
  Shape<2> shape;
  shape[0] = 10;
  shape[1] = 10;
  Tensor<float, 2, type::device::cpu> a(shape), b(shape), c(shape);
  a.alloc();
  b.alloc();
  c.alloc();
  a = rand_init<Distribution::Uniform>(0.0f, 1.0f);
  a.printMatrix();
  system("pause");
  return 1;
}
