#include <iostream>

#include "Elastic.h"
using namespace std;
using namespace Elastic;

int main() {
  Shape<2> shape;
  shape[0] = 8;
  shape[1] = 8;
  float data[70];
  for (int i = 0; i < 64; ++i) {
    data[i] = 1;
  }
  Tensor<float, 2, type::device::cpu> a(shape), b(shape, data), c(shape, data);
  a.alloc();
  a = dot(b,c)+b;
  a.printMatrix();
  system("pause");
  return 1;
}
