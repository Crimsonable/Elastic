#include <iostream>
#include "../../src/Elastic.h"
using namespace std;
using namespace Elastic;

int main() {
  Shape<2> shape;
  shape[0] = 10;
  shape[1] = 10;
  //auto s = NewStream<type::device::gpu>(true);
  Tensor<float, 2, type::device::gpu> a(shape), b(shape), c(shape);
  //a.stream = s;
  AllocSpace(&a);
  AllocSpace(&b);
  AllocSpace(&c);
  //a = rand_init<Distribution::Uniform>(0.0f, 1.0f);
  a = b + c;
  //a.printMatrix();
  system("pause");
  return 1;
}
