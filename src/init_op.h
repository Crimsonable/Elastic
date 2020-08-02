#pragma once
#include "base.h"
#include "rand.h"
#include "ExpBase.h"

namespace Elastic {
template <int Method, typename Dtype, typename type::device Device>
class InitExp
    : public ExpBase<InitExp<Method, Dtype, Device>, Dtype, type::keepDim> {
 public:
  RandEngine<Method, Dtype, Device> engine;

  InitExp(Dtype s, Dtype e) { engine.set(s, e); }

  template <typename T, typename Container = void>
  ELASTIC_CALL FORCE_INLINE T Eval(index x, index y, Container* dst = nullptr) {
    return engine.template Eval<T>(x, y);
  }
};

template <int Method, typename T, typename type::device Device>
inline InitExp<Method, T, Device> rand_init(T s, T e) {
  return InitExp<Method, T, Device>(s, e);
}

template <int Method, typename T, typename type::device Device>
struct ExpTraits<InitExp<Method, T, Device>> {
  static constexpr type::device dev = Device;
  static constexpr bool exp = true;
};
}  // namespace Elastic