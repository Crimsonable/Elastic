#pragma once
#include "ExpBase.h"
#include "MetaTools.h"
#include "base.h"

namespace Elastic {
enum Distribution { Uniform, Normal };

template <int Method, typename T>
struct RandEngine;

template <typename T>
struct RandEngine<Distribution::Uniform, T> {
  std::random_device rd;
  std::mt19937 gen = std::mt19937(rd());
  std::uniform_real_distribution<T> dis;
  RandEngine(T s, T e) { dis = std::uniform_real_distribution<T>(s, e); }
  RandEngine() {}

  FORCE_INLINE void set(T s, T e) {
    dis = std::uniform_real_distribution<T>(s, e);
  }

  template <typename Dtype, ENABLE_IF(std::is_arithmetic_v<Dtype>)>
  FORCE_INLINE Dtype Eval(index x, index y) {
    return dis(gen);
  }

  template <typename Dtype, ENABLE_IF(!std::is_arithmetic_v<Dtype>)>
  FORCE_INLINE Dtype Eval(index x, index y) {
    typename Packet::PacketHandle<T>::type res;
    for (unsigned int i = 0;
         i < sizeof(typename Packet::PacketHandle<T>::type) / sizeof(T); ++i) {
      *((T*)(&res) + i) = dis(gen);
    }
    return res;
  }
};

template <int Method, typename Dtype, typename type::device Device>
class InitExp
    : public ExpBase<InitExp<Method, Dtype, Device>, Dtype, type::keepDim> {
 public:
  RandEngine<Method, Dtype> engine;

  InitExp(Dtype s = 0.0, Dtype e = 1.0) { engine.set(s, e); }

  template <typename T, typename Container = void>
  FORCE_INLINE T Eval(index x, index y, Container* dst = nullptr) {
    return engine.template Eval<T>(x, y);
  }
};

template <int Method, typename T, typename type::device Device>
inline InitExp<Method, T, Device> rand_init(T s, T e) {
  return InitExp<Method, T, Device>(s, e);
}

template <int Method, typename T, typename type::device Device>
struct ExpTraits<InitExp<Method, T, Device>> {
  static constexpr type::device dev = type::device::None;
  static constexpr bool exp = true;
};
}  // namespace Elastic