#pragma once
#include <curand.h>
#include "MetaTools.h"
#include "base.h"
#include "Packet.h"

namespace Elastic {
enum Distribution { Uniform, Normal };

template <int Method, typename T, typename type::device Device>
struct RandEngine;

template <typename T>
struct RandEngine<Distribution::Uniform, T, type::device::cpu> {
  std::mt19937 gen;
  std::uniform_real_distribution<T> dis;

  RandEngine() {}

  FORCE_INLINE void set(T s, T e) {
    std::random_device rd;
    gen = std::mt19937(rd());
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

/*template<typename T>
struct RandEngine<Distribution::Uniform, T, type::device::cpu> {
  curandGenerator_t gen;


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
};*/
}  // namespace Elastic