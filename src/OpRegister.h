#pragma once
#include "MetaTools.h"
#include "Packet.h"
#include "base.h"
#include "Packet.h"

using Packet::PacketHandle;

#ifdef _MSC_VER
using Packet::operator-, Packet::operator*, Packet::operator+;
#endif

    namespace OP {
struct NoneOp {
  template <typename Dtype>
  FORCE_INLINE static void apply(Dtype& dst, Dtype& val) {}
  using OpType = NoneOp;
};

struct assign {
  template <typename Dtype>
  FORCE_INLINE static void apply(Dtype& dst, const Dtype& src) {
    dst = src;
  }
  template <typename T, typename Dtype>
  FORCE_INLINE static void apply(T& dst, const Dtype& src) {
    Packet::PacketHandle<T>::store(&dst, src);
  }

  using OpType = assign;
};

struct plusto {
  template <typename T>
  FORCE_INLINE static void apply(T& dst, const T& src) {
    dst += src;
  }

  template <typename T, typename Dtype>
  FORCE_INLINE static void apply(T& dst, const Dtype& src) {
    PacketHandle<T>::store(&dst, PacketHandle<T>::load(&dst) + src);
  }
  using OpType = plusto;
};

struct minusto {
  template <typename T>
  FORCE_INLINE static void apply(T& dst, const T& src) {
    dst -= src;
  }

  template <typename T, typename Dtype>
  FORCE_INLINE static void apply(T& dst, const Dtype& src) {
    PacketHandle<T>::store(&dst, PacketHandle<T>::load(&dst) - src);
  }
  using OpType = minusto;
};

struct multo {
  template <typename T>
  FORCE_INLINE static void apply(T& dst, const T& src) {
    dst *= src;
  }

  template <typename T, typename Dtype>
  FORCE_INLINE static void apply(T& dst, const Dtype& src) {
    PacketHandle<T>::store(&dst, PacketHandle<T>::load(&dst) * src);
  }
  using OpType = multo;
};

struct plus {
  template <typename Dtype>
  FORCE_INLINE static Dtype apply(const Dtype& a, const Dtype& b) {
    return a + b;
  }
  using OpType = plus;
};

struct minus {
  template <typename Dtype>
  FORCE_INLINE static Dtype apply(const Dtype& a, const Dtype& b) {
    return a - b;
  }
  using OpType = minus;
};

struct mul {
  template <typename Dtype>
  FORCE_INLINE static Dtype apply(const Dtype& a, const Dtype& b) {
    return a * b;
  }
  using OpType = mul;
};
}  // namespace OP