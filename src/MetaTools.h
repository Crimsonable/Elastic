#pragma once
#include <tuple>

#define IS_SAME(T1, T2) std::is_same_v<T1, T2>
#define IS_BASE_OF(Base, Type) \
  std::is_base_of_v<Base, std::remove_reference_t<Type>>
#define ENABLE_IF(Condition) typename std::enable_if<Condition>::type* = nullptr

namespace CSM {
template <typename Types>
struct Typelist;
template <template <typename... Types> class ClassType, typename... Types>
struct Typelist<ClassType<Types...>> {
  template <std::size_t I>
  using getType = typename std::tuple_element<I, std::tuple<Types...>>::type;
  using Type = ClassType<Types...>;
};

template <bool Condition, typename T1, typename T2>
struct TypeCondition;
template <typename T1, typename T2>
struct TypeCondition<true, T1, T2> {
  using type = T1;
};
template <typename T1, typename T2>
struct TypeCondition<false, T1, T2> {
  using type = T2;
};
}  // namespace CSM