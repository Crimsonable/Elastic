#pragma once
#include "PlainArray.h"
#include "base.h"
#include "Shape.h"

namespace Elastic {


template <typename T, int Device>
struct DenseStorage;

template <typename T>
struct DenseStorage<T, type::device::cpu> {
  Memory::plain_array<T, -1> m_storage;
  index capability = 0;
  inline void alloc(index newsize) {
    if (newsize > capability) {
      m_storage.free();
      m_storage.alloc(newsize);
      capability = newsize;
    }
  }
 
  inline T* data() { return m_storage.array; }
};

/*template <typename Derived>
class DenseStorageBase {
 public:
  FORCE_INLINE Derived* derived() { return static_cast<Derived*>(this); }
};

template <typename T>
class DenseStorage : public DenseStorageBase<DenseStorage<T>> {
 private:
  using plainType = typename Memory::plain_array<T, -1>;
  using plainType_ptr = typename std::shared_ptr<plainType>;
  plainType_ptr m_storage;

 public:
  DenseStorage() {}
  DenseStorage() { resize(rows, cols); }

  void resize(int _size) {
    if (_size > capacity) {
      m_storage = std::make_shared<plainType>(_size);
      capacity = _size;
    }
    size = _size;
  }

  void resize(int _rows, int _cols) {
    int newSize = _rows * _cols;
    if (newSize > capacity) {
      m_storage = std::make_shared<plainType>(newSize);
      capacity = newSize;
    }
    rows = _rows;
    ld = rows;
    cols = _cols;
    size = newSize;
  }

  template <typename Container>
  void copy(Container* src) {
    resize(src->rows, src->cols);
    memcpy(data(), src->data(), sizeof(T) * rows * cols);
  }

  template <typename Container>
  inline void share(Container* other_storage) {
    if (other_storage->m_storage == m_storage) return;
    m_storage = other_storage->m_storage;
    capacity = other_storage->capacity;
    size = other_storage->size;
    rows = other_storage->rows;
    cols = other_storage->cols;
    ld = rows;
  }

  template <typename Container>
  inline void swap(Container* other_storage) {
    if (other_storage->m_storage == m_storage) return;
    size = other_storage->size;
    rows = other_storage->rows;
    ld = rows;
    cols = other_storage->cols;
    capacity = other_storage->capacity;
    std::swap(m_storage, other_storage->m_storage);
  }

  FORCE_INLINE T* data() { return m_storage.get()->array; }
  inline const T* cdata() const { return m_storage->array; }

  int rows = -1, cols = -1, size = -1, capacity = -1, ld = -1;
};

template <typename T>
class DenseStorageMap : public DenseStorageBase<DenseStorageMap<T>> {
 private:
  T* m_storage;

 public:
  DenseStorageMap() {}
  DenseStorageMap(T* data, int ld, int rows, int cols)
      : m_storage(data), ld(ld), rows(rows), cols(cols), size(rows * cols) {}

  void resize(int _rows, int _cols) {
    if (rows != _rows || cols != _cols) abort();
  }

  template <typename Container>
  void share(Container* otherStorage) {}

  template <typename Container>
  void swap(Container* otherStorage) {}

  FORCE_INLINE T* data() { return m_storage; }
  FORCE_INLINE const T* cdata() { return m_storage; }

  int rows = -1, cols = -1, size = -1, capacity = -1, ld = -1;
};*/
}  // namespace Elastic