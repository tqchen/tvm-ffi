/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file tvm/ffi/container/dict.h
 * \brief Mutable dictionary container.
 *
 * tvm::ffi::Dict<K, V> is a mutable hash map that shares its
 * underlying storage (no copy-on-write).
 */
#ifndef TVM_FFI_CONTAINER_DICT_H_
#define TVM_FFI_CONTAINER_DICT_H_

#include <tvm/ffi/container/map.h>
#include <tvm/ffi/container/map_base.h>

#include <sstream>
#include <unordered_map>

namespace tvm {
namespace ffi {

/*!
 * \brief DictObj is the FFI type registration for Dict with type index kTVMFFIDict.
 *
 * DictObj inherits from MapBaseObj and registers kTVMFFIDict as a
 * child type of kTVMFFIMap. The actual runtime containers are
 * SmallMapObj and DenseMapObj (inheriting from MapBaseObj) with
 * their type_index overridden to kTVMFFIDict at creation time.
 */
class DictObj : public MapBaseObj {
 public:
  /// \cond Doxygen_Suppress
  static constexpr const int32_t _type_index = TypeIndex::kTVMFFIDict;
  static const constexpr bool _type_final = true;
  TVM_FFI_DECLARE_OBJECT_INFO_STATIC(StaticTypeKey::kTVMFFIDict, DictObj, MapBaseObj);
  /// \endcond
};

/*!
 * \brief Dict container - a mutable hash map.
 *
 * Unlike Map, Dict is mutable and does not implement copy-on-write.
 * Mutations happen directly on the underlying shared DictObj.
 *
 * \note Thread safety: Dict is **not** thread-safe. Concurrent reads and writes
 *       from multiple threads require external synchronization.
 *
 * \tparam K The key type, must be compatible with tvm::ffi::Any
 * \tparam V The value type, must be compatible with tvm::ffi::Any
 */
template <typename K, typename V,
          typename = typename std::enable_if_t<details::storage_enabled_v<K> &&
                                               details::storage_enabled_v<V>>>
class Dict : public ObjectRef {
 public:
  /*! \brief The key type of the dict */
  using key_type = K;
  /*! \brief The mapped type of the dict */
  using mapped_type = V;
  /*! \brief The iterator type of the dict */
  class iterator;

  /*! \brief Construct a Dict with UnsafeInit */
  explicit Dict(UnsafeInit tag) : ObjectRef(tag) {}
  /*! \brief default constructor */
  Dict() { data_ = MapBaseObj::Empty(TypeIndex::kTVMFFIDict); }
  /*! \brief Move constructor */
  Dict(Dict<K, V>&& other)  // NOLINT(google-explicit-constructor)
      : ObjectRef(std::move(other.data_)) {}
  /*! \brief Copy constructor */
  Dict(const Dict<K, V>& other)  // NOLINT(google-explicit-constructor)
      : ObjectRef(other.data_) {}

  /*!
   * \brief Move constructor from another dict
   * \tparam KU The key type of the other dict
   * \tparam VU The mapped type of the other dict
   */
  template <typename KU, typename VU,
            typename = std::enable_if_t<details::type_contains_v<K, KU> &&
                                        details::type_contains_v<V, VU>>>
  Dict(Dict<KU, VU>&& other)  // NOLINT(google-explicit-constructor)
      : ObjectRef(std::move(other.data_)) {}

  /*!
   * \brief Copy constructor from another dict
   * \tparam KU The key type of the other dict
   * \tparam VU The mapped type of the other dict
   */
  template <typename KU, typename VU,
            typename = std::enable_if_t<details::type_contains_v<K, KU> &&
                                        details::type_contains_v<V, VU>>>
  Dict(const Dict<KU, VU>& other)  // NOLINT(google-explicit-constructor)
      : ObjectRef(other.data_) {}

  /*!
   * \brief Move assignment
   * \param other The other dict
   */
  Dict<K, V>& operator=(Dict<K, V>&& other) {
    data_ = std::move(other.data_);
    return *this;
  }

  /*!
   * \brief Copy assignment
   * \param other The other dict
   */
  Dict<K, V>& operator=(const Dict<K, V>& other) {
    data_ = other.data_;
    return *this;
  }

  /*!
   * \brief Move assignment from another dict
   * \tparam KU The key type of the other dict
   * \tparam VU The mapped type of the other dict
   */
  template <typename KU, typename VU,
            typename = std::enable_if_t<details::type_contains_v<K, KU> &&
                                        details::type_contains_v<V, VU>>>
  Dict<K, V>& operator=(Dict<KU, VU>&& other) {
    data_ = std::move(other.data_);
    return *this;
  }

  /*!
   * \brief Copy assignment from another dict
   * \tparam KU The key type of the other dict
   * \tparam VU The mapped type of the other dict
   */
  template <typename KU, typename VU,
            typename = std::enable_if_t<details::type_contains_v<K, KU> &&
                                        details::type_contains_v<V, VU>>>
  Dict<K, V>& operator=(const Dict<KU, VU>& other) {
    data_ = other.data_;
    return *this;
  }

  /*! \brief Constructor from pointer */
  explicit Dict(ObjectPtr<Object> n) : ObjectRef(std::move(n)) {}

  /*!
   * \brief constructor from iterator
   * \param first begin of iterator
   * \param last end of iterator
   * \tparam IterType The type of iterator
   */
  template <typename IterType>
  Dict(IterType first, IterType last) {
    data_ = MapBaseObj::CreateFromRange(first, last, TypeIndex::kTVMFFIDict);
  }

  /*!
   * \brief constructor from initializer list
   * \param init The initializer list
   */
  Dict(std::initializer_list<std::pair<K, V>> init) {
    data_ = MapBaseObj::CreateFromRange(init.begin(), init.end(), TypeIndex::kTVMFFIDict);
  }

  /*!
   * \brief constructor from unordered_map
   * \param init The unordered_map
   */
  template <typename Hash, typename Equal>
  Dict(const std::unordered_map<K, V, Hash, Equal>& init) {  // NOLINT(*)
    data_ = MapBaseObj::CreateFromRange(init.begin(), init.end(), TypeIndex::kTVMFFIDict);
  }

  /*!
   * \brief Read element from dict.
   * \param key The key
   * \return the corresponding element.
   */
  const V at(const K& key) const {
    return details::AnyUnsafe::CopyFromAnyViewAfterCheck<V>(GetDictObj()->at(key));
  }
  /*!
   * \brief Read element from dict.
   * \param key The key
   * \return the corresponding element.
   */
  const V operator[](const K& key) const { return this->at(key); }
  /*! \return The size of the dict */
  size_t size() const {
    MapBaseObj* n = GetDictObj();
    return n == nullptr ? 0 : n->size();
  }
  /*! \return The number of elements of the key */
  size_t count(const K& key) const {
    MapBaseObj* n = GetDictObj();
    return n == nullptr ? 0 : GetDictObj()->count(key);
  }
  /*! \return whether dict is empty */
  bool empty() const { return size() == 0; }
  /*! \brief Release reference to all the elements */
  void clear() {
    MapBaseObj* n = GetDictObj();
    if (n != nullptr) {
      data_ = MapBaseObj::Empty(TypeIndex::kTVMFFIDict);
    }
  }
  /*!
   * \brief Set an entry in the Dict (mutates in place).
   * \param key The index key.
   * \param value The value to be set.
   */
  void Set(const K& key, const V& value) {
    EnsureDictObj();
    MapBaseObj::InsertMaybeReHash(MapBaseObj::KVType(key, value), &data_);
  }
  /*! \return begin iterator */
  iterator begin() const { return iterator(GetDictObj()->begin()); }
  /*! \return end iterator */
  iterator end() const { return iterator(GetDictObj()->end()); }
  /*! \return find the key and returns the associated iterator */
  iterator find(const K& key) const { return iterator(GetDictObj()->find(key)); }
  /*! \return The value associated with the key, std::nullopt if not found */
  std::optional<V> Get(const K& key) const {
    MapBaseObj::iterator iter = GetDictObj()->find(key);
    if (iter == GetDictObj()->end()) {
      return std::nullopt;
    }
    return details::AnyUnsafe::CopyFromAnyViewAfterCheck<V>(iter->second);
  }

  /*!
   * \brief Erase the entry associated with the key (mutates in place).
   * \param key The key
   */
  void erase(const K& key) { EnsureDictObj()->erase(key); }

  /*! \brief specify container node */
  using ContainerType = DictObj;

  /// \cond Doxygen_Suppress
  /*! \brief Iterator of the hash map */
  class iterator {
   public:
    using iterator_category = std::bidirectional_iterator_tag;
    using difference_type = int64_t;
    using value_type = const std::pair<K, V>;
    using pointer = value_type*;
    using reference = value_type;

    iterator() : itr() {}

    /*! \brief Compare iterators */
    bool operator==(const iterator& other) const { return itr == other.itr; }
    /*! \brief Compare iterators */
    bool operator!=(const iterator& other) const { return itr != other.itr; }
    /*! \brief De-reference iterators is not allowed */
    pointer operator->() const = delete;
    /*! \brief De-reference iterators */
    reference operator*() const {
      auto& kv = *itr;
      return std::make_pair(details::AnyUnsafe::CopyFromAnyViewAfterCheck<K>(kv.first),
                            details::AnyUnsafe::CopyFromAnyViewAfterCheck<V>(kv.second));
    }
    /*! \brief Prefix self increment, e.g. ++iter */
    iterator& operator++() {
      ++itr;
      return *this;
    }
    /*! \brief Suffix self increment */
    iterator operator++(int) {
      iterator copy = *this;
      ++(*this);
      return copy;
    }

    /*! \brief Prefix self decrement, e.g. --iter */
    iterator& operator--() {
      --itr;
      return *this;
    }
    /*! \brief Suffix self decrement */
    iterator operator--(int) {
      iterator copy = *this;
      --(*this);
      return copy;
    }

   private:
    iterator(const MapBaseObj::iterator& itr)  // NOLINT(*)
        : itr(itr) {}

    template <typename, typename, typename>
    friend class Dict;

    MapBaseObj::iterator itr;
  };
  /// \endcond

 private:
  /*! \brief Return data_ as type of pointer of MapBaseObj */
  MapBaseObj* GetDictObj() const { return static_cast<MapBaseObj*>(data_.get()); }

  /*!
   * \brief Ensure this dict has a container object.
   * \return MapBaseObj pointer
   */
  MapBaseObj* EnsureDictObj() {
    if (data_ == nullptr) {
      data_ = MapBaseObj::Empty(TypeIndex::kTVMFFIDict);
    }
    return GetDictObj();
  }

  template <typename, typename, typename>
  friend class Dict;
};

// Traits for Dict
template <typename K, typename V>
inline constexpr bool use_default_type_traits_v<Dict<K, V>> = false;

template <typename K, typename V>
struct TypeTraits<Dict<K, V>>
    : public MapTypeTraitsBase<TypeTraits<Dict<K, V>>, Dict<K, V>, K, V> {
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIDict;
  static constexpr int32_t kPrimaryTypeIndex = TypeIndex::kTVMFFIDict;
  static constexpr int32_t kOtherTypeIndex = TypeIndex::kTVMFFIMap;
  static constexpr const char* kTypeName = "Dict";
  static constexpr const char* kStaticTypeKey = StaticTypeKey::kTVMFFIDict;

  TVM_FFI_INLINE static std::string TypeSchema() {
    std::ostringstream oss;
    oss << R"({"type":")" << kStaticTypeKey << R"(","args":[)";
    oss << details::TypeSchema<K>::v() << ",";
    oss << details::TypeSchema<V>::v();
    oss << "]}";
    return oss.str();
  }
};

namespace details {
template <typename K, typename V, typename KU, typename VU>
inline constexpr bool type_contains_v<Dict<K, V>, Dict<KU, VU>> =
    type_contains_v<K, KU> && type_contains_v<V, VU>;
}  // namespace details

}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_CONTAINER_DICT_H_
