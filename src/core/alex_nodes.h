// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/*
 * This file contains code for ALEX nodes. There are two types of nodes in ALEX:
 * - Model nodes (equivalent to internal/inner nodes of a B+ Tree)
 * - Data nodes, sometimes referred to as leaf nodes (equivalent to leaf nodes
 * of a B+ Tree)
 */
 

#pragma once

#include "leafDS/StructOfArrays/SizedInt.hpp"
#include "leafDS/StructOfArrays/soa.hpp"
#include "leafDS/StructOfArrays/aos.hpp"
#include "leafDS/helpers.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <tuple>
#include <type_traits>
#include <immintrin.h>
#include <typeinfo>
#include <set>
#include "alex_base.h"

// Whether we store key and payload arrays separately in data nodes
// By default, we store them separately
#define ALEX_DATA_NODE_SEP_ARRAYS 1

#if ALEX_DATA_NODE_SEP_ARRAYS
#define ALEX_DATA_NODE_KEY_AT(i) key_slots_[i]
#define ALEX_DATA_NODE_PAYLOAD_AT(i) payload_slots_[i]
#else
#define ALEX_DATA_NODE_KEY_AT(i) data_slots_[i].first
#define ALEX_DATA_NODE_PAYLOAD_AT(i) data_slots_[i].second
#endif

// Whether we use lzcnt and tzcnt when manipulating a bitmap (e.g., when finding
// the closest gap).
// If your hardware does not support lzcnt/tzcnt (e.g., your Intel CPU is
// pre-Haswell), set this to 0.
#define ALEX_USE_LZCNT 1

namespace alex {

// A parent class for both types of ALEX nodes
template <class T, class P>
class AlexNode {
 public:
  // Whether this node is a leaf (data) node
  bool is_leaf_ = false;

  // Power of 2 to which the pointer to this node is duplicated in its parent
  // model node
  // For example, if duplication_factor_ is 3, then there are 8 redundant
  // pointers to this node in its parent
  uint8_t duplication_factor_ = 0;

  // Node's level in the RMI. Root node is level 0
  short level_ = 0;

  // Both model nodes and data nodes nodes use models
  LinearModel<T> model_;

  // Could be either the expected or empirical cost, depending on how this field
  // is used
  double cost_ = 0.0;

  AlexNode() = default;
  explicit AlexNode(short level) : level_(level) {}
  AlexNode(short level, bool is_leaf) : is_leaf_(is_leaf), level_(level) {}
  virtual ~AlexNode() = default;

  // The size in bytes of all member variables in this class
  virtual long long node_size() const = 0;
};

template <class T, class P, class Alloc = std::allocator<std::pair<T, P>>>
class AlexModelNode : public AlexNode<T, P> {
 public:
  typedef AlexModelNode<T, P, Alloc> self_type;
  typedef typename Alloc::template rebind<self_type>::other alloc_type;
  typedef typename Alloc::template rebind<AlexNode<T, P>*>::other
      pointer_alloc_type;

  const Alloc& allocator_;

  // Number of logical children. Must be a power of 2
  int num_children_ = 0;

  // Array of pointers to children
  AlexNode<T, P>** children_ = nullptr;

  explicit AlexModelNode(const Alloc& alloc = Alloc())
      : AlexNode<T, P>(0, false), allocator_(alloc) {}

  explicit AlexModelNode(short level, const Alloc& alloc = Alloc())
      : AlexNode<T, P>(level, false), allocator_(alloc) {}

  ~AlexModelNode() {
    if (children_ == nullptr) {
      return;
    }
    pointer_allocator().deallocate(children_, num_children_);
  }

  AlexModelNode(const self_type& other)
      : AlexNode<T, P>(other),
        allocator_(other.allocator_),
        num_children_(other.num_children_) {
    children_ = new (pointer_allocator().allocate(other.num_children_))
        AlexNode<T, P>*[other.num_children_];
    std::copy(other.children_, other.children_ + other.num_children_,
              children_);
  }

  // Given a key, traverses to the child node responsible for that key
  inline AlexNode<T, P>* get_child_node(const T& key) {
    int bucketID = this->model_.predict(key);
    bucketID = std::min<int>(std::max<int>(bucketID, 0), num_children_ - 1);
    return children_[bucketID];
  }

  // Expand by a power of 2 by creating duplicates of all existing child
  // pointers.
  // Input is the base 2 log of the expansion factor, in order to guarantee
  // expanding by a power of 2.
  // Returns the expansion factor.
  int expand(int log2_expansion_factor) {
    assert(log2_expansion_factor >= 0);
    int expansion_factor = 1 << log2_expansion_factor;
    int num_new_children = num_children_ * expansion_factor;
    auto new_children = new (pointer_allocator().allocate(num_new_children))
        AlexNode<T, P>*[num_new_children];
    int cur = 0;
    while (cur < num_children_) {
      AlexNode<T, P>* cur_child = children_[cur];
      int cur_child_repeats = 1 << cur_child->duplication_factor_;
      for (int i = expansion_factor * cur;
           i < expansion_factor * (cur + cur_child_repeats); i++) {
        new_children[i] = cur_child;
      }
      cur_child->duplication_factor_ += log2_expansion_factor;
      cur += cur_child_repeats;
    }
    pointer_allocator().deallocate(children_, num_children_);
    children_ = new_children;
    num_children_ = num_new_children;
    this->model_.expand(expansion_factor);
    return expansion_factor;
  }

  pointer_alloc_type pointer_allocator() {
    return pointer_alloc_type(allocator_);
  }

  long long node_size() const override {
    long long size = sizeof(self_type);
    size += num_children_ * sizeof(AlexNode<T, P>*);  // pointers to children
    return size;
  }

  // Helpful for debugging
  bool validate_structure(bool verbose = false) const {
    if (num_children_ == 0) {
      if (verbose) {
        std::cout << "[Childless node] addr: " << this << ", level "
                  << this->level_ << std::endl;
      }
      return false;
    }
    if (num_children_ == 1) {
      if (verbose) {
        std::cout << "[Single child node] addr: " << this << ", level "
                  << this->level_ << std::endl;
      }
      return false;
    }
    if (std::ceil(std::log2(num_children_)) !=
        std::floor(std::log2(num_children_))) {
      if (verbose) {
        std::cout << "[Num children not a power of 2] num children: "
                  << num_children_ << std::endl;
      }
      return false;
    }

    if (this->model_.a_ == 0) {
      if (verbose) {
        std::cout << "[Model node with zero slope] addr: " << this << ", level "
                  << this->level_ << std::endl;
      }
      return false;
    }

    AlexNode<T, P>* cur_child = children_[0];
    int cur_repeats = 1;
    int i;
    for (i = 1; i < num_children_; i++) {
      if (children_[i] == cur_child) {
        cur_repeats++;
      } else {
        if (cur_repeats != (1 << cur_child->duplication_factor_)) {
          if (verbose) {
            std::cout << "[Incorrect duplication factor] num actual repeats: "
                      << cur_repeats << ", num dup_factor repeats: "
                      << (1 << cur_child->duplication_factor_)
                      << ", parent addr: " << this
                      << ", parent level: " << this->level_
                      << ", parent num children: " << num_children_
                      << ", child addr: " << children_[i - cur_repeats]
                      << ", child pointer indexes: [" << i - cur_repeats << ", "
                      << i << ")" << std::endl;
          }
          return false;
        }
        if (std::ceil(std::log2(cur_repeats)) !=
            std::floor(std::log2(cur_repeats))) {
          if (verbose) {
            std::cout
                << "[Num duplicates not a power of 2] num actual repeats: "
                << cur_repeats << std::endl;
          }
          return false;
        }
        if (i % cur_repeats != 0) {
          if (verbose) {
            std::cout
                << "[Duplicate region incorrectly aligned] num actual repeats: "
                << cur_repeats << ", num dup_factor repeats: "
                << (1 << cur_child->duplication_factor_)
                << ", child pointer indexes: [" << i - cur_repeats << ", " << i
                << ")" << std::endl;
          }
          return false;
        }
        cur_child = children_[i];
        cur_repeats = 1;
      }
    }
    if (cur_repeats != (1 << cur_child->duplication_factor_)) {
      if (verbose) {
        std::cout << "[Incorrect duplication factor] num actual repeats: "
                  << cur_repeats << ", num dup_factor repeats: "
                  << (1 << cur_child->duplication_factor_)
                  << ", parent addr: " << this
                  << ", parent level: " << this->level_
                  << ", parent num children: " << num_children_
                  << ", child addr: " << children_[i - cur_repeats]
                  << ", child pointer indexes: [" << i - cur_repeats << ", "
                  << i << ")" << std::endl;
      }
      return false;
    }
    if (std::ceil(std::log2(cur_repeats)) !=
        std::floor(std::log2(cur_repeats))) {
      if (verbose) {
        std::cout << "[Num duplicates not a power of 2] num actual repeats: "
                  << cur_repeats << std::endl;
      }
      return false;
    }
    if (i % cur_repeats != 0) {
      if (verbose) {
        std::cout
            << "[Duplicate region incorrectly aligned] num actual repeats: "
            << cur_repeats << ", num dup_factor repeats: "
            << (1 << cur_child->duplication_factor_)
            << ", child pointer indexes: [" << i - cur_repeats << ", " << i
            << ")" << std::endl;
      }
      return false;
    }
    if (cur_repeats == num_children_) {
      if (verbose) {
        std::cout << "[All children are the same] num actual repeats: "
                  << cur_repeats << ", parent addr: " << this
                  << ", parent level: " << this->level_
                  << ", parent num children: " << num_children_ << std::endl;
      }
      return false;
    }

    return true;
  }
};

/*
* Functions are organized into different sections:
* - Constructors and destructors
* - General helper functions
* - Iterator
* - Cost model
* - Bulk loading and model building (e.g., bulk_load, bulk_load_from_existing)
* - Lookups (e.g., find_key, find_lower, find_upper, lower_bound, upper_bound)
* - Inserts and resizes (e.g, insert)
* - Deletes (e.g., erase, erase_one)
* - Stats
* - Debugging
*/
template <class T, class P, class Compare = AlexCompare,
          class Alloc = std::allocator<std::pair<T, P>>,
          bool allow_duplicates = true>
class AlexDataNode : public AlexNode<T, P> {
 public:
  typedef std::pair<T, P> V;
  typedef AlexDataNode<T, P, Compare, Alloc, allow_duplicates> self_type;
  typedef typename Alloc::template rebind<self_type>::other alloc_type;
  typedef typename Alloc::template rebind<T>::other key_alloc_type;
  typedef typename Alloc::template rebind<P>::other payload_alloc_type;
  typedef typename Alloc::template rebind<V>::other value_alloc_type;
  typedef typename Alloc::template rebind<uint64_t>::other bitmap_alloc_type;

  const Compare& key_less_;
  const Alloc& allocator_;

  // Forward declaration
  template <typename node_type = self_type, typename payload_return_type = P,
            typename value_return_type = V>
  class Iterator;
  typedef Iterator<> iterator_type;
  typedef Iterator<const self_type, const P, const V> const_iterator_type;

  self_type* next_leaf_ = nullptr;
  self_type* prev_leaf_ = nullptr;

#if ALEX_DATA_NODE_SEP_ARRAYS
  T* key_slots_ = nullptr;  // holds keys
  P* payload_slots_ =
      nullptr;  // holds payloads, must be same size as key_slots
#else
  V* data_slots_ = nullptr;  // holds key-payload pairs
#endif

  int data_capacity_ = 0;  // size of key/data_slots array
  int num_keys_ = 0;  // number of filled key/data slots (as opposed to gaps)

  // Bitmap: each uint64_t represents 64 positions in reverse order
  // (i.e., each uint64_t is "read" from the right-most bit to the left-most
  // bit)
  uint64_t* bitmap_ = nullptr;
  int bitmap_size_ = 0;  // number of int64_t in bitmap

  // Variables related to resizing (expansions and contractions)
  static constexpr double kMaxDensity_ = 0.8;  // density after contracting,
                                               // also determines the expansion
                                               // threshold
  static constexpr double kInitDensity_ =
      0.7;  // density of data nodes after bulk loading
  static constexpr double kMinDensity_ = 0.6;  // density after expanding, also
                                               // determines the contraction
                                               // threshold
  double expansion_threshold_ = 1;  // expand after m_num_keys is >= this number
  double contraction_threshold_ =
      0;  // contract after m_num_keys is < this number
  static constexpr int kDefaultMaxDataNodeBytes_ =
      1 << 24;  // by default, maximum data node size is 16MB
  int max_slots_ =
      kDefaultMaxDataNodeBytes_ /
      sizeof(V);  // cannot expand beyond this number of key/data slots

  // Counters used in cost models
  long long num_shifts_ = 0;                 // does not reset after resizing
  long long num_exp_search_iterations_ = 0;  // does not reset after resizing
  int num_lookups_ = 0;                      // does not reset after resizing
  int num_inserts_ = 0;                      // does not reset after resizing
  int num_resizes_ = 0;  // technically not required, but nice to have

  // Variables for determining append-mostly behavior
  T max_key_ = std::numeric_limits<
      T>::lowest();  // max key in node, updates after inserts but not erases
  T min_key_ = std::numeric_limits<T>::max();  // min key in node, updates after
                                               // inserts but not erases
  int num_right_out_of_bounds_inserts_ =
      0;  // number of inserts that are larger than the max key
  int num_left_out_of_bounds_inserts_ =
      0;  // number of inserts that are smaller than the min key
  // Node is considered append-mostly if the fraction of inserts that are out of
  // bounds is above this threshold
  // Append-mostly nodes will expand in a manner that anticipates further
  // appends
  static constexpr double kAppendMostlyThreshold = 0.9;

  // Purely for benchmark debugging purposes
  double expected_avg_exp_search_iterations_ = 0;
  double expected_avg_shifts_ = 0;

  // Placed at the end of the key/data slots if there are gaps after the max key
  static constexpr T kEndSentinel_ = std::numeric_limits<T>::max();

  /*** Constructors and destructors ***/

  explicit AlexDataNode(const Compare& comp = Compare(),
                        const Alloc& alloc = Alloc())
      : AlexNode<T, P>(0, true), key_less_(comp), allocator_(alloc) {}

  AlexDataNode(short level, int max_data_node_slots,
               const Compare& comp = Compare(), const Alloc& alloc = Alloc())
      : AlexNode<T, P>(level, true),
        key_less_(comp),
        allocator_(alloc),
        max_slots_(max_data_node_slots) {}

  ~AlexDataNode() {
#if ALEX_DATA_NODE_SEP_ARRAYS
    if (key_slots_ == nullptr) {
      return;
    }
    key_allocator().deallocate(key_slots_, data_capacity_);
    payload_allocator().deallocate(payload_slots_, data_capacity_);
#else
    if (data_slots_ == nullptr) {
      return;
    }
    value_allocator().deallocate(data_slots_, data_capacity_);
#endif
    bitmap_allocator().deallocate(bitmap_, bitmap_size_);
  }

  AlexDataNode(const self_type& other)
      : AlexNode<T, P>(other),
        key_less_(other.key_less_),
        allocator_(other.allocator_),
        next_leaf_(other.next_leaf_),
        prev_leaf_(other.prev_leaf_),
        data_capacity_(other.data_capacity_),
        num_keys_(other.num_keys_),
        bitmap_size_(other.bitmap_size_),
        expansion_threshold_(other.expansion_threshold_),
        contraction_threshold_(other.contraction_threshold_),
        max_slots_(other.max_slots_),
        num_shifts_(other.num_shifts_),
        num_exp_search_iterations_(other.num_exp_search_iterations_),
        num_lookups_(other.num_lookups_),
        num_inserts_(other.num_inserts_),
        num_resizes_(other.num_resizes_),
        max_key_(other.max_key_),
        min_key_(other.min_key_),
        num_right_out_of_bounds_inserts_(
            other.num_right_out_of_bounds_inserts_),
        num_left_out_of_bounds_inserts_(other.num_left_out_of_bounds_inserts_),
        expected_avg_exp_search_iterations_(
            other.expected_avg_exp_search_iterations_),
        expected_avg_shifts_(other.expected_avg_shifts_) {
#if ALEX_DATA_NODE_SEP_ARRAYS
    key_slots_ = new (key_allocator().allocate(other.data_capacity_))
        T[other.data_capacity_];
    std::copy(other.key_slots_, other.key_slots_ + other.data_capacity_,
              key_slots_);
    payload_slots_ = new (payload_allocator().allocate(other.data_capacity_))
        P[other.data_capacity_];
    std::copy(other.payload_slots_, other.payload_slots_ + other.data_capacity_,
              payload_slots_);
#else
    data_slots_ = new (value_allocator().allocate(other.data_capacity_))
        V[other.data_capacity_];
    std::copy(other.data_slots_, other.data_slots_ + other.data_capacity_,
              data_slots_);
#endif
    bitmap_ = new (bitmap_allocator().allocate(other.bitmap_size_))
        uint64_t[other.bitmap_size_];
    std::copy(other.bitmap_, other.bitmap_ + other.bitmap_size_, bitmap_);
  }

  /*** Allocators ***/

  key_alloc_type key_allocator() { return key_alloc_type(allocator_); }

  payload_alloc_type payload_allocator() {
    return payload_alloc_type(allocator_);
  }

  value_alloc_type value_allocator() { return value_alloc_type(allocator_); }

  bitmap_alloc_type bitmap_allocator() { return bitmap_alloc_type(allocator_); }

  /*** General helper functions ***/

  inline T& get_key(int pos) const { return ALEX_DATA_NODE_KEY_AT(pos); }

  inline P& get_payload(int pos) const {
    return ALEX_DATA_NODE_PAYLOAD_AT(pos);
  }

  // Check whether the position corresponds to a key (as opposed to a gap)
  inline bool check_exists(int pos) const {
    assert(pos >= 0 && pos < data_capacity_);
    int bitmap_pos = pos >> 6;
    int bit_pos = pos - (bitmap_pos << 6);
    return static_cast<bool>(bitmap_[bitmap_pos] & (1ULL << bit_pos));
  }

  // Mark the entry for position in the bitmap
  inline void set_bit(int pos) {
    assert(pos >= 0 && pos < data_capacity_);
    int bitmap_pos = pos >> 6;
    int bit_pos = pos - (bitmap_pos << 6);
    bitmap_[bitmap_pos] |= (1ULL << bit_pos);
  }

  // Mark the entry for position in the bitmap
  inline void set_bit(uint64_t bitmap[], int pos) {
    int bitmap_pos = pos >> 6;
    int bit_pos = pos - (bitmap_pos << 6);
    bitmap[bitmap_pos] |= (1ULL << bit_pos);
  }

  // Unmark the entry for position in the bitmap
  inline void unset_bit(int pos) {
    assert(pos >= 0 && pos < data_capacity_);
    int bitmap_pos = pos >> 6;
    int bit_pos = pos - (bitmap_pos << 6);
    bitmap_[bitmap_pos] &= ~(1ULL << bit_pos);
  }

  // Value of first (i.e., min) key
  T first_key() const {
    for (int i = 0; i < data_capacity_; i++) {
      if (check_exists(i)) return get_key(i);
    }
    return std::numeric_limits<T>::max();
  }

  // Value of last (i.e., max) key
  T last_key() const {
    for (int i = data_capacity_ - 1; i >= 0; i--) {
      if (check_exists(i)) return get_key(i);
    }
    return std::numeric_limits<T>::lowest();
  }

  // Position in key/data_slots of first (i.e., min) key
  int first_pos() const {
    for (int i = 0; i < data_capacity_; i++) {
      if (check_exists(i)) return i;
    }
    return 0;
  }

  // Position in key/data_slots of last (i.e., max) key
  int last_pos() const {
    for (int i = data_capacity_ - 1; i >= 0; i--) {
      if (check_exists(i)) return i;
    }
    return 0;
  }

  // Number of keys between positions left and right (exclusive) in
  // key/data_slots
  int num_keys_in_range(int left, int right) const {
    assert(left >= 0 && left <= right && right <= data_capacity_);
    int num_keys = 0;
    int left_bitmap_idx = left >> 6;
    int right_bitmap_idx = right >> 6;
    if (left_bitmap_idx == right_bitmap_idx) {
      uint64_t bitmap_data = bitmap_[left_bitmap_idx];
      int left_bit_pos = left - (left_bitmap_idx << 6);
      bitmap_data &= ~((1ULL << left_bit_pos) - 1);
      int right_bit_pos = right - (right_bitmap_idx << 6);
      bitmap_data &= ((1ULL << right_bit_pos) - 1);
      num_keys += _mm_popcnt_u64(bitmap_data);
    } else {
      uint64_t left_bitmap_data = bitmap_[left_bitmap_idx];
      int bit_pos = left - (left_bitmap_idx << 6);
      left_bitmap_data &= ~((1ULL << bit_pos) - 1);
      num_keys += _mm_popcnt_u64(left_bitmap_data);
      for (int i = left_bitmap_idx + 1; i < right_bitmap_idx; i++) {
        num_keys += _mm_popcnt_u64(bitmap_[i]);
      }
      if (right_bitmap_idx != bitmap_size_) {
        uint64_t right_bitmap_data = bitmap_[right_bitmap_idx];
        bit_pos = right - (right_bitmap_idx << 6);
        right_bitmap_data &= ((1ULL << bit_pos) - 1);
        num_keys += _mm_popcnt_u64(right_bitmap_data);
      }
    }
    return num_keys;
  }

  // True if a < b
  template <class K>
  forceinline bool key_less(const T& a, const K& b) const {
    return key_less_(a, b);
  }

  // True if a <= b
  template <class K>
  forceinline bool key_lessequal(const T& a, const K& b) const {
    return !key_less_(b, a);
  }

  // True if a > b
  template <class K>
  forceinline bool key_greater(const T& a, const K& b) const {
    return key_less_(b, a);
  }

  // True if a >= b
  template <class K>
  forceinline bool key_greaterequal(const T& a, const K& b) const {
    return !key_less_(a, b);
  }

  // True if a == b
  template <class K>
  forceinline bool key_equal(const T& a, const K& b) const {
    return !key_less_(a, b) && !key_less_(b, a);
  }

  /*** Iterator ***/

  // Forward iterator meant for iterating over a single data node.
  // By default, it is a "normal" non-const iterator.
  // Can be templated to be a const iterator.
  template <typename node_type, typename payload_return_type,
            typename value_return_type>
  class Iterator {
   public:
    node_type* node_;
    int cur_idx_ = 0;  // current position in key/data_slots, -1 if at end
    int cur_bitmap_idx_ = 0;  // current position in bitmap
    uint64_t cur_bitmap_data_ =
        0;  // caches the relevant data in the current bitmap position

    explicit Iterator(node_type* node) : node_(node) {}

    Iterator(node_type* node, int idx) : node_(node), cur_idx_(idx) {
      initialize();
    }

    void initialize() {
      cur_bitmap_idx_ = cur_idx_ >> 6;
      cur_bitmap_data_ = node_->bitmap_[cur_bitmap_idx_];

      // Zero out extra bits
      int bit_pos = cur_idx_ - (cur_bitmap_idx_ << 6);
      cur_bitmap_data_ &= ~((1ULL << bit_pos) - 1);

      (*this)++;
    }

    void operator++(int) {
      while (cur_bitmap_data_ == 0) {
        cur_bitmap_idx_++;
        if (cur_bitmap_idx_ >= node_->bitmap_size_) {
          cur_idx_ = -1;
          return;
        }
        cur_bitmap_data_ = node_->bitmap_[cur_bitmap_idx_];
      }
      uint64_t bit = extract_rightmost_one(cur_bitmap_data_);
      cur_idx_ = get_offset(cur_bitmap_idx_, bit);
      cur_bitmap_data_ = remove_rightmost_one(cur_bitmap_data_);
    }

#if ALEX_DATA_NODE_SEP_ARRAYS
    V operator*() const {
      return std::make_pair(node_->key_slots_[cur_idx_],
                            node_->payload_slots_[cur_idx_]);
    }
#else
    value_return_type& operator*() const {
      return node_->data_slots_[cur_idx_];
    }
#endif

    const T& key() const {
#if ALEX_DATA_NODE_SEP_ARRAYS
      return node_->key_slots_[cur_idx_];
#else
      return node_->data_slots_[cur_idx_].first;
#endif
    }

    payload_return_type& payload() const {
#if ALEX_DATA_NODE_SEP_ARRAYS
      return node_->payload_slots_[cur_idx_];
#else
      return node_->data_slots_[cur_idx_].second;
#endif
    }

    bool is_end() const { return cur_idx_ == -1; }

    bool operator==(const Iterator& rhs) const {
      return cur_idx_ == rhs.cur_idx_;
    }

    bool operator!=(const Iterator& rhs) const { return !(*this == rhs); };
  };

  iterator_type begin() { return iterator_type(this, 0); }

  /*** Cost model ***/

  // Empirical average number of shifts per insert
  double shifts_per_insert() const {
    if (num_inserts_ == 0) {
      return 0;
    }
    return num_shifts_ / static_cast<double>(num_inserts_);
  }

  // Empirical average number of exponential search iterations per operation
  // (either lookup or insert)
  double exp_search_iterations_per_operation() const {
    if (num_inserts_ + num_lookups_ == 0) {
      return 0;
    }
    return num_exp_search_iterations_ /
           static_cast<double>(num_inserts_ + num_lookups_);
  }

  double empirical_cost() const {
    if (num_inserts_ + num_lookups_ == 0) {
      return 0;
    }
    double frac_inserts =
        static_cast<double>(num_inserts_) / (num_inserts_ + num_lookups_);
    return kExpSearchIterationsWeight * exp_search_iterations_per_operation() +
           kShiftsWeight * shifts_per_insert() * frac_inserts;
  }

  // Empirical fraction of operations (either lookup or insert) that are inserts
  double frac_inserts() const {
    int num_ops = num_inserts_ + num_lookups_;
    if (num_ops == 0) {
      return 0;  // if no operations, assume no inserts
    }
    return static_cast<double>(num_inserts_) / (num_inserts_ + num_lookups_);
  }

  void reset_stats() {
    num_shifts_ = 0;
    num_exp_search_iterations_ = 0;
    num_lookups_ = 0;
    num_inserts_ = 0;
    num_resizes_ = 0;
  }

  // Computes the expected cost of the current node
  double compute_expected_cost(double frac_inserts = 0) {
    if (num_keys_ == 0) {
      return 0;
    }

    ExpectedSearchIterationsAccumulator search_iters_accumulator;
    ExpectedShiftsAccumulator shifts_accumulator(data_capacity_);
    const_iterator_type it(this, 0);
    for (; !it.is_end(); it++) {
      int predicted_position = std::max(
          0, std::min(data_capacity_ - 1, this->model_.predict(it.key())));
      search_iters_accumulator.accumulate(it.cur_idx_, predicted_position);
      shifts_accumulator.accumulate(it.cur_idx_, predicted_position);
    }
    expected_avg_exp_search_iterations_ = search_iters_accumulator.get_stat();
    expected_avg_shifts_ = shifts_accumulator.get_stat();
    double cost =
        kExpSearchIterationsWeight * expected_avg_exp_search_iterations_ +
        kShiftsWeight * expected_avg_shifts_ * frac_inserts;
    return cost;
  }

  // Computes the expected cost of a data node constructed using the input dense
  // array of keys
  // Assumes existing_model is trained on the dense array of keys
  static double compute_expected_cost(
      const V* values, int num_keys, double density,
      double expected_insert_frac,
      const LinearModel<T>* existing_model = nullptr, bool use_sampling = false,
      DataNodeStats* stats = nullptr) {
    if (use_sampling) {
      return compute_expected_cost_sampling(values, num_keys, density,
                                            expected_insert_frac,
                                            existing_model, stats);
    }

    if (num_keys == 0) {
      return 0;
    }

    int data_capacity =
        std::max(static_cast<int>(num_keys / density), num_keys + 1);

    // Compute what the node's model would be
    LinearModel<T> model;
    if (existing_model == nullptr) {
      build_model(values, num_keys, &model);
    } else {
      model.a_ = existing_model->a_;
      model.b_ = existing_model->b_;
    }
    model.expand(static_cast<double>(data_capacity) / num_keys);

    // Compute expected stats in order to compute the expected cost
    double cost = 0;
    double expected_avg_exp_search_iterations = 0;
    double expected_avg_shifts = 0;
    if (expected_insert_frac == 0) {
      ExpectedSearchIterationsAccumulator acc;
      build_node_implicit(values, num_keys, data_capacity, &acc, &model);
      expected_avg_exp_search_iterations = acc.get_stat();
    } else {
      ExpectedIterationsAndShiftsAccumulator acc(data_capacity);
      build_node_implicit(values, num_keys, data_capacity, &acc, &model);
      expected_avg_exp_search_iterations =
          acc.get_expected_num_search_iterations();
      expected_avg_shifts = acc.get_expected_num_shifts();
    }
    cost = kExpSearchIterationsWeight * expected_avg_exp_search_iterations +
           kShiftsWeight * expected_avg_shifts * expected_insert_frac;

    if (stats) {
      stats->num_search_iterations = expected_avg_exp_search_iterations;
      stats->num_shifts = expected_avg_shifts;
    }

    return cost;
  }

  // Helper function for compute_expected_cost
  // Implicitly build the data node in order to collect the stats
  static void build_node_implicit(const V* values, int num_keys,
                                  int data_capacity, StatAccumulator* acc,
                                  const LinearModel<T>* model) {
    int last_position = -1;
    int keys_remaining = num_keys;
    for (int i = 0; i < num_keys; i++) {
      int predicted_position = std::max(
          0, std::min(data_capacity - 1, model->predict(values[i].first)));
      int actual_position =
          std::max<int>(predicted_position, last_position + 1);
      int positions_remaining = data_capacity - actual_position;
      if (positions_remaining < keys_remaining) {
        actual_position = data_capacity - keys_remaining;
        for (int j = i; j < num_keys; j++) {
          predicted_position = std::max(
              0, std::min(data_capacity - 1, model->predict(values[j].first)));
          acc->accumulate(actual_position, predicted_position);
          actual_position++;
        }
        break;
      }
      acc->accumulate(actual_position, predicted_position);
      last_position = actual_position;
      keys_remaining--;
    }
  }

  // Using sampling, approximates the expected cost of a data node constructed
  // using the input dense array of keys
  // Assumes existing_model is trained on the dense array of keys
  // Uses progressive sampling: keep increasing the sample size until the
  // computed stats stop changing drastically
  static double compute_expected_cost_sampling(
      const V* values, int num_keys, double density,
      double expected_insert_frac,
      const LinearModel<T>* existing_model = nullptr,
      DataNodeStats* stats = nullptr) {
    const static int min_sample_size = 25;

    // Stop increasing sample size if relative diff of stats between samples is
    // less than this
    const static double rel_diff_threshold = 0.2;

    // Equivalent threshold in log2-space
    const static double abs_log2_diff_threshold =
        std::log2(1 + rel_diff_threshold);

    // Increase sample size by this many times each iteration
    const static int sample_size_multiplier = 2;

    // If num_keys is below this threshold, we compute entropy exactly
    const static int exact_computation_size_threshold =
        (min_sample_size * sample_size_multiplier * sample_size_multiplier * 2);

    // Target fraction of the keys to use in the initial sample
    const static double init_sample_frac = 0.01;

    // If the number of keys is sufficiently small, we do not sample
    if (num_keys < exact_computation_size_threshold) {
      return compute_expected_cost(values, num_keys, density,
                                   expected_insert_frac, existing_model, false,
                                   stats);
    }

    LinearModel<T> model;  // trained for full dense array
    if (existing_model == nullptr) {
      build_model(values, num_keys, &model);
    } else {
      model.a_ = existing_model->a_;
      model.b_ = existing_model->b_;
    }

    // Compute initial sample size and step size
    // Right now, sample_num_keys holds the target sample num keys
    int sample_num_keys = std::max(
        static_cast<int>(num_keys * init_sample_frac), min_sample_size);
    int step_size = 1;
    double tmp_sample_size =
        num_keys;  // this helps us determine the right sample size
    while (tmp_sample_size >= sample_num_keys) {
      tmp_sample_size /= sample_size_multiplier;
      step_size *= sample_size_multiplier;
    }
    step_size /= sample_size_multiplier;
    sample_num_keys =
        num_keys /
        step_size;  // now sample_num_keys is the actual sample num keys

    std::vector<SampleDataNodeStats>
        sample_stats;  // stats computed usinig each sample
    bool compute_shifts = expected_insert_frac !=
                          0;  // whether we need to compute expected shifts
    double log2_num_keys = std::log2(num_keys);
    double expected_full_search_iters =
        0;  // extrapolated estimate for search iters on the full array
    double expected_full_shifts =
        0;  // extrapolated estimate shifts on the full array
    bool search_iters_computed =
        false;  // set to true when search iters is accurately computed
    bool shifts_computed =
        false;  // set to true when shifts is accurately computed

    // Progressively increase sample size
    while (true) {
      int sample_data_capacity = std::max(
          static_cast<int>(sample_num_keys / density), sample_num_keys + 1);
      LinearModel<T> sample_model(model.a_, model.b_);
      sample_model.expand(static_cast<double>(sample_data_capacity) / num_keys);

      // Compute stats using the sample
      if (expected_insert_frac == 0) {
        ExpectedSearchIterationsAccumulator acc;
        build_node_implicit_sampling(values, num_keys, sample_num_keys,
                                     sample_data_capacity, step_size, &acc,
                                     &sample_model);
        sample_stats.push_back({std::log2(sample_num_keys), acc.get_stat(), 0});
      } else {
        ExpectedIterationsAndShiftsAccumulator acc(sample_data_capacity);
        build_node_implicit_sampling(values, num_keys, sample_num_keys,
                                     sample_data_capacity, step_size, &acc,
                                     &sample_model);
        sample_stats.push_back({std::log2(sample_num_keys),
                                acc.get_expected_num_search_iterations(),
                                std::log2(acc.get_expected_num_shifts())});
      }

      if (sample_stats.size() >= 3) {
        // Check if the diff in stats is sufficiently small
        SampleDataNodeStats& s0 = sample_stats[sample_stats.size() - 3];
        SampleDataNodeStats& s1 = sample_stats[sample_stats.size() - 2];
        SampleDataNodeStats& s2 = sample_stats[sample_stats.size() - 1];
        // (y1 - y0) / (x1 - x0) = (y2 - y1) / (x2 - x1) --> y2 = (y1 - y0) /
        // (x1 - x0) * (x2 - x1) + y1
        double expected_s2_search_iters =
            (s1.num_search_iterations - s0.num_search_iterations) /
                (s1.log2_sample_size - s0.log2_sample_size) *
                (s2.log2_sample_size - s1.log2_sample_size) +
            s1.num_search_iterations;
        double rel_diff =
            std::abs((s2.num_search_iterations - expected_s2_search_iters) /
                     s2.num_search_iterations);
        if (rel_diff <= rel_diff_threshold || num_keys <= 2 * sample_num_keys) {
          search_iters_computed = true;
          expected_full_search_iters =
              (s2.num_search_iterations - s1.num_search_iterations) /
                  (s2.log2_sample_size - s1.log2_sample_size) *
                  (log2_num_keys - s2.log2_sample_size) +
              s2.num_search_iterations;
        }
        if (compute_shifts) {
          double expected_s2_log2_shifts =
              (s1.log2_num_shifts - s0.log2_num_shifts) /
                  (s1.log2_sample_size - s0.log2_sample_size) *
                  (s2.log2_sample_size - s1.log2_sample_size) +
              s1.log2_num_shifts;
          double abs_diff =
              std::abs((s2.log2_num_shifts - expected_s2_log2_shifts) /
                       s2.log2_num_shifts);
          if (abs_diff <= abs_log2_diff_threshold ||
              num_keys <= 2 * sample_num_keys) {
            shifts_computed = true;
            double expected_full_log2_shifts =
                (s2.log2_num_shifts - s1.log2_num_shifts) /
                    (s2.log2_sample_size - s1.log2_sample_size) *
                    (log2_num_keys - s2.log2_sample_size) +
                s2.log2_num_shifts;
            expected_full_shifts = std::pow(2, expected_full_log2_shifts);
          }
        }

        // If diff in stats is sufficiently small, return the approximate
        // expected cost
        if ((expected_insert_frac == 0 && search_iters_computed) ||
            (expected_insert_frac > 0 && search_iters_computed &&
             shifts_computed)) {
          double cost =
              kExpSearchIterationsWeight * expected_full_search_iters +
              kShiftsWeight * expected_full_shifts * expected_insert_frac;
          if (stats) {
            stats->num_search_iterations = expected_full_search_iters;
            stats->num_shifts = expected_full_shifts;
          }
          return cost;
        }
      }

      step_size /= sample_size_multiplier;
      sample_num_keys = num_keys / step_size;
    }
  }

  // Helper function for compute_expected_cost_sampling
  // Implicitly build the data node in order to collect the stats
  // keys is the full un-sampled array of keys
  // sample_num_keys and sample_data_capacity refer to a data node that is
  // created only over the sample
  // sample_model is trained for the sampled data node
  static void build_node_implicit_sampling(const V* values, int num_keys,
                                           int sample_num_keys,
                                           int sample_data_capacity,
                                           int step_size, StatAccumulator* ent,
                                           const LinearModel<T>* sample_model) {
    int last_position = -1;
    int sample_keys_remaining = sample_num_keys;
    for (int i = 0; i < num_keys; i += step_size) {
      int predicted_position =
          std::max(0, std::min(sample_data_capacity - 1,
                               sample_model->predict(values[i].first)));
      int actual_position =
          std::max<int>(predicted_position, last_position + 1);
      int positions_remaining = sample_data_capacity - actual_position;
      if (positions_remaining < sample_keys_remaining) {
        actual_position = sample_data_capacity - sample_keys_remaining;
        for (int j = i; j < num_keys; j += step_size) {
          predicted_position =
              std::max(0, std::min(sample_data_capacity - 1,
                                   sample_model->predict(values[j].first)));
          ent->accumulate(actual_position, predicted_position);
          actual_position++;
        }
        break;
      }
      ent->accumulate(actual_position, predicted_position);
      last_position = actual_position;
      sample_keys_remaining--;
    }
  }

  // Computes the expected cost of a data node constructed using the keys
  // between left and right in the
  // key/data_slots of an existing node
  // Assumes existing_model is trained on the dense array of keys
  static double compute_expected_cost_from_existing(
      const self_type* node, int left, int right, double density,
      double expected_insert_frac,
      const LinearModel<T>* existing_model = nullptr,
      DataNodeStats* stats = nullptr) {
    assert(left >= 0 && right <= node->data_capacity_);

    LinearModel<T> model;
    int num_actual_keys = 0;
    if (existing_model == nullptr) {
      const_iterator_type it(node, left);
      LinearModelBuilder<T> builder(&model);
      for (int i = 0; it.cur_idx_ < right && !it.is_end(); it++, i++) {
        builder.add(it.key(), i);
        num_actual_keys++;
      }
      builder.build();
    } else {
      num_actual_keys = node->num_keys_in_range(left, right);
      model.a_ = existing_model->a_;
      model.b_ = existing_model->b_;
    }

    if (num_actual_keys == 0) {
      return 0;
    }
    int data_capacity = std::max(static_cast<int>(num_actual_keys / density),
                                 num_actual_keys + 1);
    model.expand(static_cast<double>(data_capacity) / num_actual_keys);

    // Compute expected stats in order to compute the expected cost
    double cost = 0;
    double expected_avg_exp_search_iterations = 0;
    double expected_avg_shifts = 0;
    if (expected_insert_frac == 0) {
      ExpectedSearchIterationsAccumulator acc;
      build_node_implicit_from_existing(node, left, right, num_actual_keys,
                                        data_capacity, &acc, &model);
      expected_avg_exp_search_iterations = acc.get_stat();
    } else {
      ExpectedIterationsAndShiftsAccumulator acc(data_capacity);
      build_node_implicit_from_existing(node, left, right, num_actual_keys,
                                        data_capacity, &acc, &model);
      expected_avg_exp_search_iterations =
          acc.get_expected_num_search_iterations();
      expected_avg_shifts = acc.get_expected_num_shifts();
    }
    cost = kExpSearchIterationsWeight * expected_avg_exp_search_iterations +
           kShiftsWeight * expected_avg_shifts * expected_insert_frac;

    if (stats) {
      stats->num_search_iterations = expected_avg_exp_search_iterations;
      stats->num_shifts = expected_avg_shifts;
    }

    return cost;
  }

  // Helper function for compute_expected_cost
  // Implicitly build the data node in order to collect the stats
  static void build_node_implicit_from_existing(const self_type* node, int left,
                                                int right, int num_actual_keys,
                                                int data_capacity,
                                                StatAccumulator* acc,
                                                const LinearModel<T>* model) {
    int last_position = -1;
    int keys_remaining = num_actual_keys;
    const_iterator_type it(node, left);
    for (; it.cur_idx_ < right && !it.is_end(); it++) {
      int predicted_position =
          std::max(0, std::min(data_capacity - 1, model->predict(it.key())));
      int actual_position =
          std::max<int>(predicted_position, last_position + 1);
      int positions_remaining = data_capacity - actual_position;
      if (positions_remaining < keys_remaining) {
        actual_position = data_capacity - keys_remaining;
        for (; actual_position < data_capacity; actual_position++, it++) {
          predicted_position = std::max(
              0, std::min(data_capacity - 1, model->predict(it.key())));
          acc->accumulate(actual_position, predicted_position);
        }
        break;
      }
      acc->accumulate(actual_position, predicted_position);
      last_position = actual_position;
      keys_remaining--;
    }
  }

  /*** Bulk loading and model building ***/

  // Initalize key/payload/bitmap arrays and relevant metadata
  void initialize(int num_keys, double density) {
    num_keys_ = num_keys;
    data_capacity_ =
        std::max(static_cast<int>(num_keys / density), num_keys + 1);
    bitmap_size_ = static_cast<size_t>(std::ceil(data_capacity_ / 64.));
    bitmap_ = new (bitmap_allocator().allocate(bitmap_size_))
        uint64_t[bitmap_size_]();  // initialize to all false
#if ALEX_DATA_NODE_SEP_ARRAYS
    key_slots_ =
        new (key_allocator().allocate(data_capacity_)) T[data_capacity_];
    payload_slots_ =
        new (payload_allocator().allocate(data_capacity_)) P[data_capacity_];
#else
    data_slots_ =
        new (value_allocator().allocate(data_capacity_)) V[data_capacity];
#endif
  }

  // Assumes pretrained_model is trained on dense array of keys
  void bulk_load(const V values[], int num_keys,
                 const LinearModel<T>* pretrained_model = nullptr,
                 bool train_with_sample = false) {
    initialize(num_keys, kInitDensity_);

    if (num_keys == 0) {
      expansion_threshold_ = data_capacity_;
      contraction_threshold_ = 0;
      for (int i = 0; i < data_capacity_; i++) {
        ALEX_DATA_NODE_KEY_AT(i) = kEndSentinel_;
      }
      return;
    }

    // Build model
    if (pretrained_model != nullptr) {
      this->model_.a_ = pretrained_model->a_;
      this->model_.b_ = pretrained_model->b_;
    } else {
      build_model(values, num_keys, &(this->model_), train_with_sample);
    }
    this->model_.expand(static_cast<double>(data_capacity_) / num_keys);

    // Model-based inserts
    int last_position = -1;
    int keys_remaining = num_keys;
    for (int i = 0; i < num_keys; i++) {
      int position = this->model_.predict(values[i].first);
      position = std::max<int>(position, last_position + 1);

      int positions_remaining = data_capacity_ - position;
      if (positions_remaining < keys_remaining) {
        // fill the rest of the store contiguously
        int pos = data_capacity_ - keys_remaining;
        for (int j = last_position + 1; j < pos; j++) {
          ALEX_DATA_NODE_KEY_AT(j) = values[i].first;
        }
        for (int j = i; j < num_keys; j++) {
#if ALEX_DATA_NODE_SEP_ARRAYS
          key_slots_[pos] = values[j].first;
          payload_slots_[pos] = values[j].second;
#else
          data_slots_[pos] = values[j];
#endif
          set_bit(pos);
          pos++;
        }
        last_position = pos - 1;
        break;
      }

      for (int j = last_position + 1; j < position; j++) {
        ALEX_DATA_NODE_KEY_AT(j) = values[i].first;
      }

#if ALEX_DATA_NODE_SEP_ARRAYS
      key_slots_[position] = values[i].first;
      payload_slots_[position] = values[i].second;
#else
      data_slots_[position] = values[i];
#endif
      set_bit(position);

      last_position = position;

      keys_remaining--;
    }

    for (int i = last_position + 1; i < data_capacity_; i++) {
      ALEX_DATA_NODE_KEY_AT(i) = kEndSentinel_;
    }

    expansion_threshold_ = std::min(std::max(data_capacity_ * kMaxDensity_,
                                             static_cast<double>(num_keys + 1)),
                                    static_cast<double>(data_capacity_));
    contraction_threshold_ = data_capacity_ * kMinDensity_;
    min_key_ = values[0].first;
    max_key_ = values[num_keys - 1].first;
  }

  // Bulk load using the keys between the left and right positions in
  // key/data_slots of an existing data node
  // keep_left and keep_right are set if the existing node was append-mostly
  // If the linear model and num_actual_keys have been precomputed, we can avoid
  // redundant work
  void bulk_load_from_existing(
      const self_type* node, int left, int right, bool keep_left = false,
      bool keep_right = false,
      const LinearModel<T>* precomputed_model = nullptr,
      int precomputed_num_actual_keys = -1) {
    assert(left >= 0 && right <= node->data_capacity_);

    // Build model
    int num_actual_keys = 0;
    if (precomputed_model == nullptr || precomputed_num_actual_keys == -1) {
      const_iterator_type it(node, left);
      LinearModelBuilder<T> builder(&(this->model_));
      for (int i = 0; it.cur_idx_ < right && !it.is_end(); it++, i++) {
        builder.add(it.key(), i);
        num_actual_keys++;
      }
      builder.build();
    } else {
      num_actual_keys = precomputed_num_actual_keys;
      this->model_.a_ = precomputed_model->a_;
      this->model_.b_ = precomputed_model->b_;
    }

    initialize(num_actual_keys, kMinDensity_);
    if (num_actual_keys == 0) {
      expansion_threshold_ = data_capacity_;
      contraction_threshold_ = 0;
      for (int i = 0; i < data_capacity_; i++) {
        ALEX_DATA_NODE_KEY_AT(i) = kEndSentinel_;
      }
      return;
    }

    // Special casing if existing node was append-mostly
    if (keep_left) {
      this->model_.expand((num_actual_keys / kMaxDensity_) / num_keys_);
    } else if (keep_right) {
      this->model_.expand((num_actual_keys / kMaxDensity_) / num_keys_);
      this->model_.b_ += (data_capacity_ - (num_actual_keys / kMaxDensity_));
    } else {
      this->model_.expand(static_cast<double>(data_capacity_) / num_keys_);
    }

    // Model-based inserts
    int last_position = -1;
    int keys_remaining = num_keys_;
    const_iterator_type it(node, left);
    min_key_ = it.key();
    for (; it.cur_idx_ < right && !it.is_end(); it++) {
      int position = this->model_.predict(it.key());
      position = std::max<int>(position, last_position + 1);

      int positions_remaining = data_capacity_ - position;
      if (positions_remaining < keys_remaining) {
        // fill the rest of the store contiguously
        int pos = data_capacity_ - keys_remaining;
        for (int j = last_position + 1; j < pos; j++) {
          ALEX_DATA_NODE_KEY_AT(j) = it.key();
        }
        for (; pos < data_capacity_; pos++, it++) {
#if ALEX_DATA_NODE_SEP_ARRAYS
          key_slots_[pos] = it.key();
          payload_slots_[pos] = it.payload();
#else
          data_slots_[pos] = *it;
#endif
          set_bit(pos);
        }
        last_position = pos - 1;
        break;
      }

      for (int j = last_position + 1; j < position; j++) {
        ALEX_DATA_NODE_KEY_AT(j) = it.key();
      }

#if ALEX_DATA_NODE_SEP_ARRAYS
      key_slots_[position] = it.key();
      payload_slots_[position] = it.payload();
#else
      data_slots_[position] = *it;
#endif
      set_bit(position);

      last_position = position;

      keys_remaining--;
    }

    for (int i = last_position + 1; i < data_capacity_; i++) {
      ALEX_DATA_NODE_KEY_AT(i) = kEndSentinel_;
    }

    max_key_ = ALEX_DATA_NODE_KEY_AT(last_position);

    expansion_threshold_ =
        std::min(std::max(data_capacity_ * kMaxDensity_,
                          static_cast<double>(num_keys_ + 1)),
                 static_cast<double>(data_capacity_));
    contraction_threshold_ = data_capacity_ * kMinDensity_;
  }

  static void build_model(const V* values, int num_keys, LinearModel<T>* model,
                          bool use_sampling = false) {
    if (use_sampling) {
      build_model_sampling(values, num_keys, model);
      return;
    }

    LinearModelBuilder<T> builder(model);
    for (int i = 0; i < num_keys; i++) {
      builder.add(values[i].first, i);
    }
    builder.build();
  }

  // Uses progressive non-random uniform sampling to build the model
  // Progressively increases sample size until model parameters are relatively
  // stable
  static void build_model_sampling(const V* values, int num_keys,
                                   LinearModel<T>* model,
                                   bool verbose = false) {
    const static int sample_size_lower_bound = 10;
    // If slope and intercept change by less than this much between samples,
    // return
    const static double rel_change_threshold = 0.01;
    // If intercept changes by less than this much between samples, return
    const static double abs_change_threshold = 0.5;
    // Increase sample size by this many times each iteration
    const static int sample_size_multiplier = 2;

    // If the number of keys is sufficiently small, we do not sample
    if (num_keys <= sample_size_lower_bound * sample_size_multiplier) {
      build_model(values, num_keys, model, false);
      return;
    }

    int step_size = 1;
    double sample_size = num_keys;
    while (sample_size >= sample_size_lower_bound) {
      sample_size /= sample_size_multiplier;
      step_size *= sample_size_multiplier;
    }
    step_size /= sample_size_multiplier;

    // Run with initial step size
    LinearModelBuilder<T> builder(model);
    for (int i = 0; i < num_keys; i += step_size) {
      builder.add(values[i].first, i);
    }
    builder.build();
    double prev_a = model->a_;
    double prev_b = model->b_;
    if (verbose) {
      std::cout << "Build index, sample size: " << num_keys / step_size
                << " (a, b): (" << prev_a << ", " << prev_b << ")" << std::endl;
    }

    // Keep decreasing step size (increasing sample size) until model does not
    // change significantly
    while (step_size > 1) {
      step_size /= sample_size_multiplier;
      // Need to avoid processing keys we already processed in previous samples
      int i = 0;
      while (i < num_keys) {
        i += step_size;
        for (int j = 1; (j < sample_size_multiplier) && (i < num_keys);
             j++, i += step_size) {
          builder.add(values[i].first, i);
        }
      }
      builder.build();

      double rel_change_in_a = std::abs((model->a_ - prev_a) / prev_a);
      double abs_change_in_b = std::abs(model->b_ - prev_b);
      double rel_change_in_b = std::abs(abs_change_in_b / prev_b);
      if (verbose) {
        std::cout << "Build index, sample size: " << num_keys / step_size
                  << " (a, b): (" << model->a_ << ", " << model->b_ << ") ("
                  << rel_change_in_a << ", " << rel_change_in_b << ")"
                  << std::endl;
      }
      if (rel_change_in_a < rel_change_threshold &&
          (rel_change_in_b < rel_change_threshold ||
           abs_change_in_b < abs_change_threshold)) {
        return;
      }
      prev_a = model->a_;
      prev_b = model->b_;
    }
  }

  // Unused function: builds a spline model by connecting the smallest and
  // largest points instead of using
  // a linear regression
  static void build_spline(const V* values, int num_keys,
                           const LinearModel<T>* model) {
    int y_max = num_keys - 1;
    int y_min = 0;
    model->a_ = static_cast<double>(y_max - y_min) /
                (values[y_max].first - values[y_min].first);
    model->b_ = -1.0 * values[y_min].first * model->a_;
  }

  /*** Lookup ***/

  // Predicts the position of a key using the model
  inline int predict_position(const T& key) const {
    int position = this->model_.predict(key);
    position = std::max<int>(std::min<int>(position, data_capacity_ - 1), 0);
    return position;
  }

  // Searches for the last non-gap position equal to key
  // If no positions equal to key, returns -1
  int find_key(const T& key) {
    num_lookups_++;
    int predicted_pos = predict_position(key);

    // The last key slot with a certain value is guaranteed to be a real key
    // (instead of a gap)
    int pos = exponential_search_upper_bound(predicted_pos, key) - 1;
    if (pos < 0 || !key_equal(ALEX_DATA_NODE_KEY_AT(pos), key)) {
      return -1;
    } else {
      return pos;
    }
  }

  // Searches for the first non-gap position no less than key
  // Returns position in range [0, data_capacity]
  // Compare with lower_bound()
  int find_lower(const T& key) {
    num_lookups_++;
    int predicted_pos = predict_position(key);

    int pos = exponential_search_lower_bound(predicted_pos, key);
    return get_next_filled_position(pos, false);
  }

  // Searches for the first non-gap position greater than key
  // Returns position in range [0, data_capacity]
  // Compare with upper_bound()
  int find_upper(const T& key) {
    num_lookups_++;
    int predicted_pos = predict_position(key);

    int pos = exponential_search_upper_bound(predicted_pos, key);
    return get_next_filled_position(pos, false);
  }

  // Finds position to insert a key.
  // First returned value takes prediction into account.
  // Second returned value is first valid position (i.e., upper_bound of key).
  // If there are duplicate keys, the insert position will be to the right of
  // all existing keys of the same value.
  std::pair<int, int> find_insert_position(const T& key) {
    int predicted_pos =
        predict_position(key);  // first use model to get prediction

    // insert to the right of duplicate keys
    int pos = exponential_search_upper_bound(predicted_pos, key);
    if (predicted_pos <= pos || check_exists(pos)) {
      return {pos, pos};
    } else {
      // Place inserted key as close as possible to the predicted position while
      // maintaining correctness
      return {std::min(predicted_pos, get_next_filled_position(pos, true) - 1),
              pos};
    }
  }

  // Starting from a position, return the first position that is not a gap
  // If no more filled positions, will return data_capacity
  // If exclusive is true, output is at least (pos + 1)
  // If exclusive is false, output can be pos itself
  int get_next_filled_position(int pos, bool exclusive) const {
    if (exclusive) {
      pos++;
      if (pos == data_capacity_) {
        return data_capacity_;
      }
    }

    int curBitmapIdx = pos >> 6;
    uint64_t curBitmapData = bitmap_[curBitmapIdx];

    // Zero out extra bits
    int bit_pos = pos - (curBitmapIdx << 6);
    curBitmapData &= ~((1ULL << (bit_pos)) - 1);

    while (curBitmapData == 0) {
      curBitmapIdx++;
      if (curBitmapIdx >= bitmap_size_) {
        return data_capacity_;
      }
      curBitmapData = bitmap_[curBitmapIdx];
    }
    uint64_t bit = extract_rightmost_one(curBitmapData);
    return get_offset(curBitmapIdx, bit);
  }

  // Searches for the first position greater than key
  // This could be the position for a gap (i.e., its bit in the bitmap is 0)
  // Returns position in range [0, data_capacity]
  // Compare with find_upper()
  template <class K>
  int upper_bound(const K& key) {
    num_lookups_++;
    int position = predict_position(key);
    return exponential_search_upper_bound(position, key);
  }

  // Searches for the first position greater than key, starting from position m
  // Returns position in range [0, data_capacity]
  template <class K>
  inline int exponential_search_upper_bound(int m, const K& key) {
    // Continue doubling the bound until it contains the upper bound. Then use
    // binary search.
    int bound = 1;
    int l, r;  // will do binary search in range [l, r)
    if (key_greater(ALEX_DATA_NODE_KEY_AT(m), key)) {
      int size = m;
      while (bound < size &&
             key_greater(ALEX_DATA_NODE_KEY_AT(m - bound), key)) {
        bound *= 2;
        num_exp_search_iterations_++;
      }
      l = m - std::min<int>(bound, size);
      r = m - bound / 2;
    } else {
      int size = data_capacity_ - m;
      while (bound < size &&
             key_lessequal(ALEX_DATA_NODE_KEY_AT(m + bound), key)) {
        bound *= 2;
        num_exp_search_iterations_++;
      }
      l = m + bound / 2;
      r = m + std::min<int>(bound, size);
    }
    return binary_search_upper_bound(l, r, key);
  }

  // Searches for the first position greater than key in range [l, r)
  // https://stackoverflow.com/questions/6443569/implementation-of-c-lower-bound
  // Returns position in range [l, r]
  template <class K>
  inline int binary_search_upper_bound(int l, int r, const K& key) const {
    while (l < r) {
      int mid = l + (r - l) / 2;
      if (key_lessequal(ALEX_DATA_NODE_KEY_AT(mid), key)) {
        l = mid + 1;
      } else {
        r = mid;
      }
    }
    return l;
  }

  // Searches for the first position no less than key
  // This could be the position for a gap (i.e., its bit in the bitmap is 0)
  // Returns position in range [0, data_capacity]
  // Compare with find_lower()
  template <class K>
  int lower_bound(const K& key) {
    num_lookups_++;
    int position = predict_position(key);
    return exponential_search_lower_bound(position, key);
  }

  // Searches for the first position no less than key, starting from position m
  // Returns position in range [0, data_capacity]
  template <class K>
  inline int exponential_search_lower_bound(int m, const K& key) {
    // Continue doubling the bound until it contains the lower bound. Then use
    // binary search.
    int bound = 1;
    int l, r;  // will do binary search in range [l, r)
    if (key_greaterequal(ALEX_DATA_NODE_KEY_AT(m), key)) {
      int size = m;
      while (bound < size &&
             key_greaterequal(ALEX_DATA_NODE_KEY_AT(m - bound), key)) {
        bound *= 2;
        num_exp_search_iterations_++;
      }
      l = m - std::min<int>(bound, size);
      r = m - bound / 2;
    } else {
      int size = data_capacity_ - m;
      while (bound < size && key_less(ALEX_DATA_NODE_KEY_AT(m + bound), key)) {
        bound *= 2;
        num_exp_search_iterations_++;
      }
      l = m + bound / 2;
      r = m + std::min<int>(bound, size);
    }
    return binary_search_lower_bound(l, r, key);
  }

  // Searches for the first position no less than key in range [l, r)
  // https://stackoverflow.com/questions/6443569/implementation-of-c-lower-bound
  // Returns position in range [l, r]
  template <class K>
  inline int binary_search_lower_bound(int l, int r, const K& key) const {
    while (l < r) {
      int mid = l + (r - l) / 2;
      if (key_greaterequal(ALEX_DATA_NODE_KEY_AT(mid), key)) {
        r = mid;
      } else {
        l = mid + 1;
      }
    }
    return l;
  }

  /*** Inserts and resizes ***/

  // Whether empirical cost deviates significantly from expected cost
  // Also returns false if empirical cost is sufficiently low and is not worth
  // splitting
  inline bool significant_cost_deviation() const {
    double emp_cost = empirical_cost();
    return emp_cost > kNodeLookupsWeight && emp_cost > 1.5 * this->cost_;
  }

  // Returns true if cost is catastrophically high and we want to force a split
  // The heuristic for this is if the number of shifts per insert (expected or
  // empirical) is over 100
  inline bool catastrophic_cost() const {
    return shifts_per_insert() > 100 || expected_avg_shifts_ > 100;
  }

  // First value in returned pair is fail flag:
  // 0 if successful insert (possibly with automatic expansion).
  // 1 if no insert because of significant cost deviation.
  // 2 if no insert because of "catastrophic" cost.
  // 3 if no insert because node is at max capacity.
  // -1 if key already exists and duplicates not allowed.
  //
  // Second value in returned pair is position of inserted key, or of the
  // already-existing key.
  // -1 if no insertion.
  std::pair<int, int> insert(const T& key, const P& payload) {
    // Periodically check for catastrophe
    if (num_inserts_ % 64 == 0 && catastrophic_cost()) {
      return {2, -1};
    }

    // Check if node is full (based on expansion_threshold)
    if (num_keys_ >= expansion_threshold_) {
      if (significant_cost_deviation()) {
        return {1, -1};
      }
      if (catastrophic_cost()) {
        return {2, -1};
      }
      if (num_keys_ > max_slots_ * kMinDensity_) {
        return {3, -1};
      }
      // Expand
      bool keep_left = is_append_mostly_right();
      bool keep_right = is_append_mostly_left();
      resize(kMinDensity_, false, keep_left, keep_right);
      num_resizes_++;
    }

    // Insert
    std::pair<int, int> positions = find_insert_position(key);
    int upper_bound_pos = positions.second;
    if (!allow_duplicates && upper_bound_pos > 0 &&
        key_equal(ALEX_DATA_NODE_KEY_AT(upper_bound_pos - 1), key)) {
      return {-1, upper_bound_pos - 1};
    }
    int insertion_position = positions.first;
    if (insertion_position < data_capacity_ &&
        !check_exists(insertion_position)) {
      insert_element_at(key, payload, insertion_position);
    } else {
      insertion_position =
          insert_using_shifts(key, payload, insertion_position);
    }

    // Update stats
    num_keys_++;
    num_inserts_++;
    if (key > max_key_) {
      max_key_ = key;
      num_right_out_of_bounds_inserts_++;
    }
    if (key < min_key_) {
      min_key_ = key;
      num_left_out_of_bounds_inserts_++;
    }
    return {0, insertion_position};
  }

  // Resize the data node to the target density
  void resize(double target_density, bool force_retrain = false,
              bool keep_left = false, bool keep_right = false) {
    if (num_keys_ == 0) {
      return;
    }

    int new_data_capacity =
        std::max(static_cast<int>(num_keys_ / target_density), num_keys_ + 1);
    auto new_bitmap_size =
        static_cast<size_t>(std::ceil(new_data_capacity / 64.));
    auto new_bitmap = new (bitmap_allocator().allocate(new_bitmap_size))
        uint64_t[new_bitmap_size]();  // initialize to all false
#if ALEX_DATA_NODE_SEP_ARRAYS
    T* new_key_slots =
        new (key_allocator().allocate(new_data_capacity)) T[new_data_capacity];
    P* new_payload_slots = new (payload_allocator().allocate(new_data_capacity))
        P[new_data_capacity];
#else
    V* new_data_slots = new (value_allocator().allocate(new_data_capacity))
        V[new_data_capacity];
#endif

    // Retrain model if the number of keys is sufficiently small (under 50)
    if (num_keys_ < 50 || force_retrain) {
      const_iterator_type it(this, 0);
      LinearModelBuilder<T> builder(&(this->model_));
      for (int i = 0; it.cur_idx_ < data_capacity_ && !it.is_end(); it++, i++) {
        builder.add(it.key(), i);
      }
      builder.build();
      if (keep_left) {
        this->model_.expand(static_cast<double>(data_capacity_) / num_keys_);
      } else if (keep_right) {
        this->model_.expand(static_cast<double>(data_capacity_) / num_keys_);
        this->model_.b_ += (new_data_capacity - data_capacity_);
      } else {
        this->model_.expand(static_cast<double>(new_data_capacity) / num_keys_);
      }
    } else {
      if (keep_right) {
        this->model_.b_ += (new_data_capacity - data_capacity_);
      } else if (!keep_left) {
        this->model_.expand(static_cast<double>(new_data_capacity) /
                            data_capacity_);
      }
    }

    int last_position = -1;
    int keys_remaining = num_keys_;
    const_iterator_type it(this, 0);
    for (; it.cur_idx_ < data_capacity_ && !it.is_end(); it++) {
      int position = this->model_.predict(it.key());
      position = std::max<int>(position, last_position + 1);

      int positions_remaining = new_data_capacity - position;
      if (positions_remaining < keys_remaining) {
        // fill the rest of the store contiguously
        int pos = new_data_capacity - keys_remaining;
        for (int j = last_position + 1; j < pos; j++) {
#if ALEX_DATA_NODE_SEP_ARRAYS
          new_key_slots[j] = it.key();
#else
          new_data_slots[j].first = it.key();
#endif
        }
        for (; pos < new_data_capacity; pos++, it++) {
#if ALEX_DATA_NODE_SEP_ARRAYS
          new_key_slots[pos] = it.key();
          new_payload_slots[pos] = it.payload();
#else
          new_data_slots[pos] = *it;
#endif
          set_bit(new_bitmap, pos);
        }
        last_position = pos - 1;
        break;
      }

      for (int j = last_position + 1; j < position; j++) {
#if ALEX_DATA_NODE_SEP_ARRAYS
        new_key_slots[j] = it.key();
#else
        new_data_slots[j].first = it.key();
#endif
      }

#if ALEX_DATA_NODE_SEP_ARRAYS
      new_key_slots[position] = it.key();
      new_payload_slots[position] = it.payload();
#else
      new_data_slots[position] = *it;
#endif
      set_bit(new_bitmap, position);

      last_position = position;

      keys_remaining--;
    }

    for (int i = last_position + 1; i < new_data_capacity; i++) {
#if ALEX_DATA_NODE_SEP_ARRAYS
      new_key_slots[i] = kEndSentinel_;
#else
      new_data_slots[i].first = kEndSentinel;
#endif
    }

#if ALEX_DATA_NODE_SEP_ARRAYS
    key_allocator().deallocate(key_slots_, data_capacity_);
    payload_allocator().deallocate(payload_slots_, data_capacity_);
#else
    value_allocator().deallocate(data_slots_, data_capacity_);
#endif
    bitmap_allocator().deallocate(bitmap_, bitmap_size_);

    data_capacity_ = new_data_capacity;
    bitmap_size_ = new_bitmap_size;
#if ALEX_DATA_NODE_SEP_ARRAYS
    key_slots_ = new_key_slots;
    payload_slots_ = new_payload_slots;
#else
    data_slots_ = new_data_slots;
#endif
    bitmap_ = new_bitmap;

    expansion_threshold_ =
        std::min(std::max(data_capacity_ * kMaxDensity_,
                          static_cast<double>(num_keys_ + 1)),
                 static_cast<double>(data_capacity_));
    contraction_threshold_ = data_capacity_ * kMinDensity_;
  }

  inline bool is_append_mostly_right() const {
    return static_cast<double>(num_right_out_of_bounds_inserts_) /
               num_inserts_ >
           kAppendMostlyThreshold;
  }

  inline bool is_append_mostly_left() const {
    return static_cast<double>(num_left_out_of_bounds_inserts_) / num_inserts_ >
           kAppendMostlyThreshold;
  }

  // Insert key into pos. The caller must guarantee that pos is a gap.
  void insert_element_at(const T& key, P payload, int pos) {
#if ALEX_DATA_NODE_SEP_ARRAYS
    key_slots_[pos] = key;
    payload_slots_[pos] = payload;
#else
    data_slots_[index] = std::make_pair(key, payload);
#endif
    set_bit(pos);

    // Overwrite preceding gaps until we reach the previous element
    pos--;
    while (pos >= 0 && !check_exists(pos)) {
      ALEX_DATA_NODE_KEY_AT(pos) = key;
      pos--;
    }
  }

  // Insert key into pos, shifting as necessary in the range [left, right)
  // Returns the actual position of insertion
  int insert_using_shifts(const T& key, P payload, int pos) {
    // Find the closest gap
    int gap_pos = closest_gap(pos);
    set_bit(gap_pos);
    if (gap_pos >= pos) {
      for (int i = gap_pos; i > pos; i--) {
#if ALEX_DATA_NODE_SEP_ARRAYS
        key_slots_[i] = key_slots_[i - 1];
        payload_slots_[i] = payload_slots_[i - 1];
#else
        data_slots_[i] = data_slots_[i - 1];
#endif
      }
      insert_element_at(key, payload, pos);
      num_shifts_ += gap_pos - pos;
      return pos;
    } else {
      for (int i = gap_pos; i < pos - 1; i++) {
#if ALEX_DATA_NODE_SEP_ARRAYS
        key_slots_[i] = key_slots_[i + 1];
        payload_slots_[i] = payload_slots_[i + 1];
#else
        data_slots_[i] = data_slots_[i + 1];
#endif
      }
      insert_element_at(key, payload, pos - 1);
      num_shifts_ += pos - gap_pos - 1;
      return pos - 1;
    }
  }

#if ALEX_USE_LZCNT
  // Returns position of closest gap to pos
  // Returns pos if pos is a gap
  int closest_gap(int pos) const {
    pos = std::min(pos, data_capacity_ - 1);
    int bitmap_pos = pos >> 6;
    int bit_pos = pos - (bitmap_pos << 6);
    if (bitmap_[bitmap_pos] == static_cast<uint64_t>(-1) ||
        (bitmap_pos == bitmap_size_ - 1 &&
         _mm_popcnt_u64(bitmap_[bitmap_pos]) ==
             data_capacity_ - ((bitmap_size_ - 1) << 6))) {
      // no gaps in this block of 64 positions, start searching in adjacent
      // blocks
      int left_bitmap_pos = 0;
      int right_bitmap_pos = ((data_capacity_ - 1) >> 6);  // inclusive
      int max_left_bitmap_offset = bitmap_pos - left_bitmap_pos;
      int max_right_bitmap_offset = right_bitmap_pos - bitmap_pos;
      int max_bidirectional_bitmap_offset =
          std::min<int>(max_left_bitmap_offset, max_right_bitmap_offset);
      int bitmap_distance = 1;
      while (bitmap_distance <= max_bidirectional_bitmap_offset) {
        uint64_t left_bitmap_data = bitmap_[bitmap_pos - bitmap_distance];
        uint64_t right_bitmap_data = bitmap_[bitmap_pos + bitmap_distance];
        if (left_bitmap_data != static_cast<uint64_t>(-1) &&
            right_bitmap_data != static_cast<uint64_t>(-1)) {
          int left_gap_pos = ((bitmap_pos - bitmap_distance + 1) << 6) -
                             static_cast<int>(_lzcnt_u64(~left_bitmap_data)) -
                             1;
          int right_gap_pos = ((bitmap_pos + bitmap_distance) << 6) +
                              static_cast<int>(_tzcnt_u64(~right_bitmap_data));
          if (pos - left_gap_pos <= right_gap_pos - pos ||
              right_gap_pos >= data_capacity_) {
            return left_gap_pos;
          } else {
            return right_gap_pos;
          }
        } else if (left_bitmap_data != static_cast<uint64_t>(-1)) {
          int left_gap_pos = ((bitmap_pos - bitmap_distance + 1) << 6) -
                             static_cast<int>(_lzcnt_u64(~left_bitmap_data)) -
                             1;
          // also need to check next block to the right
          if (bit_pos > 32 && bitmap_pos + bitmap_distance + 1 < bitmap_size_ &&
              bitmap_[bitmap_pos + bitmap_distance + 1] !=
                  static_cast<uint64_t>(-1)) {
            int right_gap_pos =
                ((bitmap_pos + bitmap_distance + 1) << 6) +
                static_cast<int>(
                    _tzcnt_u64(~bitmap_[bitmap_pos + bitmap_distance + 1]));
            if (pos - left_gap_pos <= right_gap_pos - pos ||
                right_gap_pos >= data_capacity_) {
              return left_gap_pos;
            } else {
              return right_gap_pos;
            }
          } else {
            return left_gap_pos;
          }
        } else if (right_bitmap_data != static_cast<uint64_t>(-1)) {
          int right_gap_pos = ((bitmap_pos + bitmap_distance) << 6) +
                              static_cast<int>(_tzcnt_u64(~right_bitmap_data));
          if (right_gap_pos < data_capacity_) {
            // also need to check next block to the left
            if (bit_pos < 32 && bitmap_pos - bitmap_distance > 0 &&
                bitmap_[bitmap_pos - bitmap_distance - 1] !=
                    static_cast<uint64_t>(-1)) {
              int left_gap_pos =
                  ((bitmap_pos - bitmap_distance) << 6) -
                  static_cast<int>(
                      _lzcnt_u64(~bitmap_[bitmap_pos - bitmap_distance - 1])) -
                  1;
              if (pos - left_gap_pos <= right_gap_pos - pos ||
                  right_gap_pos >= data_capacity_) {
                return left_gap_pos;
              } else {
                return right_gap_pos;
              }
            } else {
              return right_gap_pos;
            }
          }
        }
        bitmap_distance++;
      }
      if (max_left_bitmap_offset > max_right_bitmap_offset) {
        for (int i = bitmap_pos - bitmap_distance; i >= left_bitmap_pos; i--) {
          if (bitmap_[i] != static_cast<uint64_t>(-1)) {
            return ((i + 1) << 6) - static_cast<int>(_lzcnt_u64(~bitmap_[i])) -
                   1;
          }
        }
      } else {
        for (int i = bitmap_pos + bitmap_distance; i <= right_bitmap_pos; i++) {
          if (bitmap_[i] != static_cast<uint64_t>(-1)) {
            int right_gap_pos =
                (i << 6) + static_cast<int>(_tzcnt_u64(~bitmap_[i]));
            if (right_gap_pos >= data_capacity_) {
              return -1;
            } else {
              return right_gap_pos;
            }
          }
        }
      }
      return -1;
    } else {
      // search within block of 64 positions
      uint64_t bitmap_data = bitmap_[bitmap_pos];
      int closest_right_gap_distance = 64;
      int closest_left_gap_distance = 64;
      // Logically gaps to the right of pos, in the bitmap these are gaps to the
      // left of pos's bit
      // This covers the case where pos is a gap
      // For example, if pos is 3, then bitmap '10101101' -> bitmap_right_gaps
      // '01010000'
      uint64_t bitmap_right_gaps = ~(bitmap_data | ((1ULL << bit_pos) - 1));
      if (bitmap_right_gaps != 0) {
        closest_right_gap_distance =
            static_cast<int>(_tzcnt_u64(bitmap_right_gaps)) - bit_pos;
      } else if (bitmap_pos + 1 < bitmap_size_) {
        // look in the next block to the right
        closest_right_gap_distance =
            64 + static_cast<int>(_tzcnt_u64(~bitmap_[bitmap_pos + 1])) -
            bit_pos;
      }
      // Logically gaps to the left of pos, in the bitmap these are gaps to the
      // right of pos's bit
      // For example, if pos is 3, then bitmap '10101101' -> bitmap_left_gaps
      // '00000010'
      uint64_t bitmap_left_gaps = (~bitmap_data) & ((1ULL << bit_pos) - 1);
      if (bitmap_left_gaps != 0) {
        closest_left_gap_distance =
            bit_pos - (63 - static_cast<int>(_lzcnt_u64(bitmap_left_gaps)));
      } else if (bitmap_pos > 0) {
        // look in the next block to the left
        closest_left_gap_distance =
            bit_pos + static_cast<int>(_lzcnt_u64(~bitmap_[bitmap_pos - 1])) +
            1;
      }

      if (closest_right_gap_distance < closest_left_gap_distance &&
          pos + closest_right_gap_distance < data_capacity_) {
        return pos + closest_right_gap_distance;
      } else {
        return pos - closest_left_gap_distance;
      }
    }
  }
#else
  // A slower version of closest_gap that does not use lzcnt and tzcnt
  // Does not return pos if pos is a gap
  int closest_gap(int pos) const {
    int max_left_offset = pos;
    int max_right_offset = data_capacity_ - pos - 1;
    int max_bidirectional_offset =
        std::min<int>(max_left_offset, max_right_offset);
    int distance = 1;
    while (distance <= max_bidirectional_offset) {
      if (!check_exists(pos - distance)) {
        return pos - distance;
      }
      if (!check_exists(pos + distance)) {
        return pos + distance;
      }
      distance++;
    }
    if (max_left_offset > max_right_offset) {
      for (int i = pos - distance; i >= 0; i--) {
        if (!check_exists(i)) return i;
      }
    } else {
      for (int i = pos + distance; i < data_capacity_; i++) {
        if (!check_exists(i)) return i;
      }
    }
    return -1;
  }
#endif

  /*** Deletes ***/

  // Erase the left-most key with the input value
  // Returns the number of keys erased (0 or 1)
  int erase_one(const T& key) {
    int pos = find_lower(key);

    if (pos == data_capacity_ || !key_equal(ALEX_DATA_NODE_KEY_AT(pos), key))
      return 0;

    // Erase key at pos
    erase_one_at(pos);
    return 1;
  }

  // Erase the key at the given position
  void erase_one_at(int pos) {
    T next_key;
    if (pos == data_capacity_ - 1) {
      next_key = kEndSentinel_;
    } else {
      next_key = ALEX_DATA_NODE_KEY_AT(pos + 1);
    }
    ALEX_DATA_NODE_KEY_AT(pos) = next_key;
    unset_bit(pos);
    pos--;

    // Erase preceding gaps until we reach an existing key
    while (pos >= 0 && !check_exists(pos)) {
      ALEX_DATA_NODE_KEY_AT(pos) = next_key;
      pos--;
    }

    num_keys_--;

    if (num_keys_ < contraction_threshold_) {
      resize(kMaxDensity_);  // contract
      num_resizes_++;
    }
  }

  // Erase all keys with the input value
  // Returns the number of keys erased (there may be multiple keys with the same
  // value)
  int erase(const T& key) {
    int pos = upper_bound(key);

    if (pos == 0 || !key_equal(ALEX_DATA_NODE_KEY_AT(pos - 1), key)) return 0;

    // Erase preceding positions until we reach a key with smaller value
    int num_erased = 0;
    T next_key;
    if (pos == data_capacity_) {
      next_key = kEndSentinel_;
    } else {
      next_key = ALEX_DATA_NODE_KEY_AT(pos);
    }
    pos--;
    while (pos >= 0 && key_equal(ALEX_DATA_NODE_KEY_AT(pos), key)) {
      ALEX_DATA_NODE_KEY_AT(pos) = next_key;
      num_erased += check_exists(pos);
      unset_bit(pos);
      pos--;
    }

    num_keys_ -= num_erased;

    if (num_keys_ < contraction_threshold_) {
      resize(kMaxDensity_);  // contract
      num_resizes_++;
    }
    return num_erased;
  }

  // Erase keys with value between start key (inclusive) and end key.
  // Returns the number of keys erased.
  int erase_range(T start_key, T end_key, bool end_key_inclusive = false) {
    int pos;
    if (end_key_inclusive) {
      pos = upper_bound(end_key);
    } else {
      pos = lower_bound(end_key);
    }

    if (pos == 0) return 0;

    // Erase preceding positions until key value is below the start key
    int num_erased = 0;
    T next_key;
    if (pos == data_capacity_) {
      next_key = kEndSentinel_;
    } else {
      next_key = ALEX_DATA_NODE_KEY_AT(pos);
    }
    pos--;
    while (pos >= 0 &&
           key_greaterequal(ALEX_DATA_NODE_KEY_AT(pos), start_key)) {
      ALEX_DATA_NODE_KEY_AT(pos) = next_key;
      num_erased += check_exists(pos);
      unset_bit(pos);
      pos--;
    }

    num_keys_ -= num_erased;

    if (num_keys_ < contraction_threshold_) {
      resize(kMaxDensity_);  // contract
      num_resizes_++;
    }
    return num_erased;
  }

  /*** Stats ***/

  // Total size of node metadata
  long long node_size() const override { return sizeof(self_type); }

  // Total size in bytes of key/payload/data_slots and bitmap
  long long data_size() const {
    long long data_size = data_capacity_ * sizeof(T);
    data_size += data_capacity_ * sizeof(P);
    data_size += bitmap_size_ * sizeof(uint64_t);
    return data_size;
  }

  // Number of contiguous blocks of keys without gaps
  int num_packed_regions() const {
    int num_packed = 0;
    bool is_packed = check_exists(0);
    for (int i = 1; i < data_capacity_; i++) {
      if (check_exists(i) != is_packed) {
        if (is_packed) {
          num_packed++;
        }
        is_packed = !is_packed;
      }
    }
    if (is_packed) {
      num_packed++;
    }
    return num_packed;
  }

  /*** Debugging ***/

  bool validate_structure(bool verbose = false) const {
    if (this->cost_ < 0 || std::isnan(this->cost_)) {
      std::cout << "[Data node cost is invalid value]"
                << " node addr: " << this << ", node level: " << this->level_
                << ", cost: " << this->cost_ << std::endl;
      return false;
    }
    for (int i = 0; i < data_capacity_ - 1; i++) {
      if (key_greater(ALEX_DATA_NODE_KEY_AT(i), ALEX_DATA_NODE_KEY_AT(i + 1))) {
        if (verbose) {
          std::cout << "Keys should be in non-increasing order" << std::endl;
        }
        return false;
      } else if (key_less(ALEX_DATA_NODE_KEY_AT(i),
                          ALEX_DATA_NODE_KEY_AT(i + 1)) &&
                 !check_exists(i)) {
        if (verbose) {
          std::cout << "The last key of a certain value should not be a gap"
                    << std::endl;
        }
        return false;
      }
    }
    if (ALEX_DATA_NODE_KEY_AT(data_capacity_ - 1) == kEndSentinel_ &&
        check_exists(data_capacity_ - 1)) {
      if (verbose) {
        std::cout << "The sentinel should not be a valid key" << std::endl;
      }
      return false;
    }
    if (ALEX_DATA_NODE_KEY_AT(data_capacity_ - 1) != kEndSentinel_ &&
        !check_exists(data_capacity_ - 1)) {
      if (verbose) {
        std::cout << "The last key should be a valid key" << std::endl;
      }
      return false;
    }
    uint64_t num_bitmap_ones = 0;
    for (int i = 0; i < bitmap_size_; i++) {
      num_bitmap_ones += count_ones(bitmap_[i]);
    }
    if (static_cast<int>(num_bitmap_ones) != num_keys_) {
      if (verbose) {
        std::cout << "Number of ones in bitmap should match num_keys"
                  << std::endl;
      }
      return false;
    }
    return true;
  }

  // Check that a key exists in the key/data_slots
  // If validate_bitmap is true, confirm that the corresponding position in the
  // bitmap is correctly set to 1
  bool key_exists(const T& key, bool validate_bitmap) const {
    for (int i = 0; i < data_capacity_ - 1; i++) {
      if (key_equal(ALEX_DATA_NODE_KEY_AT(i), key) &&
          (!validate_bitmap || check_exists(i))) {
        return true;
      }
    }
    return false;
  }

  std::string to_string() const {
    std::string str;
    str += "Num keys: " + std::to_string(num_keys_) + ", Capacity: " +
           std::to_string(data_capacity_) + ", Expansion Threshold: " +
           std::to_string(expansion_threshold_) + "\n";
    for (int i = 0; i < data_capacity_; i++) {
      str += (std::to_string(ALEX_DATA_NODE_KEY_AT(i)) + " ");
    }
    return str;
  }
};



#if DEBUG==1
#define ASSERT(PREDICATE, ...)                                                 \
  do {                                                                          \
    if (!(PREDICATE)) {            +                                             \
      fprintf(stderr,                                                          \
              "%s:%d (%s) Assertion " #PREDICATE " failed: ", __FILE__,        \
              __LINE__, __PRETTY_FUNCTION__);                                  \
      fprintf(stderr, __VA_ARGS__);                                            \
      abort();                                                                 \
    }                                                                          \
  } while (0)
#else
#define ASSERT(PREDICATE, ...)
#endif

#define STATS 0
#define DEBUG_PRINT 0
#define ASK_ABOUT 0
#define AVX512_BROKEN 0
#define INSERT_UPDATE 0

static uint64_t debug_counter = 0;

static constexpr uint64_t one[64] = {
  1ULL << 0, 1ULL << 1, 1ULL << 2, 1ULL << 3, 1ULL << 4, 1ULL << 5, 1ULL << 6, 1ULL << 7, 1ULL << 8, 1ULL << 9,
  1ULL << 10, 1ULL << 11, 1ULL << 12, 1ULL << 13, 1ULL << 14, 1ULL << 15, 1ULL << 16, 1ULL << 17, 1ULL << 18, 1ULL << 19,
  1ULL << 20, 1ULL << 21, 1ULL << 22, 1ULL << 23, 1ULL << 24, 1ULL << 25, 1ULL << 26, 1ULL << 27, 1ULL << 28, 1ULL << 29,
  1ULL << 30, 1ULL << 31, 1ULL << 32, 1ULL << 33, 1ULL << 34, 1ULL << 35, 1ULL << 36, 1ULL << 37, 1ULL << 38, 1ULL << 39,
  1ULL << 40, 1ULL << 41, 1ULL << 42, 1ULL << 43, 1ULL << 44, 1ULL << 45, 1ULL << 46, 1ULL << 47, 1ULL << 48, 1ULL << 49,
  1ULL << 50, 1ULL << 51, 1ULL << 52, 1ULL << 53, 1ULL << 54, 1ULL << 55, 1ULL << 56, 1ULL << 57, 1ULL << 58, 1ULL << 59,
  1ULL << 60, 1ULL << 61, 1ULL << 62, 1ULL << 63};
/*
static constexpr uint64_t one[128] = {
  1ULL << 0, 1ULL << 1, 1ULL << 2, 1ULL << 3, 1ULL << 4, 1ULL << 5, 1ULL << 6, 1ULL << 7, 1ULL << 8, 1ULL << 9,
  1ULL << 10, 1ULL << 11, 1ULL << 12, 1ULL << 13, 1ULL << 14, 1ULL << 15, 1ULL << 16, 1ULL << 17, 1ULL << 18, 1ULL << 19,
  1ULL << 20, 1ULL << 21, 1ULL << 22, 1ULL << 23, 1ULL << 24, 1ULL << 25, 1ULL << 26, 1ULL << 27, 1ULL << 28, 1ULL << 29,
  1ULL << 30, 1ULL << 31, 1ULL << 32, 1ULL << 33, 1ULL << 34, 1ULL << 35, 1ULL << 36, 1ULL << 37, 1ULL << 38, 1ULL << 39,
  1ULL << 40, 1ULL << 41, 1ULL << 42, 1ULL << 43, 1ULL << 44, 1ULL << 45, 1ULL << 46, 1ULL << 47, 1ULL << 48, 1ULL << 49,
  1ULL << 50, 1ULL << 51, 1ULL << 52, 1ULL << 53, 1ULL << 54, 1ULL << 55, 1ULL << 56, 1ULL << 57, 1ULL << 58, 1ULL << 59,
  1ULL << 60, 1ULL << 61, 1ULL << 62, 1ULL << 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
*/

enum range_type {INSERTS, DELETES, BLOCK};

template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
class LeafDS {

  static constexpr bool binary = (sizeof...(Ts) == 0);

  using element_type =
      typename std::conditional<binary, std::tuple<key_type>,
                                std::tuple<key_type, Ts...>>::type;

  using value_type =
      typename std::conditional<binary, std::tuple<>, std::tuple<Ts...>>::type;

  template <int I>
  using NthType = typename std::tuple_element<I, value_type>::type;
  static constexpr int num_types = sizeof...(Ts);

//   using SOA_type = typename std::conditional<binary, SOA<key_type>,
//                                              SOA<key_type, Ts...>>::type;
  using SOA_type = typename std::conditional<binary, AOS<key_type>,
                                            AOS<key_type, Ts...>>::type;

#if AVX512
	static constexpr size_t keys_per_vector = 64 / sizeof(key_type);

	static constexpr uint32_t all_ones_vec = keys_per_vector - 1;

	using mask_type =
		typename std::conditional<sizeof(key_type) == 4, __mmask16, __mmask8>::type;

	// idk why it complained about this one
	using lr_mask_type =
		typename std::conditional<sizeof(key_type) == 4, uint16_t, uint8_t>::type;

	// TODO: can precompute the masks
	lr_mask_type get_left_mask(size_t start) {
		assert(start < keys_per_vector);
#if DEBUG_PRINT
		printf("\tget left mask start = %lu\n", start);
#endif
		lr_mask_type mask;
		if constexpr (std::is_same<key_type, uint32_t>::value) { // 32-bit keys
			mask = 0xFFFF;
		} else {
			mask = 0xFF;
		}
		mask <<= start;
		mask >>= start;
		return mask;
	}

	// TODO: can precompute the masks
	auto get_right_mask(size_t end) {
		assert(end < keys_per_vector);
#if DEBUG_PRINT
		printf("\tget right mask end = %lu\n", end);
#endif
		lr_mask_type mask;
		if constexpr (std::is_same<key_type, uint32_t>::value) { // 32-bit keys
			mask = 0xFFFF;
		} else {
			mask = 0xFF;
		}
		mask >>= end;
		mask <<= end;
		return mask;
	}


	// 64 bytes per cache line / 4 bytes per elt = 16-bit vector
	// wherever there is a match, it will set those bits
	static inline __mmask16 slot_mask_32(uint8_t * array, uint32_t key) {
		__m512i bcast = _mm512_set1_epi32(key);
		__m512i block = _mm512_loadu_si512((const __m512i *)(array));
		return _mm512_cmp_epu32_mask(bcast, block, _MM_CMPINT_EQ);
	}

	static inline __mmask8 slot_mask_64(uint8_t * array, uint64_t key) {
		__m512i bcast = _mm512_set1_epi64(key);
		__m512i block = _mm512_loadu_si512((const __m512i *)(array));
		return _mm512_cmp_epu64_mask(bcast, block, _MM_CMPINT_EQ);
	}

	static inline mask_type slot_mask(uint8_t * array, key_type key) {
		if constexpr (std::is_same<key_type, uint32_t>::value) { // 32-bit keys
			return slot_mask_32(array, key);
		} else {
			return slot_mask_64(array, key);
		}
	}
#endif

private:
	static constexpr key_type NULL_VAL = {};

	static constexpr size_t num_blocks = header_size;
	static constexpr size_t N = log_size + header_size + block_size * num_blocks;

  static_assert(N != 0, "N cannot be 0");

	// start of each section
	// insert and delete log
	static constexpr size_t header_start = log_size;
	static constexpr size_t blocks_start = header_start + header_size;

	// counters
	size_t num_inserts_in_log = 0;
	size_t num_deletes_in_log = 0;
	size_t num_elts_total = 0;

  // keep track of whether log is sorted
  bool log_is_sorted = true;

	// max elts allowed in data structure before split
	static constexpr size_t max_density = (int)( 9.0 / 10.0 * N );
	static constexpr size_t min_density = (int)( 4.0 / 10.0 * N );

	// block sorted bitmap (works with num_blocks up to 64)
	uint64_t block_sorted_bitmap = 0;		
  uint8_t count_per_block[header_size] = {0};

public:
	// TODO: NOT SAFE!! insert and delete break this because they don't increment num_elts_total until they flush
	// size_t get_num_elements() const { return num_elts_total; }

	std::array<uint8_t, SOA_type::get_size_static(N)> array = {0};

	key_type get_min_after_split() {
		return blind_read_key(header_start);
	}


  void set_block_count(size_t block_idx, uint8_t num_elts) {
    count_per_block[block_idx] = num_elts;
  }

  uint8_t get_block_count(size_t block_idx) {
    return count_per_block[block_idx];
  }

private:

  inline key_type get_key_array(uint32_t index) const {
    return std::get<0>(
        SOA_type::template get_static<0>(array.data(), N, index));
  }

	size_t count_up_elts() const {
		size_t result = 0;
		for(size_t i = 0; i < N; i++) {
			result += (blind_read_key(i) != NULL_VAL);
		}
		return result;
	}

#if STATS
  uint32_t num_redistributes = 0;
  uint32_t vol_redistributes = 0;
public:
  void report_redistributes() {
    printf("num redistributes = %u, vol redistributes %u\n", num_redistributes, vol_redistributes);
  }
#endif

public:
  key_type split(LeafDS<log_size, header_size, block_size, key_type, Ts...>* right);
  void merge(LeafDS<log_size, header_size, block_size, key_type, Ts...>* right);
  void get_max_2(key_type* max_key, key_type* second_max_key) const;
  void shift_left(LeafDS<log_size, header_size, block_size, key_type, Ts...>* right, int shiftnum);
  void shift_right(LeafDS<log_size, header_size, block_size, key_type, Ts...>* left, int shiftnum);
//   iterator lower_bound(const key_type& key);
  key_type& get_key_at_sorted_index_nonconst(size_t i);
  key_type& get_key_at_sorted_index(size_t i) const;
  key_type& get_key_at_sorted_index_with_print(size_t i) const;
  size_t get_element_at_sorted_index(size_t i);
  size_t get_num_elements() const;

private:
	// private helpers
	void update_val_at_index(element_type e, size_t index);
	void place_elt_at_index(element_type e, size_t index);
	void clear_range(size_t start, size_t end);
	size_t find_block(key_type key) const;
	size_t find_block_with_hint(key_type key, size_t hint) const;
  /*
	void global_redistribute(element_type* log_to_flush, size_t num_to_flush, unsigned short* count_per_block);
	void global_redistribute_blocks(unsigned short* count_per_block);
  */
	void global_redistribute(element_type* log_to_flush, size_t num_to_flush);
	void global_redistribute_blocks();

	void copy_src_to_dest(size_t src, size_t dest);
	void flush_log_to_blocks(size_t max_to_flush);
	void flush_deletes_to_blocks();

	void sort_log();
	void sort_range(size_t start_idx, size_t end_idx);

	// helpers for range query
	void sort_array_range(void *base_array, size_t size, size_t start_idx, size_t end_idx);
	inline bool is_in_delete_log(key_type key) const;
	auto get_sorted_block_copy(size_t block_idx) const;


	inline std::pair<size_t, size_t> get_block_range(size_t block_idx) const;

	template <range_type type>
	bool update_in_range_if_exists(size_t start, size_t end, element_type e);
	std::pair<bool, size_t> find_key_in_range(size_t start, size_t end, key_type e) const;
	bool update_in_block_if_exists(element_type e);
	bool update_in_block_if_exists(element_type e, size_t block_idx);

	uint8_t count_block(size_t block_idx) const;
	uint8_t count_larger_in_block(size_t block_idx, key_type start_key) const;

	void print_range(size_t start, size_t end) const;

	// void advance_block_ptr(size_t* blocks_ptr, size_t* cur_block, size_t* start_of_cur_block, unsigned short* count_per_block) const;
  void advance_block_ptr(size_t* blocks_ptr, size_t* cur_block, size_t* start_of_cur_block) const;


	// delete helpers
	bool delete_from_header();
	void strip_deletes_and_redistrib();
	void delete_from_block_if_exists(key_type e, size_t block_idx);

	// given a buffer of n elts, spread them evenly in the blocks
	void global_redistribute_buffer(element_type* buffer, size_t n);

	[[nodiscard]] value_type value_internal(key_type e) const;

public:
  size_t get_num_elts() const { return num_elts_total; }
  bool is_full() const { return num_elts_total >= max_density; }
  bool is_few() const { return num_elts_total <= min_density; }

  void print() const;

  [[nodiscard]] uint64_t sum_keys() const;
  [[nodiscard]] uint64_t sum_keys_with_map() const;
  [[nodiscard]] uint64_t sum_keys_direct() const;
  template <bool no_early_exit, size_t... Is, class F> bool map(F f) const;

  // main top-level functions
  // given a key, return the index of the largest elt at least e
  [[nodiscard]] uint32_t search(key_type e) const;

  // insert e, return true if it was not there
  bool insert(element_type e);

  // bulk loads into assumed empty leafds, will overwrite existing elts
  template <typename Iterator>
  void bulk_load(Iterator it, size_t num_items, key_type* max_key);

  // remove e, return true if it was present
  bool remove(key_type e);

  // true if log or relevant blocks are unsorted
  bool need_write_lock(key_type start, size_t length);

  // whether elt e was in the DS
  [[nodiscard]] bool has(key_type e) const;
  [[nodiscard]] bool has_with_print(key_type e) const;

	[[nodiscard]] auto value(key_type e) const {
		static_assert(!binary);
		if constexpr (num_types == 1) {
			return std::get<0>(value_internal(e));
		} else {
			return value_internal(e);
		}

	}

  // return the next [length] sorted elts greater than or equal to start
  template <class F>
  uint64_t sorted_range(key_type start, size_t length, F f) ;

  template <class F>
  uint64_t sorted_range_no_write(key_type start, size_t length, F f) ;

  // return sorted elts in the range [start, end] 
  template <class F>
  uint64_t sorted_range_end(key_type start, key_type end, F f) ;

  // return all elts in the range [start, end]
  template <class F>
  void unsorted_range(key_type start, key_type end, F f) ;

  [[nodiscard]] size_t get_index_in_blocks(key_type e) const;

  // index of element e in the DS, N if not found
  [[nodiscard]] size_t get_index(key_type e) const;

  auto blind_read_key(uint32_t index) const {
    return std::get<0>(SOA_type::get_static(array.data(), N, index));
  }

  // min block header is the first elt in the header part
  key_type get_min_block_key() const {
    return blind_read_key(header_start);
  }

  void blind_write_array(void* arr, size_t len, uint32_t index, element_type e)  {
    SOA_type::get_static(arr, len, index) = e;
  }

  void blind_write(element_type e, uint32_t index) {
    SOA_type::get_static(array.data(), N, index) = e;
  }

  auto blind_read(uint32_t index) const {
    return SOA_type::get_static(array.data(), N, index);
  }
  auto blind_read_array(void* arr, size_t size, uint32_t index) const {
    return SOA_type::get_static(arr, size, index);
  }

  auto blind_read_key_array(void* arr, size_t size, uint32_t index) const {
    return std::get<0>(SOA_type::get_static(arr, size, index));
  }

public:
  // Iterators

	class iterator;

	//! Constructs a read/data-write iterator that points to the first slot in
    //! the first leaf of the B+ tree.
    iterator begin() {
		// sort the leafds
        return iterator(this);
    }

    //! Constructs a read/data-write iterator that points to the first invalid
    //! slot in the last leaf of the B+ tree.
    iterator end() {
        return iterator(N, this);
    }

	iterator lower_bound(const key_type& key) {
		auto it_leafds = iterator(this);
		auto end = iterator(N, this);
		while (it_leafds != end && it_leafds.key() < key) {
			it_leafds++;
		}
		return it_leafds;
	}

	//! STL-like iterator object for B+ tree items. The iterator points to a
	//! specific slot number in a leaf.
	class iterator
	{
	public:
		// *** Types

		//! The key type of the btree. Returned by key().
		// typedef typename key_type key_type;

		// //! The value type of the btree. Returned by operator*().
		// typedef typename value_type value_type;

		//! Reference to the value_type. STL required.
		typedef value_type& reference;

		//! Pointer to the value_type. STL required.
		typedef value_type* pointer;

		//! STL-magic iterator category
		typedef std::bidirectional_iterator_tag iterator_category;

		//! STL-magic
		typedef ptrdiff_t difference_type;

		//! Our own type
		typedef iterator self;

	private:
		// *** Members

		//! Current key/data slot referenced
		size_t curr_val_ptr;
		size_t block_idx;
		bool log_only;

		LeafDS<log_size, header_size, block_size, key_type, Ts...>* leaf;
		uint8_t count_per_block[header_size];

	public:
		// *** Methods

		//! Default-Constructor of a mutable iterator
		iterator()
			: curr_val_ptr(0)
		{ 
		}

		iterator(size_t val, LeafDS<log_size, header_size, block_size, key_type, Ts...>* leafds)
			:  curr_val_ptr(val), log_only(false), block_idx(0), leaf(leafds)
		{ 
			// printf("end iter, %lu", )
		}

		iterator(LeafDS<log_size, header_size, block_size, key_type, Ts...>* leafds)
			: curr_val_ptr(0), log_only(false), block_idx(0), leaf(leafds)
		{
			leaf->sort_range(0, leaf->num_inserts_in_log); // sort inserts

			// if only log exists, just loop over log
			if (leaf->get_min_block_key() == 0) {
				log_only = true;
				curr_val_ptr = 0;
			} else { // otherwise, flush and only loop over blocks
				if(leaf->num_deletes_in_log > 0) {
					leaf->sort_range(log_size - leaf->num_deletes_in_log, log_size); // sort deletes
					if (leaf->delete_from_header()) {
						leaf->strip_deletes_and_redistrib();
					} else {
						leaf->flush_deletes_to_blocks();
					}
					leaf->num_deletes_in_log = 0;
				}

				// if inserting min, swap out the first header into the first block
				if (leaf->num_inserts_in_log > 0 && leaf->blind_read_key(0) < leaf->get_min_block_key()) {
					size_t j = leaf->blocks_start + block_size;
					// find the first zero slot in the block
					SOA_type::template map_range_with_index_static(leaf->array.data(), leaf->N, [&j](auto index, auto key, auto... values) {
						if (key == 0) {
							j = std::min(index, j);
						}
					}, leaf->blocks_start, leaf->blocks_start + block_size);

					// put the old min header in the block where it belongs
					// src = header start, dest = i
					leaf->copy_src_to_dest(leaf->header_start, j);

					// make min elt the new first header
					leaf->copy_src_to_dest(0, leaf->header_start);

					// this block is no longer sorted
					leaf->block_sorted_bitmap = leaf->block_sorted_bitmap & ~(one[0]);
				}

				assert(leaf->num_deletes_in_log == 0);
				leaf->flush_log_to_blocks(leaf->num_inserts_in_log);
				leaf->num_inserts_in_log = 0;
				leaf->clear_range(0, log_size);
				curr_val_ptr = leaf->header_start;
				log_only = false;
				
				// presort all blocks
				for (size_t i = 0; i < num_blocks; i++) {
					count_per_block[i] = leaf->count_block(i);
					if ((leaf->block_sorted_bitmap & one[i])) {
					} else {
						auto block_range = leaf->get_block_range(i);
						leaf->sort_range(block_range.first, block_range.first + count_per_block[i]);
						leaf->block_sorted_bitmap = leaf->block_sorted_bitmap | one[i];
					}
				}
			}
		}

		//! Dereference the iterator.
		auto operator * () const {
			return leaf->blind_read(curr_val_ptr);
		}

		//! Dereference the iterator.
		pointer operator -> () const {
			return &leaf->blind_read(curr_val_ptr);
		}

		//! Key of the current slot.
		const key_type& key() const {
			return std::get<0>(leaf->blind_read(curr_val_ptr));
		}

		//! Prefix++ advance the iterator to the next slot.
		iterator& operator ++ () {
			if (curr_val_ptr == leaf->N) { return *this;}

			if (log_only) {
				// printf("log inserts %lu, log only %d, curr val ptr \n", leaf->num_inserts_in_log, log_only, curr_val_ptr);
				if (curr_val_ptr + 1 < leaf->num_inserts_in_log) {
					++curr_val_ptr;
				} else {
					// this is end()
					curr_val_ptr = leaf->N;
				}
			} else {
				// increment block pointer
				if (curr_val_ptr < leaf->blocks_start && count_per_block[block_idx] > 0) {
					// in header, need to switch to block
					curr_val_ptr = leaf->blocks_start + block_idx * block_size;
				} else if (count_per_block[block_idx] == 0 || curr_val_ptr == leaf->blocks_start + block_idx * block_size + count_per_block[block_idx] - 1) {
					// at end of current block, need to switch to next block header
					block_idx++;
					if (block_idx == leaf->num_blocks) { 
						// this is end()
						curr_val_ptr = leaf->N;
						return *this; 
					}
					curr_val_ptr = leaf->header_start + block_idx;
				} else {
					// in block
					curr_val_ptr++;
				}
			}

			return *this;
		}

		//! Postfix++ advance the iterator to the next slot.
		iterator operator ++ (int) {
			iterator tmp = *this;   // copy ourselves
			if (curr_val_ptr == leaf->N) { return tmp;}
			
			if (log_only) {
				if (curr_val_ptr + 1 < leaf->num_inserts_in_log) {
					++curr_val_ptr;
				} else {
					// this is end()
					curr_val_ptr = leaf->N;
				}
			} else {
				// increment block pointer
				if (curr_val_ptr < leaf->blocks_start && count_per_block[block_idx] > 0) {
					// in header, need to switch to block
					curr_val_ptr = leaf->blocks_start + block_idx * block_size;
				} else if (count_per_block[block_idx] == 0 || curr_val_ptr == leaf->blocks_start + block_idx * block_size + count_per_block[block_idx] - 1) {
					// at end of current block, need to switch to next block header
					block_idx++;
					if (block_idx == leaf->num_blocks) { 
						// this is end()
						curr_val_ptr = leaf->N;
						return *this; 
					}
					curr_val_ptr = leaf->header_start + block_idx;
				} else {
					// in block
					curr_val_ptr++;
				}
			}

			return tmp;
		}

		//! Equality of iterators.
		bool operator == (const iterator& x) const {
			return (x.leaf == leaf) && (x.curr_val_ptr == curr_val_ptr);
		}

		//! Inequality of iterators.
		bool operator != (const iterator& x) const {
			return ((x.curr_val_ptr != curr_val_ptr) || (x.leaf != leaf));
		}
	};

};

// precondition - keys at index already match
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
void LeafDS<log_size, header_size, block_size, key_type, Ts...>::update_val_at_index(element_type e, size_t index) {
	// update its value if needed
	if constexpr (!binary) {
		if (leftshift_tuple(SOA_type::get_static(array.data(), N, index)) !=
				leftshift_tuple(e)) {
			SOA_type::get_static(array.data(), N, index) = e;
		}
	}
}

// precondition - this slot is empty
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
void LeafDS<log_size, header_size, block_size, key_type, Ts...>::place_elt_at_index(element_type e, size_t index) {
	SOA_type::get_static(array.data(), N, index) = e;
	num_elts_total++;
}

template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
void LeafDS<log_size, header_size, block_size, key_type, Ts...>::clear_range(size_t start, size_t end) {
  SOA_type::map_range_static(
      array.data(), N,
      [](auto &...args) { std::forward_as_tuple(args...) = element_type(); },
      start, end);
}


// given a merged list, put it in the DS
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... ts>
void LeafDS<log_size, header_size, block_size, key_type, ts...>::global_redistribute_buffer(element_type* buffer, size_t n) {
#if STATS
	num_redistributes++;
	vol_redistributes += N;
#endif

#if DEBUG
	assert(n < N);
#endif
  clear_range(header_start, N); // clear the header and blocks

#if DEBUG_PRINT
	printf("GLOBAL REDISTRIB BUFFER OF SIZE %lu\n", n);
#endif

  // split up the buffer into blocks
  size_t per_block = n / num_blocks;
  size_t remainder = n % num_blocks;
  size_t num_so_far = 0;
  for(size_t i = 0; i < num_blocks; i++) {
    size_t num_to_flush = per_block + (i < remainder);
#if DEBUG
    ASSERT(num_to_flush < block_size, "to flush %lu, block size %lu\n", num_to_flush, block_size);
    assert(num_to_flush >= 1);
#endif
#if DEBUG_PRINT
		printf("block %zu, num to flush %zu\n", i, num_to_flush);
#endif
    // write the header
    blind_write(buffer[num_so_far], header_start + i);
#if DEBUG_PRINT
		printf("\tset buf[%zu] = %lu as header of block %zu at pos %lu\n", num_so_far, std::get<0>(buffer[num_so_far]), i, header_start + i);
#endif
    num_to_flush--;
    num_so_far++;
    // write the rest into block
    size_t start = blocks_start + i * block_size;
    for(size_t j = 0; j < num_to_flush; j++) {
#if DEBUG
			assert(num_so_far < n);
#endif
#if DEBUG_PRINT
		printf("\tset buf[%zu] = %lu in block %zu at pos %lu\n", num_so_far, std::get<0>(buffer[num_so_far]), i, start + j);
#endif
      blind_write(buffer[num_so_far], start + j);
      num_so_far++;
    }
    assert(num_to_flush < 128);
    count_per_block[i] = (uint8_t)num_to_flush;
    assert(count_per_block[i] == count_block(i));
  }
#if DEBUG
	for (size_t i = 0; i < num_blocks; i++) {
    if (count_per_block[i] != count_block(i)) {
      printf("*** block %lu, got count %u, should be %u ***\n", i, count_per_block[i], count_block(i));
      print();
      assert(false);
    }
	}
#endif
#if DEBUG
  assert(num_so_far == n);
#endif
	num_elts_total = num_so_far + num_inserts_in_log;

#if DEBUG_PRINT
	printf("after global redistrib blocks\n");
	print();
#endif
}

// just redistrib the header/blocks
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... ts>
void LeafDS<log_size, header_size, block_size, key_type, ts...>::global_redistribute_blocks() {
// void LeafDS<log_size, header_size, block_size, key_type, ts...>::global_redistribute_blocks(unsigned short* count_per_block) {

#if STATS
	num_redistributes++;
	vol_redistributes += N - header_size;
#endif

	// sort each block
	// size_t end_blocks = 0;
	for (size_t i = 0; i < num_blocks; i++) {
		auto block_range = get_block_range(i);
		sort_range(block_range.first, block_range.first + count_per_block[i]);
	}

	// merge all elts into sorted order
	std::vector<element_type> buffer;
	size_t blocks_ptr = header_start;
	size_t cur_block = 0;
	size_t start_of_cur_block = 0;
	while(cur_block < num_blocks) {
#if DEBUG
		ASSERT(blind_read_key(blocks_ptr) != NULL_VAL, "block ptr %lu\n", blocks_ptr);
#endif
#if DEBUG_PRINT
		printf("added %lu at idx %lu to buffer\n", blind_read_key(blocks_ptr), blocks_ptr);
#endif
		buffer.push_back(blind_read(blocks_ptr));
		advance_block_ptr(&blocks_ptr, &cur_block, &start_of_cur_block);
	}
#if DEBUG
	assert(buffer.size() < num_blocks * block_size);
#endif
	num_elts_total = buffer.size();

#if DEBUG_PRINT
	printf("*** BUFFER ***\n");
	for(size_t i = 0; i < buffer.size(); i++) {
		printf("%lu\t", std::get<0>(buffer[i]));
	}
	printf("\n");
#endif

#if DEBUG
	// at this point the buffer should be sorted
	for (size_t i = 1; i < buffer.size(); i++) {
		assert(has(std::get<0>(buffer[i])));
		ASSERT(std::get<0>(buffer[i]) > std::get<0>(buffer[i-1]), "buffer[%lu] = %lu, buffer[%lu] = %lu\n", i-1, std::get<0>(buffer[i-1]), i, std::get<0>(buffer[i]));
	}
#endif

	// split up the buffer evenly amongst the rest of the blocks
	global_redistribute_buffer(buffer.data(), buffer.size());
}


// one of the blocks
// input: deduped log to flush, number of elements in the log, count of elements to flush to each block, count of elements per block
// merge all elements from blocks and log in sorted order in the intermediate buffer
// split them evenly amongst the blocks
// TODO: count global redistributes
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... ts>
// void LeafDS<log_size, header_size, block_size, key_type, ts...>::global_redistribute(element_type* log_to_flush, size_t num_to_flush, unsigned short* count_per_block) {
void LeafDS<log_size, header_size, block_size, key_type, ts...>::global_redistribute(element_type* log_to_flush, size_t num_to_flush) {

#if DEBUG
	for (size_t i = 0; i < num_blocks; i++) {
    if (count_per_block[i] != count_block(i)) {
      printf("*** block %lu, got count %u, should be %u ***\n", i, count_per_block[i], count_block(i));
      print();
      assert(false);
    }
	}
  
	// verify that the log to flush is sorted
	for(size_t i = 1; i < num_to_flush; i++) {
		assert(std::get<0>(log_to_flush[i-1]) < std::get<0>(log_to_flush[i]));
	}
#endif
	assert(num_deletes_in_log == 0);
	// sort each block
	// size_t end_blocks = 0;
	for (size_t i = 0; i < num_blocks; i++) {
		auto block_range = get_block_range(i);
		sort_range(block_range.first, block_range.first + count_per_block[i]);
	}

	assert(num_deletes_in_log == 0);

	// do a merge from sorted log and sorted blocks
	std::vector<element_type> buffer;
	size_t log_ptr = 0;
	size_t blocks_ptr = header_start;
	size_t cur_block = 0;
	size_t start_of_cur_block = 0;
	size_t log_end = num_to_flush;

#if DEBUG_PRINT
	printf("\n *** LOG TO FLUSH (size = %lu)***\n", num_to_flush);
	for(size_t i = 0; i < num_to_flush; i++) {
		printf("%lu\t", std::get<0>(log_to_flush[i]));
	}
	printf("\n\n\n");
	print();
#endif

	while(log_ptr < log_end && cur_block < num_blocks) {
		assert(blocks_ptr < N);
#if DEBUG_PRINT
		printf("log ptr %lu, cur_block %lu, blocks_ptr %lu\n", log_ptr, cur_block, blocks_ptr);
#endif
		const key_type log_key = std::get<0>(log_to_flush[log_ptr]);
		const key_type block_key = blind_read_key(blocks_ptr);
		assert(log_key != block_key);
		if (log_key < block_key) {
#if DEBUG_PRINT
			printf("pushed %lu from log to buffer\n", std::get<0>(log_to_flush[log_ptr]));
#endif
#if DEBUG
			if (buffer.size() >= 1) {
				assert(std::get<0>(buffer[buffer.size() - 1]) < std::get<0>(log_to_flush[log_ptr]));
			}
#endif
			buffer.push_back(log_to_flush[log_ptr]);
			log_ptr++;
		} else {
#if DEBUG
			assert(blocks_ptr < N);
			if (buffer.size() >= 1) {
				ASSERT(std::get<0>(buffer[buffer.size() - 1]) < blind_read_key(blocks_ptr), "buffer end %lu, blocks ptr %lu, blocks elt %lu\n", std::get<0>(buffer[buffer.size() - 1]), blocks_ptr, blind_read_key(blocks_ptr));
			}
#endif
			buffer.push_back(blind_read(blocks_ptr));
			advance_block_ptr(&blocks_ptr, &cur_block, &start_of_cur_block);
		}
	}
	assert(num_deletes_in_log == 0);

	// cleanup if necessary
	while(log_ptr < log_end) {
		buffer.push_back(log_to_flush[log_ptr]);
		log_ptr++;
	}
	assert(num_deletes_in_log == 0);

#if DEBUG_PRINT
	printf("\n*** cleaning up with blocks ***\n");
#endif
	while(cur_block < num_blocks) {
#if DEBUG
		ASSERT(blind_read_key(blocks_ptr) != NULL_VAL, "block ptr %lu\n", blocks_ptr);
#endif
#if DEBUG_PRINT
		printf("added %lu at idx %lu to buffer\n", blind_read_key(blocks_ptr), blocks_ptr);
#endif
		buffer.push_back(blind_read(blocks_ptr));
		advance_block_ptr(&blocks_ptr, &cur_block, &start_of_cur_block);
	}
#if DEBUG
	ASSERT(buffer.size() <= header_size + num_blocks * block_size, "buffer size = %lu, blocks slots = %lu\n", buffer.size(), num_blocks * block_size);
#endif
	num_elts_total = buffer.size();
	assert(num_deletes_in_log == 0);

#if DEBUG_PRINT
	printf("*** BUFFER ***\n");
	for(size_t i = 0; i < buffer.size(); i++) {
		printf("%lu\t", std::get<0>(buffer[i]));
	}
	printf("\n");
#endif

	// ASK_ABOUT
	// there is a delete being added in between this assert and the following assert, means theres something else adding to the delete log while this leafds is trying to do stuff
	assert(num_deletes_in_log == 0);

#if DEBUG
	// at this point the buffer should be sorted
	for (size_t i = 1; i < buffer.size(); i++) {
		assert(num_deletes_in_log == 0);
		if (!has(std::get<0>(buffer[i]))) {
			printf("Missing key %lu\n", std::get<0>(buffer[i]));
			print();
		}
		assert(has(std::get<0>(buffer[i])));
		ASSERT(std::get<0>(buffer[i]) > std::get<0>(buffer[i-1]), "buffer[%lu] = %lu, buffer[%lu] = %lu\n", i-1, std::get<0>(buffer[i-1]), i, std::get<0>(buffer[i]));
	}
#endif

	// we have merged in all the inserts
	num_inserts_in_log = 0;

	// split up the buffer evenly amongst the rest of the blocks
	global_redistribute_buffer(buffer.data(), buffer.size());
}

// return index of the block that this elt would fall in
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
size_t LeafDS<log_size, header_size, block_size, key_type, Ts...>::find_block(key_type key) const {
	if (key < blind_read_key(header_start)) { return 0; }
	return find_block_with_hint(key, 0);
}

// return index of the block that this elt would fall in
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
size_t LeafDS<log_size, header_size, block_size, key_type, Ts...>::find_block_with_hint(key_type key, size_t hint) const {
	if (key < blind_read_key(header_start)) { return 0; }
	assert(blind_read_key(header_start) != 0);
#if !AVX512 || DEBUG
	// scalar version for debug
	size_t i = header_start + hint;
	if (key < blind_read_key(i)) {
		print();
		ASSERT(key >= blind_read_key(i), "key = %lu, start block = %lu, header key = %lu\n", key, hint, blind_read_key(i));
	}
	size_t correct_ret = hint;

	for( ; i < blocks_start; i++) {
		correct_ret += blind_read_key(i) <= key;
	}
	if (correct_ret == num_blocks || blind_read_key(header_start + correct_ret) > key) {
		correct_ret--;
	}
#if !AVX512
	return correct_ret;
#endif
#endif
#if AVX512
	// vector version
	size_t vector_start = header_start;
	size_t vector_end = blocks_start;
	// printf("vector start = %lu, vector end = %lu, keys per vector = %lu\n", vector_start, vector_end, keys_per_vector);
	size_t ret = 0;
	mask_type mask;
	for(; vector_start < vector_end; vector_start += keys_per_vector) {
		if constexpr (std::is_same<key_type, uint32_t>::value) { // 32-bit keys
			__m512i bcast = _mm512_set1_epi32(key);
			__m512i block = _mm512_loadu_si512((const __m512i *)(array.data() + vector_start * sizeof(key_type)));
			mask = _mm512_cmple_epu32_mask(block, bcast);
		} else {
			__m512i bcast = _mm512_set1_epi64(key);
			// printf("key = %lu, vector_start = %lu, byte start = %lu\n", key, vector_start, vector_start * sizeof(key_type));

			__m512i block = _mm512_loadu_si512((const __m512i *)(array.data() + vector_start * sizeof(key_type)));
			mask = _mm512_cmple_epu64_mask(block, bcast);
			// printf("in 64-bit case, popcount got %lu\n", __builtin_popcount(mask));
			assert((size_t)(__builtin_popcount(mask)) <= keys_per_vector);
		}
		ret += __builtin_popcount(mask);
	}

	// TODO: can you do to the next line faster?
	// printf("num blocks = %lu\n", num_blocks);
	assert(ret <= num_blocks);
	if (ret == num_blocks || blind_read_key(header_start + ret) > key) {
		ret--;
	}

	ASSERT(ret == correct_ret, "searching for key %lu: got %lu, should be %lu\n", key, ret, correct_ret);
#endif
#if DEBUG
	i = header_start + hint;
	for( ; i < blocks_start; i++) {
		if(blind_read_key(i) == key)  {
			return i - header_start;
		} else if (blind_read_key(i) > key) {
			break;
		}
	}
	assert(i - header_start - 1 < num_blocks);
#if AVX512
	ASSERT(i - header_start - 1 == ret, "elt %lu, original found %lu, new is %lu\n", key, i - header_start - 1, ret);
#endif
#endif
#if AVX512
	return ret;
#endif
}

// given a src, dest indices
// move elt at src into dest
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
void LeafDS<log_size, header_size, block_size, key_type, Ts...>::copy_src_to_dest(size_t src, size_t dest) {
	SOA_type::get_static(array.data(), N, dest) =
		SOA_type::get_static(array.data(), N, src);
}


// precondition: range must be packed
// TODO: vectorized sorting
// one way would be to access the key array, sort that in a vectorized way, and apply
// the permutation vector to later value vectors
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
void LeafDS<log_size, header_size, block_size, key_type, Ts...>::sort_range(size_t start_idx, size_t end_idx) {
	assert(start_idx <= end_idx);
	auto start = typename LeafDS<log_size, header_size, block_size, key_type, Ts...>::SOA_type::Iterator(array.data(), N, start_idx);
	auto end = typename LeafDS<log_size, header_size, block_size, key_type, Ts...>::SOA_type::Iterator(array.data(), N, end_idx);

	std::sort(start, end, [](auto lhs, auto rhs) { return std::get<0>(typename SOA_type::T(lhs)) < std::get<0>(typename SOA_type::T(rhs)); } );
// #if ASK_ABOUT
#if DEBUG
	// check sortedness
	for(size_t i = start_idx + 1; i < end_idx; i++) {
		if (blind_read_key(i-1) >= blind_read_key(i)) {
			print();
		}
		ASSERT(blind_read_key(i-1) < blind_read_key(i), "i: %lu, key at i - 1: %lu, key at i: %lu, num_dels: %lu", i, blind_read_key(i-1), blind_read_key(i), num_deletes_in_log);
		assert(blind_read_key(i-1) < blind_read_key(i));

	}
#endif
// #endif
}

template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
void LeafDS<log_size, header_size, block_size, key_type, Ts...>::sort_array_range(void *base_array, size_t size, size_t start_idx, size_t end_idx) {
	auto start = typename LeafDS<log_size, header_size, block_size, key_type, Ts...>::SOA_type::Iterator(base_array, size, start_idx);
	auto end = typename LeafDS<log_size, header_size, block_size, key_type, Ts...>::SOA_type::Iterator(base_array, size, end_idx);

	std::sort(start, end, [](auto lhs, auto rhs) { return std::get<0>(typename SOA_type::T(lhs)) < std::get<0>(typename SOA_type::T(rhs)); } );
}

// sort the log
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
void LeafDS<log_size, header_size, block_size, key_type, Ts...>::sort_log() {
	sort_range(0, num_inserts_in_log);
}

// if you expect at most 1 occurrence, use compare to 0
// then use tzcnt to tell you the index of the first 1

// if you could possibly have more than 1 match:
// use popcount to tell you how many matches there are
// __builtin_popcountll(mask64)
// if yes, use select to find the index
static inline uint8_t word_select(uint64_t val, int rank) {
  val = _pdep_u64(one[rank], val);
  return _tzcnt_u64(val);
}

// given a range [start, end), look for elt e
// if e is in the range, update it and return true
// otherwise return false
// also return index found or index stopped at
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
template <range_type type>
bool LeafDS<log_size, header_size, block_size, key_type, Ts...>::update_in_range_if_exists(size_t start, size_t end, element_type e) {
#if DEBUG_PRINT
	printf("*** start = %lu, end = %lu ***\n", start, end);
#endif

	[[maybe_unused]] bool correct_answer = false;
	[[maybe_unused]] size_t test_i = start;
#if !AVX512_BROKEN || DEBUG
	// scalar version for correctness
	const key_type key = std::get<0>(e);

	for(; test_i < end; test_i++) {
		if (key == get_key_array(test_i)) {
		#if DEBUG_PRINT
			printf("CORRECT FOUND KEY %lu AT IDX %zu\n", std::get<0>(e), test_i);
			print();
		#endif
			correct_answer = true;
			break;
		} else if (get_key_array(test_i) == NULL_VAL) {
			correct_answer = false;
			break;
		}
	}
#if !AVX512_BROKEN
	return correct_answer;
#endif
#endif
#if AVX512_BROKEN
	// vector version
	if constexpr (type == BLOCK) {
		assert(start % keys_per_vector == 0);
		assert(end % keys_per_vector == 0);

		for(size_t vector_start = start; vector_start < end; vector_start += keys_per_vector) {
			mask_type mask = slot_mask(array.data() + vector_start * sizeof(key_type), std::get<0>(e));
			if (mask > 0) {
				auto i = vector_start + _tzcnt_u64(mask);

				assert(test_i == i);
				assert((correct_answer == true));
#if INSERT_UPDATE
				update_val_at_index(e, i);
#endif
				return true;
			}
		}
		return false;
	}

	// if you are in the log
	size_t vector_start = start & ~(all_ones_vec);
	size_t vector_end = (end + all_ones_vec) & ~(all_ones_vec);
#if DEBUG_PRINT
	printf("\tvector start %lu, vector end %lu\n", vector_start, vector_end);
#endif
	mask_type mask = slot_mask(array.data() + vector_start * sizeof(key_type), std::get<0>(e));

	if constexpr (type == DELETES) {
		// TODO: check that this is right
		lr_mask_type left_mask = get_right_mask(start % keys_per_vector);
		mask &= left_mask;
		if (mask > 0) {
			auto idx = vector_start + _tzcnt_u64(mask); // check if there is an off-by-1 in tzcnt

			ASSERT(test_i == idx, "test_i = %zu, i = %llu, mask = %d\n", test_i, idx, mask);
			assert((correct_answer == true));
#if INSERT_UPDATE
			update_val_at_index(e, idx);
#endif
			return true;
		}
		vector_start += keys_per_vector;
	}

	if (vector_end < keys_per_vector) {
		assert((correct_answer == false));
		return false;
	}

	// do blocks or any full blocks of the log
	// will miss the last full block, if there is one
	while(vector_start + keys_per_vector <= end) {
		auto mask = slot_mask(array.data() + vector_start * sizeof(key_type), std::get<0>(e));
		if (mask > 0) {
			auto i = vector_start + _tzcnt_u64(mask);

			assert(test_i == i);
			assert((correct_answer == true));
#if DEBUG_PRINT
			printf("\tVECTOR LOOP FOUND AT IDX %lu\n", i);
#endif
#if INSERT_UPDATE
			update_val_at_index(e, i);
#endif
			return true;
		}
		vector_start += keys_per_vector;
	}
#if DEBUG_PRINT
	printf("\tvector start %lu, end %lu\n", vector_start, end);
#endif

	// do the remaining ragged right, if there is any.
	if constexpr (type == INSERTS) {
#if DEBUG_PRINT
		printf("\tINSERTS: vector start %lu, end %lu\n", vector_start, end);
#endif
		if (vector_start < end) { // this is not exactly optimal, TBD how to remove it
			// vector fills from the right (tzcnt), so use get_left_mask
			lr_mask_type left_mask = get_left_mask(end % keys_per_vector);
			mask = slot_mask(array.data() + vector_start * sizeof(key_type), std::get<0>(e));
#if DEBUG_PRINT
			printf("\tleft mask %u, mask %u\n", left_mask, mask);
#endif
			mask &= left_mask;
			if (mask > 0) {
				auto i = vector_start + _tzcnt_u64(mask);
				assert(test_i == i);
#if DEBUG_PRINT
				printf("\tVECTOR END FOUND AT IDX %lu\n", i);
#endif
#if INSERT_UPDATE
				update_val_at_index(e, i);
#endif
				assert((correct_answer == true));
				return true;
			}
		}
	}

	ASSERT((correct_answer == false), "searching for key %lu in range [%lu, %lu)\n", std::get<0>(e), start, end);
	return false;
#endif
}

// (only used in delete)
// given a range [start, end), look for elt e
// if e is in the range, update it and return true
// otherwise return false
// also return index found or index stopped at
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
std::pair<bool, size_t> LeafDS<log_size, header_size, block_size, key_type, Ts...>::find_key_in_range(size_t start,
	size_t end, key_type key) const {
	size_t i = start;

	for(; i < end; i++) { // TODO: vectorize this for loop using AVX-512
		// if found, update the val and return
		if (key == get_key_array(i)) {
			return {true, i};
		} else if (get_key_array(i) == NULL_VAL) {
			return {false, i};
		}

	}
	return {false, end};
}


// given a block index, return its range [start, end)
// TODO: precompute this in a table
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
inline std::pair<size_t, size_t> LeafDS<log_size, header_size, block_size, key_type, Ts...>::get_block_range(size_t block_idx) const {
	size_t block_start = blocks_start + block_idx * block_size;
	size_t block_end = block_start + block_size;
#if DEBUG
	ASSERT(block_idx < num_blocks, "block idx %lu\n", block_idx);
	assert(block_start < N);
	assert(block_end <= N);
#endif
	return {block_start, block_end};
}

// count up the number of elements in this b lock
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
uint8_t LeafDS<log_size, header_size, block_size, key_type, Ts...>::count_larger_in_block(size_t block_idx, key_type start_key) const {
	size_t block_start = blocks_start + block_idx * block_size;
	size_t block_end = block_start + block_size;

	// count number of nonzero elts in this block
	uint8_t correct_count = 0;
	SOA_type::template map_range_static(array.data(), N, [&correct_count, &start_key](auto key, auto... values) {correct_count += key >= start_key;}, block_start, block_end);
	return correct_count;
}


// count up the number of elements in this b lock
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
uint8_t LeafDS<log_size, header_size, block_size, key_type, Ts...>::count_block(size_t block_idx) const {
	size_t block_start = blocks_start + block_idx * block_size;
	size_t block_end = block_start + block_size;

#if !AVX512 || DEBUG
	// count number of nonzero elts in this block
	uint8_t correct_count = 0;
	SOA_type::template map_range_static(array.data(), N, [&correct_count](auto key, auto... values) {correct_count += key != 0;}, block_start, block_end);
#if !AVX512
	return correct_count;
#endif
#endif
#if AVX512
	uint64_t num_zeroes = 0;
	mask_type mask;
	for(; block_start < block_end; block_start += keys_per_vector) {
		if constexpr (std::is_same<key_type, uint32_t>::value) { // 32-bit keys
			__m512i bcast = _mm512_set1_epi32(0);
			__m512i block = _mm512_loadu_si512((const __m512i *)(array.data() + block_start * sizeof(key_type)));
			mask = _mm512_cmpeq_epu32_mask(block, bcast);
		} else {
			__m512i bcast = _mm512_set1_epi64(0);
			__m512i block = _mm512_loadu_si512((const __m512i *)(array.data() + block_start * sizeof(key_type)));
			mask = _mm512_cmpeq_epu64_mask(block, bcast);
		}
		num_zeroes += __builtin_popcount(mask);
	}

	ASSERT(num_zeroes <= block_size, "counted zeroes %lu, block size %lu\n", num_zeroes, block_size);

	uint64_t count = block_size - num_zeroes;

	ASSERT(correct_count == count, "counting block %lu, got count %lu, should be %lu\n", block_idx, count, correct_count);
	return count;
#endif
}

// flush the log to the blocks
// precondition: log has been sorted
// precondition: deletes are flushed
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
void LeafDS<log_size, header_size, block_size, key_type, Ts...>::flush_log_to_blocks(size_t max_to_flush) {
#if DEBUG
	for (size_t i = 0; i < num_blocks; i++) {
    if (count_per_block[i] != count_block(i)) {
      printf("*** block %lu, got count %u, should be %u ***\n", i, count_per_block[i], count_block(i));
      print();
      assert(false);
    }
	}
#endif
#if DEBUG_PRINT
	printf("BEFORE FLUSH\n");
	print();
#endif
	assert(num_deletes_in_log == 0);
	// dedup the log wrt the blocks, putting all new elts in log_to_flush
	element_type log_to_flush[log_size];
	unsigned short num_to_flush = 0;
	unsigned short num_to_flush_per_block[num_blocks];
	size_t hint = 0;
	memset(num_to_flush_per_block, 0, num_blocks * sizeof(unsigned short));

	// look for the sorted log in the blocks
	for(size_t i = 0; i < max_to_flush; i++) {
		key_type key_to_flush = blind_read_key(i);

#if DEBUG
		ASSERT(key_to_flush >= blind_read_key(header_start), "key to flush = %lu at index in log i = %zu, first header = %lu\n", key_to_flush, i, blind_read_key(header_start));
#endif

		size_t block_idx = find_block_with_hint(key_to_flush, hint);
#if DEBUG
		assert(block_idx < num_blocks);
#endif
		// if it is in the header, update the header
		if (blind_read_key(header_start + block_idx) == key_to_flush) {
#if INSERT_UPDATE
				copy_src_to_dest(i, header_start + block_idx);
#endif
#if DEBUG_PRINT
				printf("found duplicate in header idx %zu of elt %lu\n", block_idx, key_to_flush);
#endif
				num_elts_total--;
		} else {
			// otherwise, look for it in the block
			auto update_block = update_in_block_if_exists(blind_read(i), block_idx);
			// update hint
      hint = block_idx;
			// if (hint < block_idx) { hint = block_idx; }

			// if it was in the block, do nothing bc we have already updated it
			// if not found, add to the deduped log_to_flush
			if (!update_block) {
				// if not found, the second thing in the pair is the index at the end
#if DEBUG_PRINT
				printf("\tflushing elt %lu to block %lu, header %lu\n", blind_read_key(i), block_idx, blind_read_key(header_start + block_idx));
				printf("set log_to_flush[%hu] = %lu\n", num_to_flush, blind_read_key(i));
#endif
				log_to_flush[num_to_flush] = blind_read(i);
				num_to_flush++;
				num_to_flush_per_block[block_idx]++;
				// set's block_idx's bit to 0
				block_sorted_bitmap = block_sorted_bitmap & ~(one[block_idx]);
			} else {
				num_elts_total--;
#if DEBUG_PRINT
				printf("found duplicate in block %lu of elt %zu\n", block_idx, key_to_flush);
				printf("num elts now %zu\n", num_elts_total);
#endif
			}
		}
	}

  // TODO: keep this at top level / make short-> uint8_t
	// count the number of elements in each block
	// unsigned short count_per_block[num_blocks];

	// TODO: merge these loops and count the rest in global redistribute
  /*
	for (size_t i = 0; i < num_blocks; i++) {
		count_per_block[i] = count_block(i);
	}
  */

	// if any of them overflow, redistribute
	// TODO: can vectorize this part
	bool need_global_redistrubute = false;
	for (size_t i = 0; i < num_blocks; i++) {
		if (count_per_block[i] + num_to_flush_per_block[i] >= block_size) {
			need_global_redistrubute = true;
#if DEBUG_PRINT
			printf("*** GLOBAL REDISTRIB FROM OVERFLOW ***\n");
			printf("at block %lu, count_per_block = %u, num to flush = %u\n", i, count_per_block[i], num_to_flush_per_block[i]);
#endif
		}
	}
	if (need_global_redistrubute) {
		assert(num_deletes_in_log == 0);
    global_redistribute(log_to_flush, num_to_flush);
		// global_redistribute(log_to_flush, num_to_flush, count_per_block);
		// blocks are sorted after global redist
		block_sorted_bitmap = ~(0ULL);
		return; // log gets taken care of in global redistribute
	}

	// otherwise, flush the log to the blocks
	size_t idx_in_log = 0;
	for(size_t i = 0; i < num_blocks; i++) {
		// pointer to start of block
		size_t write_start = blocks_start + i * block_size + count_per_block[i];
		for(size_t j = 0; j < num_to_flush_per_block[i]; j++) {
			blind_write(log_to_flush[idx_in_log + j], write_start + j);
			block_sorted_bitmap = block_sorted_bitmap & ~(one[i]);
		}
    count_per_block[i] += num_to_flush_per_block[i];
#if DEBUG
    if (count_per_block[i] != count_block(i)) {
      printf("*** block %lu, got count %u, should be %u ***\n", i, count_per_block[i], count_block(i));
      print();
      assert(false);
    }
#endif
		idx_in_log += num_to_flush_per_block[i];
	}
#if DEBUG
	for (size_t i = 0; i < num_blocks; i++) {
    if (count_per_block[i] != count_block(i)) {
      printf("*** block %lu, got count %u, should be %u ***\n", i, count_per_block[i], count_block(i));
      print();
      assert(false);
    }
	}
#endif
#if DEBUG
	ASSERT(idx_in_log == num_to_flush, "flushed %lu, should have flushed %u\n", idx_in_log, num_to_flush);
#endif

#if DEBUG_PRINT
	printf("AFTER FLUSH\n");
	print();
#endif
}

template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
bool LeafDS<log_size, header_size, block_size, key_type, Ts...>::update_in_block_if_exists(element_type e) {
	const key_type key = std::get<0>(e);
	// if key is in the current range of this node, try to find it in the block
	auto block_idx = find_block(key);
	auto block_range = get_block_range(block_idx);
	// if found, update and return
	return update_in_range_if_exists<BLOCK>(block_range.first, block_range.second, e);
}

// take in the block idx
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
bool LeafDS<log_size, header_size, block_size, key_type, Ts...>::update_in_block_if_exists(element_type e, size_t block_idx) {
	// if key is in the current range of this node, try to find it in the block
	auto block_range = get_block_range(block_idx);
	// if found, update and return
	return update_in_range_if_exists<BLOCK>(block_range.first, block_range.second, e);
}

// take in the block idx
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
void LeafDS<log_size, header_size, block_size, key_type, Ts...>::delete_from_block_if_exists(key_type e, size_t block_idx) {
	// if key is in the current range of this node, try to find it in the block
	auto block_range = get_block_range(block_idx);
	// if found, shift everything left by 1
	for(size_t i = block_range.first; i < block_range.second; i++) {
		if(blind_read_key(i) == e) {
#if DEBUG_PRINT
			printf("found elt %zu to delete in block %zu, idx %zu\n", e, block_idx, i);
#endif
			for(size_t j = i; j < block_range.second - 1; j++) {
				SOA_type::get_static(array.data(), N, j) = SOA_type::get_static(array.data(), N, j+1);
			}

			// TODO: is there a better way to clear a single element at the end?
			clear_range(block_range.second - 1, block_range.second);
			break;
		}
	}
}

// return true if the element was inserted, false otherwise
// may return true if the element was already there due to it being a pseudo-set
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
bool LeafDS<log_size, header_size, block_size, key_type, Ts...>::insert(element_type e) {
	assert(std::get<0>(e) != 0);
	// first try to update the key if it is in the log
	auto result = update_in_range_if_exists<INSERTS>(0, num_inserts_in_log, e);
	if (result) {
		return false;
	}

#if DEBUG
	// there should always be space in the log
	assert(num_inserts_in_log + num_deletes_in_log < log_size);
#endif

	// if not found, add it to the log
	blind_write(e, num_inserts_in_log);
	num_elts_total++; // num elts (may be duplicates in log and block)
	num_inserts_in_log++;
  log_is_sorted = false;

	if (num_inserts_in_log + num_deletes_in_log == log_size) { // we filled the log
		sort_range(0, num_inserts_in_log); // sort inserts

		// if this is the first time we are flushing the log, just make the sorted log the header
		if (get_min_block_key() == 0) {
#if DELETES
			if (num_deletes_in_log > 0) { // if we cannot fill the header
				clear_range(num_inserts_in_log, log_size); // clear deletes
				num_deletes_in_log = 0;
				return true;
			} else {	
#endif
  #if DEBUG_PRINT
				printf("\nmake sorted log the header\n");
				print();
	#endif

				for(size_t i = 0; i < log_size; i++) {
					SOA_type::get_static(array.data(), N, i + header_start) =
						SOA_type::get_static(array.data(), N, i);
				}
#if DELETES
			}
#endif
		} else { // otherwise, there are some elements in the block / header part
#if DELETES
			assert(num_inserts_in_log > 0);
			if(num_deletes_in_log > 0) {
				sort_range(log_size - num_deletes_in_log, log_size); // sort deletes
				if (delete_from_header()) {
					strip_deletes_and_redistrib();
				} else {
					flush_deletes_to_blocks();
				}
				num_deletes_in_log = 0;
			}
#endif

			// if inserting min, swap out the first header into the first block
			if (num_inserts_in_log > 0 && blind_read_key(0) < get_min_block_key()) {
        // printf("swap key %lu for min, current min %lu\n", blind_read_key(0), get_min_block_key());
#if DEBUG
		    size_t i = blocks_start;
				for(; i < blocks_start + block_size; i++) {
					if (blind_read_key(i) == 0) {
						break;
					}
				}
#endif

				size_t j = blocks_start + block_size;
				// find the first zero slot in the block
				// TODO: this didn't work (didn't find the first zero)
				SOA_type::template map_range_with_index_static(array.data(), N, [&j](auto index, auto key, auto... values) {
					if (key == 0) {
						j = std::min(index, j);
					}
				}, blocks_start, blocks_start + block_size);

				ASSERT(i == j, "got %zu, should be %zu\n", j, i);
				assert(i < blocks_start + block_size);

				// put the old min header in the block where it belongs
				// src = header start, dest = i
				copy_src_to_dest(header_start, j);

				// make min elt the new first header
				copy_src_to_dest(0, header_start);

				// this block is no longer sorted
				block_sorted_bitmap = block_sorted_bitmap & ~(one[0]);
				num_elts_total++;
        count_per_block[0]++;
        assert(count_per_block[0] == count_block(0));
			}

			assert(num_deletes_in_log == 0);
			flush_log_to_blocks(num_inserts_in_log);
		}

		// clear log
		num_inserts_in_log = 0;
		clear_range(0, log_size);
    log_is_sorted = true;
	}

	return true;
}

// bulk loads into assumed empty leafds, will overwrite existing elts
// assumes no duplicates
// returns max key
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
template <typename Iterator>
void LeafDS<log_size, header_size, block_size, key_type, Ts...>::bulk_load(Iterator it, size_t num_items, key_type* max_key) {
	assert(num_items < header_size + header_size * block_size);
	
	clear_range(0, N); // clear the leaf

	// split up the buffer into blocks
	size_t per_block = num_items / num_blocks;
	size_t remainder = num_items % num_blocks;
	size_t num_so_far = 0;
	// printf("\tPer block %lu, remainder %lu\n", per_block, remainder);
	for(size_t i = 0; i < num_blocks; i++) {
		size_t num_to_flush = per_block + (i < remainder);

		// write the header
		blind_write(*it, header_start + i);
		*max_key = blind_read_key(header_start + i);
		// printf("\tWrote header block %lu, it =%lu, val = %lu\n", i, it - ibegin, std::get<0>(*it));
		num_to_flush--;
		num_so_far++;
		++it;

		// write the rest into block
		size_t start = blocks_start + i * block_size;
		for(size_t j = 0; j < num_to_flush; j++) {

			blind_write(*it, start + j);
			*max_key = blind_read_key(start + j);
			num_so_far++;
			++it;

		}
	}
	assert(num_so_far == num_items);
	return;
}


template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
inline void LeafDS<log_size, header_size, block_size, key_type, Ts...>::advance_block_ptr(size_t* blocks_ptr, size_t* cur_block, size_t* start_of_cur_block) const {
// inline void LeafDS<log_size, header_size, block_size, key_type, Ts...>::advance_block_ptr(size_t* blocks_ptr, size_t* cur_block, size_t* start_of_cur_block, unsigned short* count_per_block) const {
#if DEBUG_PRINT
		if (blind_read_key(*blocks_ptr) == NULL_VAL) {
			printf("null blocks ptr %zu\n", *blocks_ptr);
			assert(false);
		}
		printf("\tpushed %lu from blocks to buffer\n", blind_read_key(*blocks_ptr));
#endif
#if DEBUG
		size_t prev_blocks_ptr = *blocks_ptr;
#endif

		// if we are in the header, go to the block if the block is nonempty
		if (*blocks_ptr < blocks_start) {
			*start_of_cur_block = blocks_start + (*cur_block) * block_size;

			if (blind_read_key(*start_of_cur_block) != NULL_VAL) {
				*blocks_ptr = *start_of_cur_block;
			} else { // if this block is empty, move to the next header
				(*cur_block)++;
				(*blocks_ptr)++;
			}
		} else if (*blocks_ptr == *start_of_cur_block + count_per_block[*cur_block] - 1) {
			// if we have merged in this entire block, go back to the header
			(*cur_block)++;
			*blocks_ptr = header_start + *cur_block;
		} else { // if we are still in this block, keep going
			(*blocks_ptr)++;
		}
#if DEBUG
		assert(prev_blocks_ptr != *blocks_ptr); // made sure we advanced
#endif
}

// precondition: we are not deleting from the header
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
void LeafDS<log_size, header_size, block_size, key_type, Ts...>::flush_deletes_to_blocks() {
#if DEBUG_PRINT
	printf("flushing deletes, num deletes = %lu\n", num_deletes_in_log);
#endif
	size_t hint = 0;
	// process the deletes

#if ASK_ABOUT
	// sorting sometimes doesn't work for the first key, what should be a larger key ends up in the first position
	key_type prev_key = 0;
	key_type prev_first = blind_read_key(log_size - num_deletes_in_log);
	sort_range(log_size - num_deletes_in_log, log_size);
	key_type curr_first = blind_read_key(log_size - num_deletes_in_log);
	if (prev_first != curr_first) {
		printf("sorting didnt work, found %lu instead of %lu at start of del log\n", prev_first, curr_first);
		print();
	}
#endif
	sort_range(log_size - num_deletes_in_log, log_size);

	for(size_t i = log_size - num_deletes_in_log; i < log_size; i++) {
		key_type key_to_delete = blind_read_key(i);
#if DEBUG_PRINT
		printf("\tflushing delete %lu\n", key_to_delete);
#endif
#if ASK_ABOUT
		if (prev_key && key_to_delete <= prev_key) {
			printf("prev_key : %lu is larger than key_to_delete %lu at index %i \n", prev_key, key_to_delete, i);
			print();
		}
		prev_key = key_to_delete;
#endif
		size_t block_idx = find_block_with_hint(key_to_delete, hint);
#if DEBUG_PRINT
		printf("\tflushing delete %lu to block %lu\n", key_to_delete, block_idx);
#endif
		// try to delete it from the blocks if it exists
		delete_from_block_if_exists(key_to_delete, block_idx);

		if (hint < block_idx) { hint = block_idx; }
	}

	// clear delete log
	clear_range(log_size - num_deletes_in_log, log_size);

	// count the blocks
	// unsigned short count_per_block[num_blocks];
	bool redistribute = false;
	for (size_t i = 0; i < num_blocks; i++) {
    		count_per_block[i] = count_block(i);
		if (count_per_block[i] == 0) { redistribute = true; }
	}

	// if any of the blocks become empty, do a global redistribute
	if (redistribute) {
	  // just redistribute the header/blocks
	  global_redistribute_blocks();
	}
}

template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
void LeafDS<log_size, header_size, block_size, key_type, Ts...>::strip_deletes_and_redistrib() {
  // count the number of elements in each block
#if DEBUG_PRINT
  printf("\nstrip deletes and redistrib, num deletes = %lu\n", num_deletes_in_log);
  print();
#endif
  // unsigned short count_per_block[num_blocks];
  size_t total_count = 0;
  for (size_t i = 0; i < num_blocks; i++) {
    count_per_block[i] = count_block(i);
    total_count += count_per_block[i];
  }

  // sort the blocks
  for (size_t i = 0; i < num_blocks; i++) {
    auto block_range = get_block_range(i);
    sort_range(block_range.first, block_range.first + count_per_block[i]);
  }

  // merge into buffer
  element_type buffer[total_count + header_size];

  // sort delete log
  // printf("sort range [%lu, %lu)\n", log_size - num_deletes_in_log, log_size);
//   sort_range(log_size - num_deletes_in_log, log_size);
#if DEBUG
  for(size_t i = log_size - num_deletes_in_log + 1; i < log_size; i++) {
		if (blind_read_key(i) <= blind_read_key(i-1)) {
			print();
		}
		ASSERT(blind_read_key(i) > blind_read_key(i-1), "del log unsorted with %lu elems, i-1: %lu with key %lu, i: %lu with key %lu", num_deletes_in_log, i-1, blind_read_key(i-1), i, blind_read_key(i));
  }
#endif

  // two-finger strip of log from blocks/header
  size_t log_ptr = log_size - num_deletes_in_log;
  size_t blocks_ptr = header_start;
  size_t cur_block = 0;
  size_t start_of_cur_block = 0;
  size_t log_end = log_size;
  size_t buffer_ptr = 0;

  while(log_ptr < log_end && cur_block < num_blocks) {
    const key_type log_key = blind_read_key(log_ptr);
    const key_type block_key = blind_read_key(blocks_ptr);
    // if we are deleting this key
    if (log_key == block_key) {
#if DEBUG_PRINT
	printf("\tstrip %lu from log\n", log_key);
#endif
	log_ptr++;
	advance_block_ptr(&blocks_ptr, &cur_block, &start_of_cur_block);
	// increment block pointer
      } else if (log_key < block_key) {
	log_ptr++;
      } else { // merge in elts that we are keeping
	buffer[buffer_ptr++] = blind_read(blocks_ptr);
	advance_block_ptr(&blocks_ptr, &cur_block, &start_of_cur_block);
      }
    }

    // cleanup by merging in rest of DS
    while(cur_block < num_blocks) {
      buffer[buffer_ptr++] = blind_read(blocks_ptr);
      advance_block_ptr(&blocks_ptr, &cur_block, &start_of_cur_block);
    }

    // we have removed all the deletes. there may still be stuff in the insert log.

    // num elts total is num_insert_log + from blocks
    // but there might be some repetitions between log/blocks
    // split up buffer into blocks if there is enough
    if (buffer_ptr > header_size) {
#if DEBUG_PRINT
      printf("large buffer case\n");
#endif
      global_redistribute_buffer(buffer, buffer_ptr);
    } else { // otherwise, put it in the header
#if DEBUG_PRINT
	printf("before small buffer case, buffer_ptr = %lu\n", buffer_ptr);
	print();
#endif
	    // make sure to delete the repetitions if you are going to the log
	    sort_log();

	    // merge in inserts log
	    element_type buffer2[num_inserts_in_log + buffer_ptr];
	    log_ptr = 0;
	    log_end = num_inserts_in_log;
	    auto buffer_end = buffer_ptr;
	    buffer_ptr = 0;
	    size_t out_ptr = 0;
	    while(log_ptr < log_end && buffer_ptr < buffer_end) {
		    const key_type buffer_key = std::get<0>(buffer[buffer_ptr]);
		    const key_type log_key = blind_read_key(log_ptr);
		    // if they are equal, the one the ds is newest
		    if (log_key == buffer_key) {
			    buffer2[out_ptr++] = blind_read(log_ptr);
			    log_ptr++;
			    buffer_ptr++;
		    } else if (log_key > buffer_key) {
			    buffer2[out_ptr++] = buffer[buffer_ptr++];
		    } else {
			    buffer2[out_ptr++] = blind_read(log_ptr);
			    log_ptr++;
		    }
	    }
	    // finish up
	    while(log_ptr < log_end) {
		    buffer2[out_ptr++] = blind_read(log_ptr);
		    log_ptr++;
	    }
	    while(buffer_ptr < buffer_end) {
		    buffer2[out_ptr++] = buffer[buffer_ptr++];
	    }

		if (out_ptr <= 32) {
			printf("outptr <=32 %lu", out_ptr);
		}

	    num_deletes_in_log = 0;
	    clear_range(0, N);
	    printf("num left after merging log and block = %lu\n", out_ptr);
	    if (out_ptr < log_size) { // if they can all fit in the log
		    size_t i = 0;

		    for(; i < out_ptr; i++) {
			    place_elt_at_index(buffer2[i], i);
		    }
		    num_inserts_in_log = i;
		    num_elts_total = i;
	    } else { //otherwise, put the first some into the headers and the rest into log
		    size_t i = 0;
		    for(; i < log_size; i++) {
			    place_elt_at_index(buffer2[i], header_start + i);
		    }
		    for(; i < out_ptr; i++) {
			    place_elt_at_index(buffer2[i], i - log_size);
		    }
		    num_elts_total = out_ptr;
		    num_inserts_in_log = out_ptr - log_size;
	    }

#if DEBUG_PRINT
	    printf("\nput the rest into the insert log\n");
	    print();
#endif
    }
}

// return whether we are deleting from the header or not
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
bool LeafDS<log_size, header_size, block_size, key_type, Ts...>::delete_from_header() {

	// rebuild if we are deleting from the header
	size_t log_ptr = log_size - num_deletes_in_log;
	size_t header_ptr = header_start;

	while(log_ptr < log_size && header_ptr < header_start + header_size) {
		if(blind_read_key(header_ptr) == blind_read_key(log_ptr)) {
			return true;
		} else if (blind_read_key(header_ptr) > blind_read_key(log_ptr)) {
			log_ptr++;
		} else {
			header_ptr++;
		}
	}
	return false;
}

// return true if the element was deleted, false otherwise
// may return false even if the elt is there due to it being a pseudo-set
// return N if not found, otherwise return the slot the key is at
// TODO: what if key_type is not elt_type? do you make a fake elt?
// TODO: this breaks get_num_elements() prior to flushing
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
bool LeafDS<log_size, header_size, block_size, key_type, Ts...>::remove(key_type e) {

#if DEBUG
	assert(num_deletes_in_log + num_inserts_in_log < log_size);
#endif
	// delete log requires element_type
	element_type elt_e;
	std::get<0>(elt_e) = e;

	// check if the element is in the insert log
	bool was_insert = false;
	for(size_t i = 0; i < num_inserts_in_log; i++) {
		// if so, shift everything left by 1 (cancel the insert)
		if (blind_read_key(i) == e) {
			was_insert = true;
#if DEBUG_PRINT
			printf("\tfound in insert log\n");
#endif
			for(size_t j = i; j < num_inserts_in_log - 1; j++) {
				SOA_type::get_static(array.data(), N, j) = SOA_type::get_static(array.data(), N, j+1);
			}
			clear_range(num_inserts_in_log - 1, num_inserts_in_log);
			num_inserts_in_log--;
			break;
		}
	}

	// if it was not an insert, look through the delete log for it
	if(!was_insert) {
		auto result = find_key_in_range(log_size - num_deletes_in_log, log_size, e);
		if (!result.first) { // if not in delete log, add it
			num_deletes_in_log++;
			blind_write(elt_e, log_size - num_deletes_in_log); // grow left
		} else {
			// printf("\tfound in delete log\n");
		}
	} else { // otherwise, just add it to the delete log
		auto also_in_del_log = find_key_in_range(log_size - num_deletes_in_log, log_size, e);
		if (also_in_del_log.first) { // if in delete log previously, don't read it
			// printf("triggered edge case woo \n");
		} else { // otherwise add it
			num_deletes_in_log++;
			blind_write(elt_e, log_size - num_deletes_in_log);
		}
	}

	// now check if the log is full
	if (num_deletes_in_log + num_inserts_in_log == log_size) {
#if DEBUG_PRINT
		printf("flushing delete log because full\n");
		print();
#endif
		// if the header is empty, the deletes just disappear
		// only do the flushing if there is stuff later in the DS
#if DEBUG_PRINT
		printf("\tmin block key = %lu\n", get_min_block_key());
#endif
		if (get_min_block_key() != 0) {
			sort_range(log_size - num_deletes_in_log, log_size);
			// if we are deleting from the header, do a global rewrite
			if (delete_from_header()) {
#if DEBUG_PRINT
				printf("deleting from header\n");
#endif
				// strip deletes and redistrib
				strip_deletes_and_redistrib();
			} else {
#if DEBUG_PRINT
				printf("\tflush deletes to blocks\n");
#endif
				flush_deletes_to_blocks();
			}
		}

		clear_range(log_size - num_deletes_in_log, log_size);
		num_deletes_in_log = 0;
	}
	return true; // ?
}

// return N if not found, otherwise return the slot the key is at
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
inline size_t LeafDS<log_size, header_size, block_size, key_type, Ts...>::get_index_in_blocks(key_type e) const {
	// if there is no header / stuff in blocks, key is not there
	// if less than current min, should not be in ds
	if (blind_read_key(header_start) == 0 ) {
		assert(blind_read_key(header_start + 1) == 0);
		return N;
	}
	if (e < blind_read_key(header_start)) {
		return N;
	}
	assert( e >= blind_read_key(header_start) );
	size_t block_idx = find_block(e);
#if DEBUG_PRINT
	printf("\tin has for elt %lu, find block returned %lu\n", e, block_idx);
#endif
	// check the header
	assert(e >= blind_read_key(header_start + block_idx));
	if (e == blind_read_key(header_start + block_idx)) {
		return header_start + block_idx;
	}

	// check the block
	// TODO: vectorize this search
	auto range = get_block_range(block_idx);

#if DEBUG_PRINT
	printf("\tblock range [%lu, %lu)\n", range.first, range.second);
#endif
	for(size_t i = range.first; i < range.second; i++) {
		if (blind_read_key(i) == NULL_VAL) {
			return N;
		}
		if (blind_read_key(i) == e) {
			return i;
		}
	}
	return N;
}


// return N if not found, otherwise return the slot the key is at
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
size_t LeafDS<log_size, header_size, block_size, key_type, Ts...>::get_index(key_type e) const {
	// check the log
	// TODO: vectorize the search in the log
	// replace it with the vectorized search update_if_exists
	for(size_t i = 0; i < num_inserts_in_log; i++) {
		if(e == blind_read_key(i)) {
			return i;
		}
	}

	return get_index_in_blocks(e);
}

// return true iff element exists in the data structure
// bug: insert 100 | del 100 in log, insert overrides delete so early exit is bad, 100 should exist
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
bool LeafDS<log_size, header_size, block_size, key_type, Ts...>::has(key_type e) const {
	// first check if it is in the delete log

#if DEBUG_PRINT
	printf("in has for %lu\n", e);
#endif
	bool in_del_log = false;
	for(size_t i = log_size - 1; i > log_size - num_deletes_in_log - 1; i--) {
		if(blind_read_key(i) == e) {
#if DEBUG_PRINT
			printf("\tfound %lu in delete log\n", e);
			print();
#endif
			// return false;
			in_del_log = true;
		}
	}

	// otherwise search in insert log and rest of DS
	auto idx = get_index(e);
	if (idx < num_inserts_in_log) {
		return true;
	}
	return (idx != N && !in_del_log);
}
template <size_t log_size, size_t header_size, size_t block_size,
          typename key_type, typename... Ts>
typename LeafDS<log_size, header_size, block_size, key_type, Ts...>::value_type
LeafDS<log_size, header_size, block_size, key_type, Ts...>::value_internal(
    key_type e) const {
			// first check if it is in the delete log
  for (size_t i = log_size - 1; i > log_size - num_deletes_in_log - 1; i--) {
    if (blind_read_key(i) == e) {
      return {};
    }
  }
  auto idx = get_index(e);
	if (idx == N) {
		return {};
	}
	return leftshift_tuple(blind_read(idx));
}

// return true iff element exists in the data structure
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
bool LeafDS<log_size, header_size, block_size, key_type, Ts...>::has_with_print(key_type e) const {
	// first check if it is in the delete log
	for(size_t i = log_size; i > log_size - num_deletes_in_log; i--) {
		if(blind_read_key(i) == e) {
			printf("found %u in delete log\n", e);
			print();
			return false;
		}
	}

	// otherwise search in insert log and rest of DS
	auto idx = get_index(e);
	if (idx == N) { printf("%lu not found\n", e); print(); }
	return (idx != N);
}
// print the range [start, end)
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
void LeafDS<log_size, header_size, block_size, key_type, Ts...>::print_range(size_t start, size_t end) const {
	SOA_type::map_range_with_index_static(
			(void *)array.data(), N,
			[](size_t index, key_type key, auto... args) {
				if (key != NULL_VAL) {
					if constexpr (binary) {
						std::cout << key << ", ";
					} else {
						std::cout << "((_" << index << "_)" << key << ", ";
						((std::cout << ", " << args), ...);
						std::cout << "), ";
					}
				} else {
					std::cout << "_" << index << "_,";
				}
			},
			start, end);
	printf("\n");
}

// print the entire thing
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
void LeafDS<log_size, header_size, block_size, key_type, Ts...>::print() const {
  auto num_elts = count_up_elts();
  printf("total num elts via count_up_elts %lu\n", num_elts);
  printf("total num elts %lu\n", num_elts_total);
  printf("num inserts in log = %lu\n", num_inserts_in_log);
  printf("num deletes in log = %lu\n", num_deletes_in_log);
  SOA_type::print_type_details();

	if (num_elts == 0) {
    printf("the ds is empty\n");
  }

	printf("\nlog: \n");
	print_range(0, log_size);

	printf("\nheaders:\n");
	print_range(header_start, blocks_start);

	for (uint32_t i = blocks_start; i < N; i += block_size) {
		printf("\nblock %lu (header = %lu)\n", (i - blocks_start) / block_size, blind_read_key(header_start + (i - blocks_start) / block_size));
		print_range(i, i + block_size);
	}
	printf("\n");
}

// apply the function F to the entire data structure
// most general map function without inverse
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
template <bool no_early_exit, size_t... Is, class F>
bool LeafDS<log_size, header_size, block_size, key_type, Ts...>::map(F f) const {
  // read-only version of map
  // for each elt in the log, search for it in the blocks
  // if it was found in the blocks, add the index of it into the duplicates list
  size_t skip_index[log_size];
  size_t num_to_skip = 0;

#if DEBUG_PRINT
  print();
#endif
  // add inserts to remove, if any
  for(size_t i = 0; i < num_inserts_in_log; i++) {
      key_type key = blind_read_key(i);
      size_t idx = get_index_in_blocks(key);
      if (idx < N) {
#if DEBUG_PRINT
	printf("\tskip %lu from insert log\n", key);
#endif
	skip_index[num_to_skip] = idx;
	num_to_skip++;
      }
  }
  // add deletes to skip, if any
  for(size_t i = log_size - 1; i > log_size - num_deletes_in_log - 1; i--) {
    key_type key = blind_read_key(i);
    size_t idx = get_index_in_blocks(key);
    if (idx < N) {
#if DEBUG_PRINT
      printf("\tskip %lu from delete log\n", key);
#endif
      skip_index[num_to_skip] = idx;
      num_to_skip++;
    }
  }

  static_assert(std::is_invocable_v<decltype(&F::operator()), F &, uint32_t,
                                    NthType<Is>...>,
                "update function must match given types");

  // map over insert log
  for (size_t i = 0; i < num_inserts_in_log; i++) {
    auto element =
	    SOA_type::template get_static<0, (Is + 1)...>(array.data(), N, i);
    if constexpr (no_early_exit) {
	    std::apply(f, element);
    } else {
	    if (std::apply(f, element)) {
		    return true;
	    }
    }
  }

  // map over the rest after the delete log
  for (size_t i = header_start; i < N; i++) {
    auto index = get_key_array(i);
    // skip over deletes
    if (index != NULL_VAL) {
			// skip if duplicated
			bool skip = false;
			for(size_t j = 0; j < num_to_skip; j++) {
				if(i == skip_index[j]) {
					skip = true;
#if DEBUG_PRINT
					printf("skip elt %lu at idx %lu\n", index, i);
#endif
				}
			}
	if(skip) { continue; }

      auto element =
          SOA_type::template get_static<0, (Is + 1)...>(array.data(), N, i);
      if constexpr (no_early_exit) {
        std::apply(f, element);
      } else {
        if (std::apply(f, element)) {
          return true;
        }
      }
    }
  }
  return false;
}

template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
uint64_t LeafDS<log_size, header_size, block_size, key_type, Ts...>::sum_keys_with_map() const {
  uint64_t result = 0;
  map<true>([&](key_type key) { result += key; });
  return result;
}

template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
uint64_t LeafDS<log_size, header_size, block_size, key_type, Ts...>::sum_keys_direct() const {
#if DEBUG_PRINT
  printf("*** sum with subtraction ***\n");
#endif
  uint64_t result = 0;

  size_t skip_index[log_size];
  size_t num_to_skip = 0;

  // add inserts to remove, if any
  for(size_t i = 0; i < num_inserts_in_log; i++) {
    key_type key = blind_read_key(i);
    size_t idx = get_index_in_blocks(key);
    if (idx < N) {
	    skip_index[num_to_skip] = idx;
	    num_to_skip++;
    }
  }
  // add deletes to skip, if any
  for(size_t i = log_size - 1; i > log_size - num_deletes_in_log - 1; i--) {
    key_type key = blind_read_key(i);
    size_t idx = get_index_in_blocks(key);
    if (idx < N) {
#if DEBUG_PRINT
	    printf("\tskip key %lu from deletes\n", key);
#endif
	    skip_index[num_to_skip] = idx;
	    num_to_skip++;
    }
  }
  assert(num_to_skip < log_size);

  // do inserts
  for (size_t i = 0; i < num_inserts_in_log; i++) {
    result += blind_read_key(i);
  }

  for (size_t i = header_start; i < N; i++) {
    result += blind_read_key(i);
  }

  for(size_t i = 0; i < num_to_skip; i++) {
    result -= blind_read_key(skip_index[i]);
  }
  return result;
}


// return true if the given key is going to be deleted
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
inline bool LeafDS<log_size, header_size, block_size, key_type, Ts...>::is_in_delete_log(key_type key) const {
	for(size_t j = log_size - 1; j > log_size - 1 - num_deletes_in_log; j--) {
		if(blind_read_key(j) == key) { return true; }
	}
	return false;
}

template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
auto LeafDS<log_size, header_size, block_size, key_type, Ts...>::get_sorted_block_copy(size_t block_idx) const {
	assert(block_idx < num_blocks);
#if DEBUG_PRINT
	printf("get sorted block copy of block %lu with header %lu\n", block_idx, blind_read_key(header_start + block_idx));
#endif
	size_t elts_in_block_copy = 0;
	// now find the corresponding block
	std::array<uint8_t, SOA_type::get_size_static(block_size)> block_copy = {0};

	// copy only if its not in the delete log
	if (!is_in_delete_log(blind_read_key(header_start + block_idx))) {
#if DEBUG_PRINT
		printf("\tcopy[0] = %lu\n", blind_read_key(header_start + block_idx));
#endif
		SOA_type::get_static(block_copy.data(), block_size, 0) = blind_read(header_start + block_idx);
		
		elts_in_block_copy++;
	}

	size_t elts_in_block = count_block(block_idx);
	// now copy in the rest of the block
	for(size_t i = 0; i < elts_in_block; i++) {
		// copy only if its not in the delete log
		size_t idx = blocks_start + block_idx * block_size + i;
		key_type block_key = blind_read_key(idx);
#if DEBUG_PRINT
		printf("elt %lu in block %lu\n", block_key, block_idx);
#endif
		if(!is_in_delete_log(block_key)) {
#if DEBUG_PRINT
			printf("\tcopy[%lu] = %lu\n", elts_in_block_copy, block_key);
#endif
			SOA_type::get_static(block_copy.data(), block_size, elts_in_block_copy) = blind_read(idx);
			elts_in_block_copy++;
		}
	}
#if DEBUG_PRINT
	printf("elts in copy = %lu\n", elts_in_block_copy);
#endif
	// sort_array_range(block_copy.data(), block_size, 0, elts_in_block_copy);
	auto start_sort = typename LeafDS<log_size, header_size, block_size, key_type, Ts...>::SOA_type::Iterator(block_copy.data(), block_size, 0);
	auto end_sort = typename LeafDS<log_size, header_size, block_size, key_type, Ts...>::SOA_type::Iterator(block_copy.data(), block_size, elts_in_block_copy);
	std::sort(start_sort, end_sort, [](auto lhs, auto rhs) { return std::get<0>(typename SOA_type::T(lhs)) < std::get<0>(typename SOA_type::T(rhs)); } );

	return std::make_pair(block_copy, elts_in_block_copy);
}

// return a vector of element type
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
template <class F>
void LeafDS<log_size, header_size, block_size, key_type, Ts...>::unsorted_range(key_type start, key_type end, F f) {
#if DEBUG_PRINT
	printf("\n\n*** unsorted range [%lu, %lu] ***\n", start, end);
#endif
	// first apply it to everything in the insert log
	if (start == end) {
		return;
	}

  // TODO: change this to copy the elements in the log that fall in the range [start, end)
  std::array<uint8_t, SOA_type::get_size_static(log_size)> log_copy = {0};
  
  for(uint32_t i = 0; i < num_inserts_in_log; i++) {
    blind_write_array(log_copy.data(), log_size, i, blind_read(i));
  }

  auto start_sort = typename LeafDS<log_size, header_size, block_size, key_type, Ts...>::SOA_type::Iterator(log_copy.data(), log_size, 0);
  auto end_sort = typename LeafDS<log_size, header_size, block_size, key_type, Ts...>::SOA_type::Iterator(log_copy.data(), log_size, num_inserts_in_log);
  std::sort(start_sort, end_sort, [](auto lhs, auto rhs) { return std::get<0>(typename SOA_type::T(lhs)) < std::get<0>(typename SOA_type::T(rhs)); } );

  // log copy should be sorted
#if DEBUG
  for(uint32_t i = 1; i < num_inserts_in_log; i++) {
    assert(blind_read_key_array(log_copy.data(), log_size, i-1) < blind_read_key_array(log_copy.data(), log_size, i));
  }
#endif
	// do in place sort of log
	// sort_range(0, num_inserts_in_log);
#if DELETES
	sort_range(log_size - num_deletes_in_log, log_size);
#endif

	// pre-process per block if something exists in the range, and the number of elts in the log that fall in the range
	uint8_t log_per_block[num_blocks] = {0};
#if DELETES
	unsigned short deletes_per_block[num_blocks] = {0};
#endif

	size_t block_idx = 0;
	key_type block_start = blind_read_key(header_start + block_idx);
	key_type block_end = blind_read_key(header_start + block_idx + 1);
	size_t i = 0;
	while(i < num_inserts_in_log) {
		key_type log_key = blind_read_key_array(log_copy.data(), log_size, i); // blind_read_key(i);
		// need to do the min_key check in log on first block
		if ((log_key >= block_start && log_key < block_end) || (block_idx == 0 && log_key < block_end)) {
			log_per_block[block_idx]++;
			// TODO: make this do the function thing
			if (log_key >= start && log_key < end) {
				// output.push_back(blind_read_array(log_copy.data(), log_size, i));
				std::apply(f, blind_read_array(log_copy.data(), log_size, i));
			}
			i++;
		} else { // otherwise, advance the block
			ASSERT(log_key > block_end, "log_key = %lu, block_end = %lu\n", log_key, block_end);
			block_idx++;
			block_start = blind_read_key(header_start + block_idx);
			if (block_idx == num_blocks - 1) {
				block_end = std::numeric_limits<key_type>::max();
			} else {
				block_end = blind_read_key(header_start + block_idx + 1);
			}
		}
	}

	i = 0;
	block_start = blind_read_key(header_start + block_idx);
	block_end = blind_read_key(header_start + block_idx + 1);
#if DELETES
	while(i < num_deletes_in_log) {
		key_type delete_key = blind_read_key(log_size - num_deletes_in_log + i);
		if (delete_key >= block_start && delete_key < block_end) {
			deletes_per_block[block_idx]++;
			i++;
		} else { // otherwise, advance the block
			assert(delete_key > block_end);
			block_idx++;

			block_start = blind_read_key(header_start + block_idx);
			if (block_idx == num_blocks - 1) {
				block_end = std::numeric_limits<key_type>::max();
			} else {
				block_end = blind_read_key(header_start + block_idx + 1);
			}
		}
	}
#endif
	// do prefix sum
#if DEBUG_PRINT
	printf("\tlog_per_block[0] = %hu, delete_per_block[0] = %hu\n", log_per_block[0], deletes_per_block[0]);
#endif
	for(i = 1; i < num_blocks; i++) {
		log_per_block[i] += log_per_block[i-1];
#if DEBUG_PRINT
		printf("\tlog_per_block[%lu] = %hu, deletes_per_block[%lu] = %hu\n", i, log_per_block[i], i, deletes_per_block[i]);
#endif
#if DELETES
		deletes_per_block[i] += deletes_per_block[i-1];
#endif
	}
	assert(log_per_block[num_blocks - 1] == num_inserts_in_log);
#if DELETES
	assert(deletes_per_block[num_blocks - 1] == num_deletes_in_log);
#endif

	// go through blocks, only checking for skips if there was anything in the block
	size_t start_block = find_block(start);
	size_t end_block = find_block(end);
#if DEBUG_PRINT
	printf("start block = %lu, start header = %lu, end block  = %lu, end header = %lu\n", start_block, blind_read_key(header_start + start_block), end_block, blind_read_key(header_start + end_block));
#endif
	size_t delete_start, delete_end, log_start, log_end, j;
	for(block_idx = start_block; (block_idx <= end_block && block_idx < num_blocks); block_idx++) {
		assert(count_block(block_idx) == count_per_block[block_idx]);
    auto num_elts_in_block = count_per_block[block_idx];
		auto block_start = get_block_range(block_idx).first;
#if DEBUG_PRINT
		printf("block idx = %lu, num elts = %hu\n", block_idx, num_elts_in_block);
#endif
		size_t log_overlap = log_per_block[block_idx];
		assert(log_overlap <= num_inserts_in_log);
#if DELETES
		size_t delete_overlap = deletes_per_block[block_idx];
		assert(delete_overlap <= num_deletes_in_log);
#endif
		if (block_idx > 0) {
			log_overlap = log_per_block[block_idx] - log_per_block[block_idx - 1];
#if DELETES
			delete_overlap = deletes_per_block[block_idx] - deletes_per_block[block_idx - 1];
#endif
		}
#if DEBUG_PRINT
		printf("log overlap %lu, delete overlap %lu\n", log_overlap, delete_overlap);
#endif
		if (log_overlap) {
#if DELETES
			if (delete_overlap) {
				delete_start = 0;
				delete_end = deletes_per_block[block_idx];
				log_start = 0;
				log_end = log_per_block[block_idx];

				if (block_idx > 0) {
					log_start = log_per_block[block_idx-1];
					delete_start = deletes_per_block[block_idx-1];
				}
				// check against both deletes and log
				for(i = block_start; i < block_start + num_elts_in_block; i++) {
					key_type key = blind_read_key(i);
					for(j = log_start; j < log_end; j++) {
						if (blind_read_key(j) == key) {
							continue;
						}
					}
					for(j = delete_start; j < delete_end; j++) {
						if (blind_read_key(log_size - num_deletes_in_log + j) == key) {
							continue;
						}
					}
					// output.push_back(blind_read(i));
					std::apply(f, blind_read(i));
				}
			} else {
#endif
				// just check against log
				// start and end in log
				log_start = 0;
				log_end = log_per_block[block_idx];
				if (block_idx > 0) { log_start = log_per_block[block_idx-1]; }
#if DEBUG_PRINT
				printf("\tlog range [%lu, %lu)\n", log_start, log_end);
#endif
				assert(log_end <= log_size);
				assert(log_start <= log_size);
				// add header
				bool add_header = false;

				// add if it is in the range
				if (blind_read_key(header_start + block_idx) >= start && blind_read_key(header_start + block_idx) < end) {
					add_header = true;
					for(j = log_start; j < log_end; j++) {
						if (blind_read_key_array(log_copy.data(), log_size, j) == blind_read_key(header_start + block_idx)) {
							add_header = false;
						}
					}
				}
				if (add_header) { 
					std::apply(f, blind_read(header_start + block_idx));
				}
				// then do the rest of the block
				for(i = block_start; i < block_start + num_elts_in_block; i++) {
					key_type key = blind_read_key(i);
#if DEBUG_PRINT
					printf("\t\tkey = %lu\n", key);
#endif
					for(j = log_start; j < log_end; j++) {
						if (blind_read_key_array(log_copy.data(), log_size, j) == key) {
							continue;
						}
					}
					if(key >= start && key < end) {
#if DEBUG_PRINT
						printf("\t\t\tadd %lu to output, not found in log\n", key);
#endif
						// output.push_back(blind_read(i));
						std::apply(f, blind_read(i));
					}
				}
#if DELETES
			}
#endif
		} else {
			assert(!log_overlap);
#if DELETES
			if (delete_overlap) { // check against deletes
				delete_start = 0;
				delete_end = deletes_per_block[block_idx];
				if (block_idx > 0) { delete_start = deletes_per_block[block_idx-1]; }
				for(i = block_start; i < block_start + num_elts_in_block; i++) {
					key_type key = blind_read_key(i);
					for(j = delete_start; j < delete_end; j++) {
						if (blind_read_key(log_size - num_deletes_in_log + j) == key) {
							continue;
						}
					}
					if (blind_read_key(i) >= start && blind_read_key(i) < end) {
						// output.push_back(blind_read(i));
						std::apply(f, blind_read(i));
					}
				}
			} else { // no overlap with either insert or delete log
#endif
				// first do the header
				if (blind_read_key(header_start + block_idx) >= start && blind_read_key(header_start + block_idx) < end) { 
					// output.push_back(blind_read(header_start + block_idx)); 
					std::apply(f, blind_read(header_start + block_idx));
				}
				// then do the rest of the block
				for(i = block_start; i < block_start + num_elts_in_block; i++) {
					if(blind_read_key(i) >= start && blind_read_key(i) < end) {
#if DEBUG_PRINT
						printf("\tno overlap, adding %lu to output\n", blind_read_key(i));
#endif
						// output.push_back(blind_read(i));
						std::apply(f, blind_read(i));
					}
				}
#if DELETES
			}
#endif
		}
	}
}

void printBits(size_t const size, void const * const ptr)
{
    unsigned char *b = (unsigned char*) ptr;
    unsigned char byte;
    int i, j;
    
    for (i = size-1; i >= 0; i--) {
        for (j = 7; j >= 0; j--) {
            byte = (b[i] >> j) & 1;
            printf("%u", byte);
        }
    }
    puts("");
}

template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
bool LeafDS<log_size, header_size, block_size, key_type, Ts...>::need_write_lock(key_type start, size_t length) {
  // if the log is sorted, and the remainder of the blocks after the one the map starts at are sorted, we dont need the write log
  // printf("\ncheck if need write lock for query start %lu, length %lu\n", start, length);
#if DEBUG
  if (log_is_sorted) {
    for(size_t i = 1; i < num_inserts_in_log; i++) {
      assert(blind_read_key(i-1) < blind_read_key(i));
    }
  }
#endif

  // in the first block, count all the elts that fall in the range
	size_t block_idx = find_block(start);
  size_t num_so_far = count_larger_in_block(block_idx, start);
  assert(num_so_far <= count_per_block[block_idx]);
  size_t end_blk = block_idx + 1;
  
  while(num_so_far < length && end_blk < num_blocks) {
    // printf("\tnum so far up to block %lu = %lu\n", end_blk, num_so_far);
    num_so_far += count_per_block[end_blk];
    assert(count_per_block[end_blk] == count_block(end_blk));
    // printf("\tblock %lu had %hhu, new total = %lu\n", end_blk, count_per_block[end_blk], num_so_far);
    end_blk++;
  }

  ASSERT(num_so_far >= length || end_blk == num_blocks, "start = %lu, length = %lu, num so far = %lu\n", start, length, num_so_far);
  assert(end_blk <= num_blocks);
  // printf("\tstart block %lu, end block %lu, num so far %lu\n", block_idx, end_blk, num_so_far);  
  uint64_t check_bits_set = ~(0ULL);
  size_t num_to_clear_on_right = block_idx;
  size_t num_to_clear_on_left = (num_blocks - end_blk) + (64 - num_blocks);
  assert(num_to_clear_on_right < 64);
  assert(num_to_clear_on_left < 64);

  // TODO: get this to check only the ones that you need
  check_bits_set >>= num_to_clear_on_right;
  check_bits_set <<= num_to_clear_on_right;
  check_bits_set <<= num_to_clear_on_left;
  check_bits_set >>= num_to_clear_on_left;

  uint64_t temp_bitmap = block_sorted_bitmap;
  // printf("bitmap before shift\n");
  // printBits(sizeof(temp_bitmap), &temp_bitmap);

  temp_bitmap >>= num_to_clear_on_right;
  temp_bitmap <<= num_to_clear_on_right;
  temp_bitmap <<= num_to_clear_on_left;
  temp_bitmap >>= num_to_clear_on_left;

  bool log_needs_write = !log_is_sorted;
  bool blocks_needs_write = check_bits_set != temp_bitmap;
#if DEBUG_PRINT
  printf("\tlog needs write %u, blocks needs write %u\n", log_needs_write, blocks_needs_write);
  printBits(sizeof(check_bits_set), &check_bits_set);
  printBits(sizeof(temp_bitmap), &temp_bitmap);
#endif
  return log_needs_write || blocks_needs_write;
}

// return a vector of element type
// TODO: make this apply function f to everything in the range
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
template <class F>
uint64_t LeafDS<log_size, header_size, block_size, key_type, Ts...>::sorted_range(key_type start, size_t length, F f) {

#if DEBUG_PRINT
  printf("sorted range yes write with start key %lu, length %lu\n", start, length); 
#endif
	uint64_t num_applied = 0;

	if (length == 0) {
		return num_applied;
	}

	sort_range(0, num_inserts_in_log);
  log_is_sorted = true;

	size_t log_ptr = 0;
	while(log_ptr < num_inserts_in_log && blind_read_key(log_ptr) < start) {
		log_ptr++;
	}

	// find and sort current block
	size_t block_idx = find_block(start);
	size_t elts_in_block_copy = count_block(block_idx);

	if ((block_sorted_bitmap & one[block_idx])) {
#if DEBUG
		// if bitmap says its sorted, make sure
		for(size_t i = blocks_start + block_idx * block_size + 1; i < blocks_start + block_idx * block_size + elts_in_block_copy; i++) {
			if (blind_read_key(i-1) >= blind_read_key(i)) {
				print();
			}
			ASSERT(blind_read_key(i-1) < blind_read_key(i), "i: %lu, key at i - 1: %lu, key at i: %lu, num_dels: %lu \n bitmap = %lu , blockidx = %lu \n\n", i, blind_read_key(i-1), blind_read_key(i), num_deletes_in_log, block_sorted_bitmap, block_idx);
			
		}
#endif
	} else {
		sort_range(blocks_start + block_idx * block_size, blocks_start + block_idx * block_size + elts_in_block_copy);
		block_sorted_bitmap = block_sorted_bitmap | one[block_idx];
	}

	// start block_ptr at header of current block
	size_t block_ptr = header_start + block_idx;

	// move block ptr to location of >= start key
	while(blind_read_key(block_ptr) < start && block_idx < num_blocks) {
		if (block_ptr < blocks_start && elts_in_block_copy > 0) {
			// in header, need to switch to block
			block_ptr = blocks_start + block_idx * block_size;
		} else if (elts_in_block_copy == 0 || block_ptr == blocks_start + block_idx * block_size + elts_in_block_copy - 1) {
			// at end of current block, need to switch to next block header
			block_idx++;
			if (block_idx == num_blocks) { break; }

			elts_in_block_copy = count_block(block_idx);

			if ((block_sorted_bitmap & one[block_idx])) {
#if DEBUG
				// if bitmap says its sorted, make sure
				for(size_t i = blocks_start + block_idx * block_size + 1; i < blocks_start + block_idx * block_size + elts_in_block_copy; i++) {
					ASSERT(blind_read_key(i-1) < blind_read_key(i), "i: %lu, key at i - 1: %lu, key at i: %lu, num_dels: %lu", i, blind_read_key(i-1), blind_read_key(i), num_deletes_in_log);
				}
#endif
			} else {
				sort_range(blocks_start + block_idx * block_size, blocks_start + block_idx * block_size + elts_in_block_copy);
				block_sorted_bitmap = block_sorted_bitmap | one[block_idx];
			}
			block_ptr = header_start + block_idx;
		} else {
			// in block
			block_ptr++;
		}
	}

	while(log_ptr < num_inserts_in_log && block_idx < num_blocks) {
		key_type log_key = blind_read_key(log_ptr);
		key_type block_key = blind_read_key(block_ptr);
		// printf("num applied = %lu, log_key = %lu, block_key = %lu, length = %lu\n", num_applied, log_key, block_key, length);
		assert(num_applied < length);
		assert(log_key >= start);
		assert(block_key >= start);
		if (log_key == block_key) { // duplicate in log and blocks
			// log is the more recent one
			std::apply(f, blind_read(log_ptr));
			log_ptr++;
			num_applied++;

			// increment block pointer too
			if (block_ptr < blocks_start && elts_in_block_copy > 0) {
				// in header, need to switch to block
				block_ptr = blocks_start + block_idx * block_size;
			} else if (elts_in_block_copy == 0 || block_ptr == blocks_start + block_idx * block_size + elts_in_block_copy - 1) {
				// at end of current block, need to switch to next block header
				block_idx++;
				if (block_idx == num_blocks) { break; }
				elts_in_block_copy = count_block(block_idx);
				if ((block_sorted_bitmap & one[block_idx])) {
#if DEBUG
					// if bitmap says its sorted, make sure
					for(size_t i = blocks_start + block_idx * block_size + 1; i < blocks_start + block_idx * block_size + elts_in_block_copy; i++) {
						ASSERT(blind_read_key(i-1) < blind_read_key(i), "i: %lu, key at i - 1: %lu, key at i: %lu, num_dels: %lu", i, blind_read_key(i-1), blind_read_key(i), num_deletes_in_log);
					}
#endif
				} else {
					sort_range(blocks_start + block_idx * block_size, blocks_start + block_idx * block_size + elts_in_block_copy);
					block_sorted_bitmap = block_sorted_bitmap | one[block_idx];
				}
				block_ptr = header_start + block_idx;
			} else {
				// in block
				block_ptr++;
			}

		} else if (log_key < block_key) {
			std::apply(f, blind_read(log_ptr));
			log_ptr++;
			num_applied++;
		} else {
			std::apply(f, blind_read(block_ptr));

			if (block_ptr < blocks_start && elts_in_block_copy > 0) {
				// in header, need to switch to block
				block_ptr = blocks_start + block_idx * block_size;
			} else if (elts_in_block_copy == 0 || block_ptr == blocks_start + block_idx * block_size + elts_in_block_copy - 1) {
				// at end of current block, need to switch to next block header
				block_idx++;
				if (block_idx == num_blocks) { break; }
				elts_in_block_copy = count_block(block_idx);
				if ((block_sorted_bitmap & one[block_idx])) {
#if DEBUG
					// if bitmap says its sorted, make sure
					for(size_t i = blocks_start + block_idx * block_size + 1; i < blocks_start + block_idx * block_size + elts_in_block_copy; i++) {
						ASSERT(blind_read_key(i-1) < blind_read_key(i), "i: %lu, key at i - 1: %lu, key at i: %lu, num_dels: %lu", i, blind_read_key(i-1), blind_read_key(i), num_deletes_in_log);
					}
#endif
				} else {
					sort_range(blocks_start + block_idx * block_size, blocks_start + block_idx * block_size + elts_in_block_copy);
					block_sorted_bitmap = block_sorted_bitmap | one[block_idx];
				}
				block_ptr = header_start + block_idx;
			} else {
				// in block
				block_ptr++;
			}
			num_applied++;
		}

		if (num_applied == length) { return num_applied; }
	}

	if (num_applied == length) { return num_applied; }

	assert(num_applied < length);

	// cleanup with log
	while(log_ptr < num_inserts_in_log) {
		// printf("in log cleanup num applied = %lu, log_key = %lu, length = %lu\n", num_applied, blind_read_key(log_ptr), length);
		std::apply(f, blind_read(log_ptr));
		log_ptr++;
		num_applied++;
		if (num_applied == length) { return num_applied; }
	}

	if (num_applied == length) { return num_applied; }

	assert(num_applied < length);

	// cleanup with blocks
	while(block_idx < num_blocks) {
		// printf("\toutput[%lu] = %lu from block\n", output.size(), blind_read_key_array(block_copy.data(), block_size, block_ptr));
		// output.push_back(blind_read_array(block_copy.data(), block_size, block_ptr));
		std::apply(f, blind_read(block_ptr));
		num_applied++;

		if (block_ptr < blocks_start && elts_in_block_copy > 0) {

			// in header, need to switch to block
			block_ptr = blocks_start + block_idx * block_size;
		} else if (elts_in_block_copy == 0 || block_ptr == blocks_start + block_idx * block_size + elts_in_block_copy - 1) {
			// at end of current block, need to switch to next block header
			block_idx++;
			if (block_idx == num_blocks) { break; }
			elts_in_block_copy = count_block(block_idx);
			if ((block_sorted_bitmap & one[block_idx])) {
#if DEBUG
				// if bitmap says its sorted, make sure
				assert(elts_in_block_copy < block_size);
				for(size_t i = blocks_start + block_idx * block_size + 1; i < blocks_start + block_idx * block_size + elts_in_block_copy; i++) {
					ASSERT(blind_read_key(i-1) < blind_read_key(i), "i: %lu, key at i - 1: %lu, key at i: %lu, num_dels: %lu", i, blind_read_key(i-1), blind_read_key(i), num_deletes_in_log);
				}
#endif
			} else {
				sort_range(blocks_start + block_idx * block_size, blocks_start + block_idx * block_size + elts_in_block_copy);
				block_sorted_bitmap = block_sorted_bitmap | one[block_idx];
			}
			block_ptr = header_start + block_idx;
		} else {
			// in block
			block_ptr++;
		}

		if (num_applied == length) { return num_applied; }
	}

	return num_applied;
}

// return a vector of element type
// TODO: make this apply function f to everything in the range
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
template <class F>
uint64_t LeafDS<log_size, header_size, block_size, key_type, Ts...>::sorted_range_no_write(key_type start, size_t length, F f) {
	uint64_t num_applied = 0;

	if (length == 0) {
		return num_applied;
	}
 
#if DEBUG
  assert(log_is_sorted); 
  for(size_t i = 1; i < num_inserts_in_log; i++) {
    assert(blind_read_key(i-1) < blind_read_key(i));
  }
#endif

	size_t log_ptr = 0;
	while(log_ptr < num_inserts_in_log && blind_read_key(log_ptr) < start) {
		log_ptr++;
	}

	// find and sort current block
	size_t block_idx = find_block(start);
	size_t elts_in_block_copy = count_block(block_idx);
  // printf("\tstarting at block %lu\n", block_idx);
	assert ((block_sorted_bitmap & one[block_idx]));

	// start block_ptr at header of current block
	size_t block_ptr = header_start + block_idx;

	// move block ptr to location of >= start key
	while(blind_read_key(block_ptr) < start && block_idx < num_blocks) {
		if (block_ptr < blocks_start && elts_in_block_copy > 0) {
			// in header, need to switch to block
			block_ptr = blocks_start + block_idx * block_size;
		} else if (elts_in_block_copy == 0 || block_ptr == blocks_start + block_idx * block_size + elts_in_block_copy - 1) {
			// at end of current block, need to switch to next block header
			block_idx++;
			if (block_idx == num_blocks) { break; }

      assert(count_block(block_idx) == count_per_block[block_idx]);
			elts_in_block_copy = count_per_block[block_idx];
     	assert ((block_sorted_bitmap & one[block_idx]));

			block_ptr = header_start + block_idx;
		} else {
			// in block
			block_ptr++;
		}
	}

	while(log_ptr < num_inserts_in_log && block_idx < num_blocks) {
		key_type log_key = blind_read_key(log_ptr);
		key_type block_key = blind_read_key(block_ptr);
		assert(num_applied < length);
		assert(log_key >= start);
		assert(block_key >= start);
		if (log_key == block_key) { // duplicate in log and blocks
			// log is the more recent one
			std::apply(f, blind_read(log_ptr));
			log_ptr++;
			num_applied++;

			// increment block pointer too
			if (block_ptr < blocks_start && elts_in_block_copy > 0) {
				// in header, need to switch to block
				block_ptr = blocks_start + block_idx * block_size;
			} else if (elts_in_block_copy == 0 || block_ptr == blocks_start + block_idx * block_size + elts_in_block_copy - 1) {
				// at end of current block, need to switch to next block header
				block_idx++;
				if (block_idx == num_blocks) { break; }
				assert(count_block(block_idx) == count_per_block[block_idx]);
        elts_in_block_copy = count_per_block[block_idx];
       	assert ((block_sorted_bitmap & one[block_idx]));
				block_ptr = header_start + block_idx;
			} else {
				// in block
				block_ptr++;
        assert(blind_read_key(block_ptr) > blind_read_key(block_ptr - 1));
			}

		} else if (log_key < block_key) {
			std::apply(f, blind_read(log_ptr));
			log_ptr++;
			num_applied++;
		} else {
			std::apply(f, blind_read(block_ptr));
			num_applied++;
			// in header, need to switch to block
			if (block_ptr < blocks_start && elts_in_block_copy > 0) {
				block_ptr = blocks_start + block_idx * block_size;
			} else if (elts_in_block_copy == 0 || block_ptr == blocks_start + block_idx * block_size + elts_in_block_copy - 1) {
				// at end of current block, need to switch to next block header
				block_idx++;
				if (block_idx == num_blocks) { break; }
        //printf("advancing block to %lu, num so far %lu\n", block_idx, num_applied);
				assert(count_block(block_idx) == count_per_block[block_idx]);
				elts_in_block_copy = count_per_block[block_idx];
#if DEBUG
        if (length < num_applied && !(block_sorted_bitmap & one[block_idx])) {
          printf("block_idx %lu, num applied %lu\n", block_idx, num_applied);
          printBits(sizeof(one[block_idx]), &one[block_idx]);
          printBits(sizeof(block_sorted_bitmap), &block_sorted_bitmap);
       	  assert ((block_sorted_bitmap & one[block_idx]));
        }
#endif
				block_ptr = header_start + block_idx;
			} else {
				block_ptr++;
			}
		}

		if (num_applied == length) { return num_applied; }
	}

	if (num_applied == length) { return num_applied; }

	assert(num_applied < length);

	// cleanup with log
	while(log_ptr < num_inserts_in_log) {
		// printf("in log cleanup num applied = %lu, log_key = %lu, length = %lu\n", num_applied, blind_read_key(log_ptr), length);
		std::apply(f, blind_read(log_ptr));
		log_ptr++;
		num_applied++;
		if (num_applied == length) { return num_applied; }
	}

	if (num_applied == length) { return num_applied; }

	assert(num_applied < length);

	// cleanup with blocks
	while(block_idx < num_blocks) {
		// printf("\toutput[%lu] = %lu from block\n", output.size(), blind_read_key_array(block_copy.data(), block_size, block_ptr));
		// output.push_back(blind_read_array(block_copy.data(), block_size, block_ptr));
		std::apply(f, blind_read(block_ptr));
		num_applied++;

		if (block_ptr < blocks_start && elts_in_block_copy > 0) {

			// in header, need to switch to block
			block_ptr = blocks_start + block_idx * block_size;
		} else if (elts_in_block_copy == 0 || block_ptr == blocks_start + block_idx * block_size + elts_in_block_copy - 1) {
			// at end of current block, need to switch to next block header
			block_idx++;
			if (block_idx == num_blocks) { break; }
      assert(count_per_block[block_idx] == count_block(block_idx));
			elts_in_block_copy = count_per_block[block_idx]; // count_block(block_idx);
#if DEBUG
        if (length < num_applied && !(block_sorted_bitmap & one[block_idx])) {
          printf("block_idx %lu, num applied %lu\n", block_idx, num_applied);
          printBits(sizeof(one[block_idx]), &one[block_idx]);
          printBits(sizeof(block_sorted_bitmap), &block_sorted_bitmap);
       	  assert ((block_sorted_bitmap & one[block_idx]));
        }
#endif
			block_ptr = header_start + block_idx;
		} else {
			// in block
			block_ptr++;
		}

		if (num_applied == length) { return num_applied; }
	}

	return num_applied;
}

// return a vector of element type
// TODO: make this apply function f to everything in the range
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
template <class F>
uint64_t LeafDS<log_size, header_size, block_size, key_type, Ts...>::sorted_range_end(key_type start, key_type end, F f) {
#if DEBUG_PRINT
	printf("\n\n*** sorted range starting at %lu ending at %lu ***\n", start, end);
#endif
	// printf("start = %lu, length = %lu\n", start, length);
	// copy log out-of-place and sort it
	// std::vector<element_type> output;
	uint64_t num_applied = 0;

	if (start == end) {
		return num_applied;
	}

	sort_range(0, num_inserts_in_log);

	size_t log_ptr = 0;
	while(log_ptr < num_inserts_in_log && blind_read_key(log_ptr) < start) {
		log_ptr++;
	}

	// find and sort current block
	size_t block_idx = find_block(start);
	size_t elts_in_block_copy = count_block(block_idx);

	if ((block_sorted_bitmap & one[block_idx])) {
#if DEBUG
		// if bitmap says its sorted, make sure
		for(size_t i = blocks_start + block_idx * block_size + 1; i < blocks_start + block_idx * block_size + elts_in_block_copy; i++) {
			if (blind_read_key(i-1) >= blind_read_key(i)) {
				print();
			}
			ASSERT(blind_read_key(i-1) < blind_read_key(i), "i: %lu, key at i - 1: %lu, key at i: %lu, num_dels: %lu \n bitmap = %lu , blockidx = %lu \n\n", i, blind_read_key(i-1), blind_read_key(i), num_deletes_in_log, block_sorted_bitmap, block_idx);
			
		}
#endif
	} else {
		sort_range(blocks_start + block_idx * block_size, blocks_start + block_idx * block_size + elts_in_block_copy);
		block_sorted_bitmap = block_sorted_bitmap | one[block_idx];
	}

	// start block_ptr at header of current block
	size_t block_ptr = header_start + block_idx;

	// move block ptr to location of >= start key
	while(blind_read_key(block_ptr) < start && block_idx < num_blocks) {
		if (block_ptr < blocks_start && elts_in_block_copy > 0) {
			// in header, need to switch to block
			block_ptr = blocks_start + block_idx * block_size;
		} else if (elts_in_block_copy == 0 || block_ptr == blocks_start + block_idx * block_size + elts_in_block_copy - 1) {
			// at end of current block, need to switch to next block header
			block_idx++;
			if (block_idx == num_blocks) { break; }

			elts_in_block_copy = count_block(block_idx);

			if ((block_sorted_bitmap & one[block_idx])) {
#if DEBUG
				// if bitmap says its sorted, make sure
				for(size_t i = blocks_start + block_idx * block_size + 1; i < blocks_start + block_idx * block_size + elts_in_block_copy; i++) {
					ASSERT(blind_read_key(i-1) < blind_read_key(i), "i: %lu, key at i - 1: %lu, key at i: %lu, num_dels: %lu", i, blind_read_key(i-1), blind_read_key(i), num_deletes_in_log);
				}
#endif
			} else {
				sort_range(blocks_start + block_idx * block_size, blocks_start + block_idx * block_size + elts_in_block_copy);
				block_sorted_bitmap = block_sorted_bitmap | one[block_idx];
			}
			block_ptr = header_start + block_idx;
		} else {
			// in block
			block_ptr++;
		}
	}

	while(log_ptr < num_inserts_in_log && block_idx < num_blocks) {
		key_type log_key = blind_read_key(log_ptr);
		key_type block_key = blind_read_key(block_ptr);
		if (log_key >= end && block_key >= end) {
			return num_applied;
		} else if (log_key >= end ) {
			log_ptr = num_inserts_in_log;
			break;
		} else if (block_key >= end) {
			block_idx = num_blocks;
			break;
		}
		assert(log_key >= start);
		assert(block_key >= start);
		if (log_key == block_key) { // duplicate in log and blocks
			// log is the more recent one
			std::apply(f, blind_read(log_ptr));
			log_ptr++;
			num_applied++;

			// increment block pointer too
			if (block_ptr < blocks_start && elts_in_block_copy > 0) {
				// in header, need to switch to block
				block_ptr = blocks_start + block_idx * block_size;
			} else if (elts_in_block_copy == 0 || block_ptr == blocks_start + block_idx * block_size + elts_in_block_copy - 1) {
				// at end of current block, need to switch to next block header
				block_idx++;
				if (block_idx == num_blocks) { break; }
				elts_in_block_copy = count_block(block_idx);
				if ((block_sorted_bitmap & one[block_idx])) {
#if DEBUG
					// if bitmap says its sorted, make sure
					for(size_t i = blocks_start + block_idx * block_size + 1; i < blocks_start + block_idx * block_size + elts_in_block_copy; i++) {
						ASSERT(blind_read_key(i-1) < blind_read_key(i), "i: %lu, key at i - 1: %lu, key at i: %lu, num_dels: %lu", i, blind_read_key(i-1), blind_read_key(i), num_deletes_in_log);
					}
#endif
				} else {
					sort_range(blocks_start + block_idx * block_size, blocks_start + block_idx * block_size + elts_in_block_copy);
					block_sorted_bitmap = block_sorted_bitmap | one[block_idx];
				}
				block_ptr = header_start + block_idx;
			} else {
				// in block
				block_ptr++;
			}

		} else if (log_key < block_key) {
			std::apply(f, blind_read(log_ptr));
			log_ptr++;
			num_applied++;
		} else {
			std::apply(f, blind_read(block_ptr));

			if (block_ptr < blocks_start && elts_in_block_copy > 0) {
				// in header, need to switch to block
				block_ptr = blocks_start + block_idx * block_size;
			} else if (elts_in_block_copy == 0 || block_ptr == blocks_start + block_idx * block_size + elts_in_block_copy - 1) {
				// at end of current block, need to switch to next block header
				block_idx++;
				if (block_idx == num_blocks) { break; }
				elts_in_block_copy = count_block(block_idx);
				if ((block_sorted_bitmap & one[block_idx])) {
#if DEBUG
					// if bitmap says its sorted, make sure
					for(size_t i = blocks_start + block_idx * block_size + 1; i < blocks_start + block_idx * block_size + elts_in_block_copy; i++) {
						ASSERT(blind_read_key(i-1) < blind_read_key(i), "i: %lu, key at i - 1: %lu, key at i: %lu, num_dels: %lu", i, blind_read_key(i-1), blind_read_key(i), num_deletes_in_log);
					}
#endif
				} else {
					sort_range(blocks_start + block_idx * block_size, blocks_start + block_idx * block_size + elts_in_block_copy);
					block_sorted_bitmap = block_sorted_bitmap | one[block_idx];
				}
				block_ptr = header_start + block_idx;
			} else {
				// in block
				block_ptr++;
			}
			num_applied++;
		}
	}

	// cleanup with log
	while(log_ptr < num_inserts_in_log) {
		// printf("in log cleanup num applied = %lu, log_key = %lu, length = %lu\n", num_applied, blind_read_key(log_ptr), length);
		if (blind_read_key(log_ptr) >= end) {
			break;
		}
		std::apply(f, blind_read(log_ptr));
		log_ptr++;
		num_applied++;
	}

	// cleanup with blocks
	while(block_idx < num_blocks) {
		if (blind_read_key(block_ptr) >= end) {
			break;
		}
		// printf("\toutput[%lu] = %lu from block\n", output.size(), blind_read_key_array(block_copy.data(), block_size, block_ptr));
		// output.push_back(blind_read_array(block_copy.data(), block_size, block_ptr));
		std::apply(f, blind_read(block_ptr));
		num_applied++;

		if (block_ptr < blocks_start && elts_in_block_copy > 0) {

			// in header, need to switch to block
			block_ptr = blocks_start + block_idx * block_size;
		} else if (elts_in_block_copy == 0 || block_ptr == blocks_start + block_idx * block_size + elts_in_block_copy - 1) {
			// at end of current block, need to switch to next block header
			block_idx++;
			if (block_idx == num_blocks) { break; }
			elts_in_block_copy = count_block(block_idx);
			if ((block_sorted_bitmap & one[block_idx])) {
#if DEBUG
				// if bitmap says its sorted, make sure
				assert(elts_in_block_copy < block_size);
				for(size_t i = blocks_start + block_idx * block_size + 1; i < blocks_start + block_idx * block_size + elts_in_block_copy; i++) {
					ASSERT(blind_read_key(i-1) < blind_read_key(i), "i: %lu, key at i - 1: %lu, key at i: %lu, num_dels: %lu", i, blind_read_key(i-1), blind_read_key(i), num_deletes_in_log);
				}
#endif
			} else {
				sort_range(blocks_start + block_idx * block_size, blocks_start + block_idx * block_size + elts_in_block_copy);
				block_sorted_bitmap = block_sorted_bitmap | one[block_idx];
			}
			block_ptr = header_start + block_idx;
		} else {
			// in block
			block_ptr++;
		}
	}

	return num_applied;
}

// split for b+-tree
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
key_type LeafDS<log_size, header_size, block_size, key_type, Ts...>::split(LeafDS<log_size, header_size, block_size, key_type, Ts...>* right) {
#if DEBUG_PRINT
	printf("\n*** SPLIT ***\n");
	print();
#endif
	// flush log to blocks
	sort_range(0, num_inserts_in_log);

	if (num_inserts_in_log > 0 && blind_read_key(0) < get_min_block_key()) {
#if DEBUG_PRINT
		printf("first key in log = %lu, min block key = %lu\n", blind_read_key(0), get_min_block_key());
#endif
		size_t j = blocks_start + block_size;
		// find the first zero slot in the block
		SOA_type::template map_range_with_index_static(array.data(), N, [&j](auto index, auto key, auto... values) {
			if (key == 0) {
				j = std::min(index, j);
			}
		}, blocks_start, blocks_start + block_size);
#if DEBUG_PRINT
		printf("first empty slot = %lu\n", j);
		print();
#endif

		// put the old min header in the block where it belongs
		// src = header start, dest = i
		copy_src_to_dest(header_start, j);
		// make min elt the new first header
		copy_src_to_dest(0, header_start);
    count_per_block[0]++;
    assert(count_per_block[0] == count_block(0));
	}

	flush_log_to_blocks(num_inserts_in_log);
	num_inserts_in_log = 0;

	// sort blocks
	// count the number of elements in each block
	// unsigned short count_per_block[num_blocks];
#if DEBUG
	for (size_t i = 0; i < num_blocks; i++) {
		assert(count_per_block[i] == count_block(i));
	}
#endif

	// sort the blocks
	for (size_t i = 0; i < num_blocks; i++) {
		auto block_range = get_block_range(i);
		sort_range(block_range.first, block_range.first + count_per_block[i]);
	}

	// merge all elts into sorted order
	std::vector<element_type> buffer;
	size_t blocks_ptr = header_start;
	size_t cur_block = 0;
	size_t start_of_cur_block = 0;
	while(cur_block < num_blocks) {
#if DEBUG
		ASSERT(blind_read_key(blocks_ptr) != NULL_VAL, "block ptr %lu\n", blocks_ptr);
#endif

		buffer.push_back(blind_read(blocks_ptr));
    advance_block_ptr(&blocks_ptr, &cur_block, &start_of_cur_block);
		// advance_block_ptr(&blocks_ptr, &cur_block, &start_of_cur_block, count_per_block);
	}
#if DEBUG
	assert(buffer.size() < num_blocks * block_size);
#endif
	num_elts_total = buffer.size();

#if DEBUG_PRINT
	printf("*** BUFFER ***\n");
	for(size_t i = 0; i < buffer.size(); i++) {
		printf("%u\t", std::get<0>(buffer[i]));
	}
	printf("\n");
#endif

#if DEBUG
	// at this point the buffer should be sorted
	for (size_t i = 1; i < buffer.size(); i++) {
		assert(has(std::get<0>(buffer[i])));
		ASSERT(std::get<0>(buffer[i]) > std::get<0>(buffer[i-1]), "buffer[%lu] = %lu, buffer[%lu] = %lu\n", i-1, std::get<0>(buffer[i-1]), i, std::get<0>(buffer[i]));
	}
#endif
	clear_range(0, N);

	// pick a midpoint to return
#if DEBUG_PRINT
	printf("buffer size = %lu\n", buffer.size());
#endif
	size_t midpoint = buffer.size() / 2;
	size_t left_size = midpoint;
	size_t elts_per_block = left_size / num_blocks;
	size_t remainder = left_size % num_blocks;
	// put the first half of the array into left, put the second half into right
	size_t cur_idx = 0;
	// size_t idx_in_block = 0;
#if DEBUG_PRINT
  printf("left size = %lu, elts per block = %lu, remainder = %lu\n", left_size, elts_per_block, remainder);
#endif
	for(size_t i = 0; i < num_blocks; i++) {
		assert(cur_idx < left_size);
		blind_write(buffer[cur_idx], header_start + i);
#if DEBUG_PRINT
		printf("\twrote buffer[%lu] = %lu as header %lu\n", cur_idx, std::get<0>(buffer[cur_idx]), i);
#endif
		size_t j;
		for(j = 1; j < elts_per_block + (i < remainder); j++) {
#if DEBUG_PRINT
			printf("\twrote buffer[%lu] = %lu at position %lu \n", cur_idx + j, std::get<0>(buffer[cur_idx + j]), blocks_start + i*block_size + j -1);
#endif
			blind_write(buffer[cur_idx + j], blocks_start + i * block_size + j - 1);
		}
    count_per_block[i] = j - 1;
    assert(count_per_block[i] == count_block(i));
		cur_idx += j;
#if DEBUG_PRINT
		printf("cur idx at end of left loop %lu\n", cur_idx);
#endif
	}

	num_elts_total = left_size;
	// do the same for right side
	size_t right_size = buffer.size() - midpoint;
	elts_per_block = right_size / num_blocks;
	remainder = right_size % num_blocks;
#if DEBUG_PRINT
	printf("right size = %lu, elts per block = %lu\n", left_size, elts_per_block);
#endif
	for(size_t i = 0; i < num_blocks; i++) {
		assert(cur_idx < buffer.size());
		right->blind_write(buffer[cur_idx], header_start + i);
		size_t j;
		for(j = 1; j < elts_per_block + (i < remainder); j++) {
			right->blind_write(buffer[cur_idx + j], blocks_start + i * block_size + j - 1);
		}
    right->set_block_count(i, j - 1);
    assert(right->get_block_count(i) == right->count_block(i));
		cur_idx += j;
#if DEBUG_PRINT
		printf("cur idx at end of right loop = %lu\n", cur_idx);
#endif
	}
	assert(cur_idx == buffer.size());
	right->num_elts_total = right_size;

	// end
	key_type mid_elt = std::get<0>(buffer[midpoint - 1]);
#if DEBUG_PRINT
	printf("mid elt = %lu\n", mid_elt);
	printf("\n\n**LEFT**\n");
	print();
	printf("\n\n**RIGHT**\n");
	right->print();
#endif

	return mid_elt;
}

//! Merge two leaf nodes. The function moves all key/data pairs from right
//! to left and sets right's slotuse to zero. The right slot is then removed
//! by the calling parent node.
// Assumes that right's log_size, header_size, and block_size is the same as this
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
void LeafDS<log_size, header_size, block_size, key_type, Ts...>::merge(LeafDS<log_size, header_size, block_size, key_type, Ts...>* right) {
	// version mirroring shift_left

	// flush deletes
	if (right->num_deletes_in_log > 0 && right->blind_read_key(right->header_start) != 0) {
		right->sort_range(log_size - right->num_deletes_in_log, log_size);
		if (right->delete_from_header()) {
			right->strip_deletes_and_redistrib();
		} else {
			right->flush_deletes_to_blocks();
		}
	}
	right->clear_range(log_size - right->num_deletes_in_log, log_size);
	right->num_deletes_in_log = 0;

	// std::vector<key_type> elts_to_remove;

	// flush inserts
	if (right->num_inserts_in_log > 0 && right->blind_read_key(right->header_start) != 0) {
		right->sort_range(0, right->num_inserts_in_log);
		// handle case of if log < header 
		if (right->blind_read_key(0) < right->get_min_block_key()) {
			size_t j = blocks_start + block_size;
			// find the first zero slot in the block
			SOA_type::template map_range_with_index_static(right->array.data(), N, [&j](auto index, auto key, auto... values) {
				if (key == 0) {
					j = std::min(index, j);
				}
			}, blocks_start, blocks_start + block_size);

			// put the old min header in the block where it belongs
			// src = header start, dest = i
			right->copy_src_to_dest(header_start, j);
			// make min elt the new first header
			right->copy_src_to_dest(0, header_start);
		}
		right->flush_log_to_blocks(right->num_inserts_in_log);
	} else if (right->num_inserts_in_log > 0) {
		// if header is empty, only iterate over log
		for (size_t i = 0; i < right->num_inserts_in_log; i++) {
			// delete from right and insert into left
			insert(right->blind_read(i));
			// elts_to_remove.push_back(right->blind_read_key(i));
		}
		right->clear_range(0, right->N);
		right->num_elts_total = 0;
		right->num_inserts_in_log = 0;
		return;
	}
	right->clear_range(0, right->num_inserts_in_log);
	right->num_inserts_in_log = 0;

	// Count + sort the blocks of right
	// unsigned short count_per_block[num_blocks];
	for (size_t i = 0; i < num_blocks; i++) {
		count_per_block[i] = right->count_block(i);
	}

	// we go forwards from start of block via header
	size_t last_header = header_start;
	size_t cur_block = 0;

	while (last_header < blocks_start) {

		// header is smallest, add this
		insert(right->blind_read(last_header));

		// loop through entire current block
		for (size_t i = blocks_start + cur_block * block_size; i < blocks_start + cur_block * block_size + count_per_block[cur_block]; i++) {
			insert(right->blind_read(i));
		}
		// increment header and block
		last_header++;
		cur_block++;
	}

	right->clear_range(0, right->N);
	right->num_elts_total = 0;
	right->num_inserts_in_log = 0;
	return;
}

// assumes manual_num_elts == true size of leafds
// maxkey and secondmaxkey inits to 0
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
void LeafDS<log_size, header_size, block_size, key_type, Ts...>::get_max_2(key_type* max_key, key_type* second_max_key) const {
	/*
	// simple const implementation without knowing total number of elements

	std::vector<key_type> sorted_insert_log;
	std::vector<key_type> sorted_delete_log;

	std::set<key_type> all_elems;
	std::vector<key_type> all_elems_sorted;
	// count the number of elements in each block
	unsigned short count_per_block[num_blocks];

	// TODO: merge these loops and count the rest in global redistribute
	for (size_t i = 0; i < num_blocks; i++) {
		count_per_block[i] = count_block(i);
	}

	size_t blocks_ptr_1 = header_start;
	size_t cur_block_1 = 0;
	size_t start_of_cur_block_1 = 0;
	while(cur_block_1 < num_blocks) {
		all_elems.insert(blind_read_key(blocks_ptr_1));
		advance_block_ptr(&blocks_ptr_1, &cur_block_1, &start_of_cur_block_1, count_per_block);
	}
	// printf("size of all elems in blocks + header: %lu\n",all_elems.size());
	for (size_t i = log_size - num_deletes_in_log; i < log_size; i++) {
		all_elems.erase(blind_read_key(i));
	}
	// printf("size of all elems in blocks + header after erasing delete log: %lu\n", all_elems.size());
	for (size_t i = 0; i < num_inserts_in_log; i++) {
		all_elems.insert(blind_read_key(i));
	}
	// printf("size of all elems in blocks + header after adding insert log: %lu\n", all_elems.size());

	for (auto e: all_elems) {
		all_elems_sorted.push_back(e);
	}
	std::sort(all_elems_sorted.begin(), all_elems_sorted.end());
	// printf("size of all elems sorted in blocks + header: %lu\n", all_elems_sorted.size());
	// printf("printing current sorted full leafds: \n");

	key_type target_max_key = all_elems_sorted[all_elems_sorted.size() - 1];
	key_type target_second_max_key = all_elems_sorted[all_elems_sorted.size() - 2];
	
	for (size_t i = 0; i < num_inserts_in_log; i++) {
		if (target_max_key == blind_read_key(i)) {
			*max_key = blind_read_key(i);
		}
		if (target_second_max_key == blind_read_key(i)) {
			*second_max_key = blind_read_key(i);
		}
	}

	blocks_ptr_1 = header_start;
	cur_block_1 = 0;
	start_of_cur_block_1 = 0;
	while(cur_block_1 < num_blocks) {
		if (target_max_key == blind_read_key(blocks_ptr_1)) {
			*max_key = blind_read_key(blocks_ptr_1);
		}
		if (target_second_max_key == blind_read_key(blocks_ptr_1)) {
			*second_max_key = blind_read_key(blocks_ptr_1);
		}
		advance_block_ptr(&blocks_ptr_1, &cur_block_1, &start_of_cur_block_1, count_per_block);
	}
	*/

	// /*
	// const implementation without knowing total number of elements
	*max_key = 0;
	*second_max_key = 0;
	
	if (blind_read_key(header_start) == 0) {
		// if only the log exists, sweep through log
		assert(num_inserts_in_log >= 2);
		for (size_t i = 0; i < num_inserts_in_log; i++) {
			if (blind_read_key(i) > *max_key) {
				*second_max_key = *max_key;
				*max_key = blind_read_key(i);
			} else if ((blind_read_key(i) > *second_max_key)) {
				*second_max_key =  blind_read_key(i);
			}
		}
		return;
	}

	// if not, we go backwards from end of block via header
	size_t last_header = blocks_start - 1;
	size_t cur_block = num_blocks - 1;
	size_t cur_block_count = count_block(cur_block);

	while (!(*max_key) || !(*second_max_key)) {
		// loop through entire unsorted current block if it exists 
		for (size_t i = blocks_start + cur_block * block_size ; i < blocks_start + cur_block * block_size + cur_block_count; i++) {
			key_type curr_key = blind_read_key(i);
			if (is_in_delete_log(curr_key)) {
				continue;
			}
			// if we beat maxkey or second max key, set it
			if ((*max_key) < curr_key) {
				*second_max_key = *max_key;
				*max_key = curr_key;
			} else if ((*second_max_key) < curr_key) {
				*second_max_key = curr_key;
			}
		}
		if (!(*max_key) && !is_in_delete_log(blind_read_key(last_header))) {
			*max_key = blind_read_key(last_header);
		} else if (!(*second_max_key) && !is_in_delete_log(blind_read_key(last_header))) {
			*second_max_key = blind_read_key(last_header);
		}
		cur_block--;
		last_header--;
		cur_block_count = count_block(cur_block);
	}

	// finally check if anything in insert log beats our block max keys
	if (num_inserts_in_log > 0) {
		for (size_t i = 0; i < num_inserts_in_log; i++) {
			if (blind_read_key(i) > *max_key) {
				*second_max_key = *max_key;
				*max_key = blind_read_key(i);
			} else if ((blind_read_key(i) > *second_max_key)) {
				*second_max_key =  blind_read_key(i);
			}
		}
	}
	return;
	// */
}


// Balance two leaf nodes. The function moves key/data pairs from right to
// left so that both nodes are equally filled. The parent node is updated
// if possible.
// assumes shiftnum < num_elts, value of right's log_size, header_size, block_size is same as this
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
void LeafDS<log_size, header_size, block_size, key_type, Ts...>::shift_left(LeafDS<log_size, header_size, block_size, key_type, Ts...>* right, int shiftnum) {
	// faster version that doesn't require size or using get_sorted_index
// /*
	// flush deletes
	if (right->num_deletes_in_log > 0 && right->blind_read_key(right->header_start) != 0) {
		right->sort_range(log_size - right->num_deletes_in_log, log_size);
		if (right->delete_from_header()) {
			// printf("deleting from right header, stripping and deleting");
			right->strip_deletes_and_redistrib();
		} else {
			right->flush_deletes_to_blocks();
		}
	}
	right->clear_range(log_size - right->num_deletes_in_log, log_size);
	right->num_deletes_in_log = 0;

	std::vector<key_type> elts_to_remove;

	// flush inserts
	if (right->num_inserts_in_log > 0 && right->blind_read_key(right->header_start) != 0) {
		right->sort_range(0, right->num_inserts_in_log);
		// handle case of if log < header 
		if (right->blind_read_key(0) < right->get_min_block_key()) {
			size_t j = blocks_start + block_size;
			// find the first zero slot in the block
			SOA_type::template map_range_with_index_static(right->array.data(), N, [&j](auto index, auto key, auto... values) {
				if (key == 0) {
					j = std::min(index, j);
				}
			}, blocks_start, blocks_start + block_size);

			// put the old min header in the block where it belongs
			// src = header start, dest = i
			right->copy_src_to_dest(header_start, j);
			// make min elt the new first header
			right->copy_src_to_dest(0, header_start);
		}
		right->flush_log_to_blocks(right->num_inserts_in_log);
	} else if (right->num_inserts_in_log > 0) {
		// if header is empty, only iterate over log
		right->sort_range(0, right->num_inserts_in_log);
		for (size_t i = 0; i < shiftnum; i++) {
			// delete from right and insert into right
			insert(right->blind_read(i));
			elts_to_remove.push_back(right->blind_read_key(i));
		}
		for (auto e: elts_to_remove) {
			right->remove(e);
		}
		return;
	}
	right->clear_range(0, right->num_inserts_in_log);
	right->num_inserts_in_log = 0;

	// Count + sort the blocks of right
	unsigned short right_count_per_block[num_blocks];
	for (size_t i = 0; i < num_blocks; i++) {
		right_count_per_block[i] = right->count_block(i);
		right->sort_range(blocks_start + i * block_size, blocks_start + i * block_size + right_count_per_block[i]);
	}

	// we go forwards from start of block via header
	size_t last_header = header_start;
	size_t cur_block = 0;
	int cur_count = 0;

	while (cur_count < shiftnum) {

		// header is smallest, add this
		insert(right->blind_read(last_header));
		elts_to_remove.push_back(right->blind_read_key(last_header));
		cur_count++;

		if (cur_count < shiftnum) {
			// if we finish header and still have more, loop through entire sorted current block
			for (size_t i = blocks_start + cur_block * block_size; i < blocks_start + cur_block * block_size + right_count_per_block[cur_block]; i++) {
				insert(right->blind_read(i));
				elts_to_remove.push_back(right->blind_read_key(i));
				cur_count++;
				if (cur_count >= shiftnum) {
					break;
				}
			}
		}
		// increment header and block
		last_header++;
		cur_block++;
	}

	for (auto e: elts_to_remove) {
		right->remove(e);
	}
	return;
}

// Balance two leaf nodes. The function moves key/data pairs from left to
// right so that both nodes are equally filled. The parent node is updated
// if possible.
// TODO: for now, since get_num_elements() is unsafe, pass in the num of elements in left so we can index into it
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
void LeafDS<log_size, header_size, block_size, key_type, Ts...>::shift_right(LeafDS<log_size, header_size, block_size, key_type, Ts...>* left, int shiftnum) {
	// faster version that doesn't require size or using get_sorted_index

	// flush deletes
	if (left->num_deletes_in_log > 0 && left->blind_read_key(left->header_start) != 0) {
		left->sort_range(log_size - left->num_deletes_in_log, log_size);
		if (left->delete_from_header()) {
			// printf("deleting from left header, stripping and deleting");
			left->strip_deletes_and_redistrib();
		} else {
			left->flush_deletes_to_blocks();
		}
	}
	left->clear_range(log_size - left->num_deletes_in_log, log_size);
	left->num_deletes_in_log = 0;

	std::vector<key_type> elts_to_remove;

	// flush inserts
	if (left->num_inserts_in_log > 0 && left->blind_read_key(left->header_start) != 0) {
		left->sort_range(0, left->num_inserts_in_log);
		// handle case of if log < header 
		if (left->blind_read_key(0) < left->get_min_block_key()) {
			size_t j = blocks_start + block_size;
			// find the first zero slot in the block
			SOA_type::template map_range_with_index_static(left->array.data(), N, [&j](auto index, auto key, auto... values) {
				if (key == 0) {
					j = std::min(index, j);
				}
			}, blocks_start, blocks_start + block_size);

			// put the old min header in the block where it belongs
			// src = header start, dest = i
			left->copy_src_to_dest(header_start, j);
			// make min elt the new first header
			left->copy_src_to_dest(0, header_start);
		}
		left->flush_log_to_blocks(left->num_inserts_in_log);
	} else if (left->num_inserts_in_log > 0) {
		// if header is empty, only iterate over log
		left->sort_range(0, left->num_inserts_in_log);
		for (size_t i = left->num_inserts_in_log - shiftnum; i < left->num_inserts_in_log; i++) {
			// delete from left and insert into right
			insert(left->blind_read(i));
			elts_to_remove.push_back(left->blind_read_key(i));
		}
		for (auto e: elts_to_remove) {
			left->remove(e);
		}
		return;
	}
	left->clear_range(0, left->num_inserts_in_log);
	left->num_inserts_in_log = 0;

	// Count + sort the blocks of left
	uint8_t left_count_per_block[num_blocks];
	for (size_t i = 0; i < num_blocks; i++) {
		left_count_per_block[i] = left->count_block(i);
		left->sort_range(blocks_start + i * block_size, blocks_start + i * block_size + left_count_per_block[i]);
	}

	// we go backwards from end of block via header
	size_t last_header = blocks_start - 1;
	size_t cur_block = num_blocks - 1;
	int cur_count = 0;

	while (cur_count < shiftnum) {

		// loop through entire sorted current block backwards first
		for (size_t i = blocks_start + cur_block * block_size + left_count_per_block[cur_block] - 1; i >= blocks_start + cur_block * block_size; i--) {
			insert(left->blind_read(i));
			elts_to_remove.push_back(left->blind_read_key(i));
			cur_count++;
			if (cur_count >= shiftnum) {
				break;
			}
		}
		// if we finish this entire block and still have more, add the header key and decrement header ptrs
		if (cur_count < shiftnum) {
			insert(left->blind_read(last_header));
			elts_to_remove.push_back(left->blind_read_key(last_header));
			last_header--;
			cur_block--;
			cur_count++;
		}
	}

	for (auto e: elts_to_remove) {
		left->remove(e);
	}
	return;
}

// Assumes i < get_num_elts
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
key_type& LeafDS<log_size, header_size, block_size, key_type, Ts...>::get_key_at_sorted_index_nonconst(size_t i) {
	// flush delete log so we can do two ptr sorted check on inserts
	if (num_deletes_in_log > 0 && blind_read_key(header_start) != 0) {
		sort_range(log_size - num_deletes_in_log, log_size);
		if (delete_from_header()) {
#if DEBUG_PRINT
			printf("deleting from header, stripping and deleting");
#endif
			strip_deletes_and_redistrib();
		} else {
			flush_deletes_to_blocks();
		}
	}
	clear_range(log_size - num_deletes_in_log, log_size);
	num_deletes_in_log = 0;

	// Sort log
	sort_range(0, num_inserts_in_log);

	// Count + sort the blocks
	// unsigned short count_per_block[num_blocks];
  /*
	for (size_t i = 0; i < num_blocks; i++) {
		count_per_block[i] = count_block(i);
	}
  */

	for (size_t i = 0; i < num_blocks; i++) {
		auto block_range = get_block_range(i);
		sort_range(block_range.first, block_range.first + count_per_block[i]);
	}

	// two pointer search for key at sorted index
	size_t log_ptr = 0;
	size_t blocks_ptr = header_start;
	size_t cur_block = 0;
	size_t start_of_cur_block = 0;
	size_t curr_index = 0;
	while ((log_ptr < num_inserts_in_log || cur_block < num_blocks) && curr_index < i) {
		// check which ptr to increment
		if (log_ptr < num_inserts_in_log && cur_block < num_blocks) {
			if (blind_read_key(log_ptr) <= blind_read_key(blocks_ptr)) {
				log_ptr++;
			} else if (blind_read_key(blocks_ptr) == 0 && cur_block == 0) {
				// edge case of first header block being 0 pre-first flush, we still want to increment log_ptr here
				log_ptr++;
			} else {
				advance_block_ptr(&blocks_ptr, &cur_block, &start_of_cur_block, count_per_block);
			}
		} else if (log_ptr < num_inserts_in_log) {
			log_ptr++;
		} else {
			advance_block_ptr(&blocks_ptr, &cur_block, &start_of_cur_block, count_per_block);
		}
		curr_index++;
	}
#if DEBUG_PRINT
	auto log_ptr_val = blind_read_key(log_ptr);
	auto blocks_ptr_val = blind_read_key(blocks_ptr);
	printf("looking for index %lu\n", i);
	printf("\tlogptr %lu, val %lu\n", log_ptr, log_ptr_val);
	printf("\tblocks_ptr %lu, val %lu\n", blocks_ptr, blocks_ptr_val);
#endif
	if (log_ptr < num_inserts_in_log && cur_block < num_blocks) {
		if (blind_read(log_ptr) <= blind_read(blocks_ptr)) {
			return std::get<0>(blind_read(log_ptr));
		} else if (blind_read_key(blocks_ptr) == 0 && cur_block == 0) {
			// edge case of first header block being 0 pre-first flush
			return std::get<0>(blind_read(log_ptr));
		} else {
			return std::get<0>(blind_read(blocks_ptr));
		}
	} else if (log_ptr < num_inserts_in_log) {
		return std::get<0>(blind_read(log_ptr));
	} else {
		return std::get<0>(blind_read(blocks_ptr));
	}
}

// Assumes i < get_num_elts
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
key_type& LeafDS<log_size, header_size, block_size, key_type, Ts...>::get_key_at_sorted_index(size_t target_i) const {
// /*
	// TODO! FIX THE FASTER VERSION
	// this very slow version is correct but too slow, the current version doesn't match up with the nonconst version in some max_2 calls
	
	// printf("\t looking for target_i %u\n", target_i);
	// make sorted copies of insert and delete log, note we only need keys
	std::vector<key_type> sorted_insert_log;
	std::vector<key_type> sorted_delete_log;

	std::set<key_type> all_elems;
	std::vector<key_type> all_elems_sorted;

	size_t blocks_ptr_1 = header_start;
	size_t cur_block_1 = 0;
	size_t start_of_cur_block_1 = 0;
	while(cur_block_1 < num_blocks) {
		all_elems.insert(blind_read_key(blocks_ptr_1));
		advance_block_ptr(&blocks_ptr_1, &cur_block_1, &start_of_cur_block_1);
	}
	// printf("size of all elems in blocks + header: %lu\n",all_elems.size());
	for (size_t i = log_size - num_deletes_in_log; i < log_size; i++) {
		all_elems.erase(blind_read_key(i));
	}
	// printf("size of all elems in blocks + header after erasing delete log: %lu\n", all_elems.size());
	for (size_t i = 0; i < num_inserts_in_log; i++) {
		all_elems.insert(blind_read_key(i));
	}
	// printf("size of all elems in blocks + header after adding insert log: %lu\n", all_elems.size());

	for (auto e: all_elems) {
		all_elems_sorted.push_back(e);
	}
	std::sort(all_elems_sorted.begin(), all_elems_sorted.end());
	// printf("size of all elems sorted in blocks + header: %lu\n", all_elems_sorted.size());
	// printf("printing current sorted full leafds: \n");

	key_type target_key = all_elems_sorted[target_i];
	
	for (size_t i = 0; i < num_inserts_in_log; i++) {
		if (target_key == blind_read_key(i)) {
			// return std::get<0>(blind_read(i));
		}
	}

	blocks_ptr_1 = header_start;
	cur_block_1 = 0;
	start_of_cur_block_1 = 0;
	while(cur_block_1 < num_blocks) {
		if (target_key == blind_read_key(blocks_ptr_1)) {
			// return std::get<0>(blind_read(blocks_ptr_1));
		}
		advance_block_ptr(&blocks_ptr_1, &cur_block_1, &start_of_cur_block_1);
	}
// */
}

// Assumes i < get_num_elts
// returns index to get elem using blind_read()
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
size_t LeafDS<log_size, header_size, block_size, key_type, Ts...>::get_element_at_sorted_index(size_t i) {
	// flush delete log so we can do two ptr sorted check on inserts
	if (num_deletes_in_log > 0) {
		sort_range(log_size - num_deletes_in_log, log_size);
		if (delete_from_header()) {
			// printf("deleting from header, stripping and deleting");
			strip_deletes_and_redistrib();
		} else {
			flush_deletes_to_blocks();
		}
	}
	clear_range(log_size - num_deletes_in_log, log_size);
	num_deletes_in_log = 0;

	// Sort log
	sort_range(0, num_inserts_in_log);

	// Count + sort the blocks
  /*
	unsigned short count_per_block[num_blocks];
	for (size_t i = 0; i < num_blocks; i++) {
		count_per_block[i] = count_block(i);
	}
  */

	for (size_t i = 0; i < num_blocks; i++) {
		auto block_range = get_block_range(i);
		sort_range(block_range.first, block_range.first + count_per_block[i]);
	}

	// two pointer search for key at sorted index
	size_t log_ptr = 0;
	size_t blocks_ptr = header_start;
	size_t cur_block = 0;
	size_t start_of_cur_block = 0;
	size_t curr_index = 0;
	while ((log_ptr < num_inserts_in_log || cur_block < num_blocks) && curr_index < i) {
		// check which ptr to increment
		if (log_ptr < num_inserts_in_log && cur_block < num_blocks) {
			if (blind_read_key(log_ptr) <= blind_read_key(blocks_ptr)) {
				log_ptr++;
			} else if (blind_read_key(blocks_ptr) == 0 && cur_block == 0) {
				// edge case of first header block being 0 pre-first flush, we still want to increment log_ptr here
				log_ptr++;
			} else {
				advance_block_ptr(&blocks_ptr, &cur_block, &start_of_cur_block);
			}
		} else if (log_ptr < num_inserts_in_log) {
			log_ptr++;
		} else {
			advance_block_ptr(&blocks_ptr, &cur_block, &start_of_cur_block);
		}
		curr_index++;
	}
#if DEBUG_PRINT
	auto log_ptr_val = blind_read_key(log_ptr);
	auto blocks_ptr_val = blind_read_key(blocks_ptr);
	printf("looking for index %lu\n", i);
	printf("\tlogptr %lu, val %lu\n", log_ptr, log_ptr_val);
	printf("\tblocks_ptr %lu, val %lu\n", blocks_ptr, blocks_ptr_val);
#endif
	if (log_ptr < num_inserts_in_log && cur_block < num_blocks) {
		if (blind_read(log_ptr) <= blind_read(blocks_ptr)) {
			return log_ptr;
		} else if (blind_read_key(blocks_ptr) == 0 && cur_block == 0) {
			// edge case of first header block being 0 pre-first flush
			return log_ptr;
		} else {
			return blocks_ptr;
		}
	} else if (log_ptr < num_inserts_in_log) {
		return log_ptr;
	} else {
		return blocks_ptr;
	}
}

// TODO: fix, gives wrong sizes sometimes
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
size_t LeafDS<log_size, header_size, block_size, key_type, Ts...>::get_num_elements() const {
	// printf("called not safe get_num_elements\n");
	return num_elts_total;
}

// Assumes i < get_num_elts
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
key_type& LeafDS<log_size, header_size, block_size, key_type, Ts...>::get_key_at_sorted_index_with_print(size_t target_i) const {
	printf("\t looking for target_i %u\n", target_i);
	// make sorted copies of insert and delete log, note we only need keys
	std::vector<key_type> sorted_insert_log;
	std::vector<key_type> sorted_delete_log;

	std::set<key_type> all_elems;
	std::vector<key_type> all_elems_sorted;
	// count the number of elements in each block
	// unsigned short count_per_block[num_blocks];
  /*
	// TODO: merge these loops and count the rest in global redistribute
	for (size_t i = 0; i < num_blocks; i++) {
		count_per_block[i] = count_block(i);
	}
  */

	size_t blocks_ptr_1 = header_start;
	size_t cur_block_1 = 0;
	size_t start_of_cur_block_1 = 0;
	while(cur_block_1 < num_blocks) {
		all_elems.insert(blind_read_key(blocks_ptr_1));
		advance_block_ptr(&blocks_ptr_1, &cur_block_1, &start_of_cur_block_1);
	}
	printf("size of all elems in blocks + header: %lu\n",all_elems.size());
	for (size_t i = log_size - num_deletes_in_log; i < log_size; i++) {
		all_elems.erase(blind_read_key(i));
	}
	printf("size of all elems in blocks + header after erasing delete log: %lu\n", all_elems.size());
	for (size_t i = 0; i < num_inserts_in_log; i++) {
		all_elems.insert(blind_read_key(i));
	}
	printf("size of all elems in blocks + header after adding insert log: %lu\n", all_elems.size());

	for (auto e: all_elems) {
		all_elems_sorted.push_back(e);
	}
	std::sort(all_elems_sorted.begin(), all_elems_sorted.end());
	printf("size of all elems sorted in blocks + header: %lu\n", all_elems_sorted.size());
	printf("printing current sorted full leafds: \n");
	for (auto e: all_elems_sorted) {
		printf("\t%lu, ", e);
	}
	printf("\n");

	for (size_t i = 0; i < num_inserts_in_log; i++) {
		sorted_insert_log.push_back(blind_read_key(i));
	}
	for (size_t i = log_size - num_deletes_in_log; i < log_size; i++) {
		sorted_delete_log.push_back(blind_read_key(i));
	}
	std::sort(sorted_insert_log.begin(), sorted_insert_log.end());
	std::sort(sorted_delete_log.begin(), sorted_delete_log.end());

	// two pointer const search for key at index i, need to check for deletes
	size_t log_ptr = 0;
	size_t delete_ptr = 0;
	size_t header_ptr = header_start;
	size_t sorted_block_ptr = 0;
	size_t cur_block = 0;
	size_t curr_index = 0;
	bool in_header = true;

	// on leaving header, initialize this to the current block
	std::vector<key_type> sorted_curr_block;
	unsigned short curr_block_count = 0;

	while ((log_ptr < num_inserts_in_log || cur_block < num_blocks) && curr_index < target_i) {
		// first check if either of the ptrs are pointing to a deleted elt and incr delete ptr
		// check if pointing to insert log
		if (delete_ptr < num_deletes_in_log && log_ptr < num_inserts_in_log && sorted_delete_log[delete_ptr] == sorted_insert_log[log_ptr]) {
			// !!! this actually only occurs if insert was called after delete, meaning it should be inserted
			// because remove() checks insert log and clears it if insert was called first
			// this means we can ignore the delete
			// log_ptr++;
			delete_ptr++;
			continue;
		}
		// check if header_ptr or sorted_block_ptr is pointing to a deleted elem
		if (delete_ptr < num_deletes_in_log && in_header && sorted_delete_log[delete_ptr] == blind_read_key(header_ptr)) {
			// going into a new block
			curr_block_count = count_block(cur_block);
			if (curr_block_count) {
				// block has elems to go into, set the sorted block
				in_header = false;
				for (int i = 0; i < curr_block_count; i++) {
					sorted_curr_block.push_back(blind_read_key(blocks_start + block_size * cur_block + i));
				}
				std::sort(sorted_curr_block.begin(), sorted_curr_block.end());
			} else {
				// empty block, increment header
				in_header = true;
				cur_block++;
				header_ptr++;
			}
			delete_ptr++;
			continue;
		} else if (delete_ptr < num_deletes_in_log && !in_header && sorted_delete_log[delete_ptr] == sorted_curr_block[sorted_block_ptr]) {
			if (sorted_block_ptr == curr_block_count - 1) {
				// we are on last elem in block and need to go back to header
				sorted_curr_block.clear();
				curr_block_count = 0;
				sorted_block_ptr = 0;
				in_header = true;
				cur_block++;
				header_ptr++;
			} else {
				in_header = false;
				sorted_block_ptr++;
			}
			delete_ptr++;
			continue;
		}

		bool increment_block_ptr = false;
		// check which ptr to increment
		if (log_ptr < num_inserts_in_log && cur_block < num_blocks) {
			key_type block_key = in_header ? blind_read_key(header_ptr) : sorted_curr_block[sorted_block_ptr];
			if (sorted_insert_log[log_ptr] < block_key) {
				log_ptr++;
			} else if (sorted_insert_log[log_ptr] == block_key) {
				// on duplicate insert in log and blocks, we want to increment both ptrs
				log_ptr++;
				increment_block_ptr = true;
			} else if (block_key == 0 && cur_block == 0) {
				// edge case of first header block being 0 pre-first flush, we still want to increment log_ptr here
				log_ptr++;
			} else {
				increment_block_ptr = true;
			}
		} else if (log_ptr < num_inserts_in_log) {
			log_ptr++;
		} else {
			increment_block_ptr = true;
		}

		// increment block ptr depending on whether we're in header or block
		if (increment_block_ptr) {
			if (in_header) {
				// going into a new block
				curr_block_count = count_block(cur_block);
				if (curr_block_count) {
					// block has elems to go into, set the sorted block
					in_header = false;
					for (int i = 0; i < curr_block_count; i++) {
						sorted_curr_block.push_back(blind_read_key(blocks_start + block_size * cur_block + i));
					}
					std::sort(sorted_curr_block.begin(), sorted_curr_block.end());
				} else {
					// empty block, increment header
					in_header = true;
					cur_block++;
					header_ptr++;
				}
			} else {
				if (sorted_block_ptr == curr_block_count - 1) {
					// we are on last elem in block and need to go back to header
					sorted_curr_block.clear();
					curr_block_count = 0;
					sorted_block_ptr = 0;
					in_header = true;
					cur_block++;
					header_ptr++;
				} else {
					in_header = false;
					sorted_block_ptr++;
				}
			}
		}
		curr_index++;
	}
	printf("printing current sorted block: \n");
	for (auto e: sorted_curr_block) {
		printf("\t%lu, ", e);
	}
	printf("\n");

	while (delete_ptr < num_deletes_in_log) {
		if (log_ptr < num_inserts_in_log && sorted_delete_log[delete_ptr] == sorted_insert_log[log_ptr]) {
			// !!! this actually only occurs if insert was called after delete, meaning it should be inserted
			// because remove() checks insert log and clears it if insert was called first
			// this means we can ignore the delete
			// log_ptr++;
			delete_ptr++;
			continue;
		}
		// check if header_ptr or sorted_block_ptr is pointing to a deleted elem
		if (in_header && sorted_delete_log[delete_ptr] == blind_read_key(header_ptr)) {
			// going into a new block
			curr_block_count = count_block(cur_block);
			if (curr_block_count) {
				// block has elems to go into, set the sorted block
				in_header = false;
				for (int i = 0; i < curr_block_count; i++) {
					sorted_curr_block.push_back(blind_read_key(blocks_start + block_size * cur_block + i));
				}
				std::sort(sorted_curr_block.begin(), sorted_curr_block.end());
			} else {
				// empty block, increment header
				in_header = true;
				cur_block++;
				header_ptr++;
			}
			delete_ptr++;
			continue;
		} else if (!in_header && sorted_delete_log[delete_ptr] == sorted_curr_block[sorted_block_ptr]) {
			if (sorted_block_ptr == curr_block_count - 1) {
				// we are on last elem in block and need to go back to header
				sorted_curr_block.clear();
				curr_block_count = 0;
				sorted_block_ptr = 0;
				in_header = true;
				cur_block++;
				header_ptr++;
			} else {
				in_header = false;
				sorted_block_ptr++;
			}
			delete_ptr++;
			continue;
		} else {
			break;
		}
	}

	// now we have the correct key to return, but we need to return the pointer to the actual address in leafDS, not a temp copy
	bool in_log;
	key_type target_key = 0;
	key_type block_key = in_header ? blind_read_key(header_ptr) : sorted_curr_block[sorted_block_ptr];

	if (log_ptr < num_inserts_in_log && cur_block < num_blocks) {
		// printf("hi\n");
		if (sorted_insert_log[log_ptr] <= block_key) {
			target_key = sorted_insert_log[log_ptr];
			in_log = true;
		} else if (block_key == 0 && cur_block == 0) {
			// edge case of first header block being 0 pre-first flush
			target_key = sorted_insert_log[log_ptr];
			in_log = true;
		} else {
			target_key = block_key;
			in_log = false;
		}
	} else if (log_ptr < num_inserts_in_log) {
		target_key = sorted_insert_log[log_ptr];
		in_log = true;
	} else {
		target_key = block_key;
		in_log = false;
	}

	// find the address of the actual key in leafDS and return that
	if (in_log) {
		// find key and return address to insert log
		for (size_t i = 0; i < num_inserts_in_log; i++) {
			if (blind_read_key(i) == target_key) {
				return std::get<0>(blind_read(i));
			}
		}
	} else if (in_header) {
		// header should be sorted so we can immediately return
		return std::get<0>(blind_read(header_ptr));
	} else {
		// need to find key in unsorted block and return address
		for (size_t i = blocks_start + cur_block * block_size; i < blocks_start + cur_block * block_size + curr_block_count; i++) {
			if (blind_read_key(i) == target_key) {
				return std::get<0>(blind_read(i));
			}
		}
	}
}


}


